# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import collections
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.utils import check_random_state
from tqdm import tqdm

from . import kmeans_gpu as kmg
from .dist_comm import (
    gather_tensor,
    get_global_rank,
    get_global_size,
    is_main_process,
    synchronize,
)
from .utils import (
    _delete_old_checkpoint,
    get_last_valid_checkpoint,
)


logger = logging.getLogger("hkmeans")

class ExtendedNumpyMemMap(object):
    """
    Class representing an arbitrary slice of a memmap to a numpy array or an array
    """

    def __init__(self, X, indices):
        """
        Parameters:
            X: memmap to a numy array, or an array
            indices: array, indices representing the slice
        """
        if not isinstance(indices, np.ndarray):
            raise ValueError("indices must be a numpy array")
        if indices.ndim != 1:
            raise ValueError("indices must have dimension 1")
        self.X = X
        self.indices = indices
        self.shape = (len(indices), X.shape[1])

    def __getitem__(self, ids):
        return self.X[self.indices[ids]]

    def __len__(self):
        return len(self.indices)

    def numpy(self):
        return np.array(self.X[self.indices])

    def to_tensor(self, dtype, device):
        return torch.tensor(self.numpy(), device=device, dtype=dtype)


def get_part_indices(num_points, world_size):
    """
    Get indices of data points managed by each worker
    """
    return [round(num_points / world_size * i) for i in range(world_size + 1)]


def get_part_len(part_idx, num_points, world_size):
    """
    Get number of data points managed by each worker
    """
    return round(num_points / world_size * (part_idx + 1)) - round(
        num_points / world_size * part_idx
    )


def load_data_to_worker(X, device="cuda", dtype=torch.float32):
    """
    Parameters:
        X: memmap / array or ExtendedNumpyMemMap, the data matrix
        device:
        dtype:

    Returns:
        part of the data allocated to the current worker
    """
    rank = get_global_rank()
    part_indices = get_part_indices(X.shape[0], get_global_size())
    logger.info(f"Rank {rank}: Loading data")
    Xi = torch.tensor(
        np.array(X[part_indices[rank] : part_indices[rank + 1]]),
        device=device,
        dtype=dtype,
    )
    synchronize()
    logger.info(f"Rank: {rank}, X.shape: {X.shape}, Xi.shape: {Xi.shape}")
    return Xi


def distributed_matmul(X, Xi, Y, do_all_gather=False):
    """
    Compute matrix multiplication XY in a distributed manner.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        Y: torch.tensor
            Same on all worker.
        do_all_gather: bool
            Whether to only store the final result in the main
            process (False) or to have a copy of it in all processes (True). In the
            former case, returns None except for the main process.

    Returns:

        Product of X and Y.

    """
    XY = torch.matmul(Xi, Y)
    return gather_tensor(XY, do_all_gather)


def compute_data_squared_norms(X, Xi, do_all_gather=False):
    """
    Compute squared L2 norm of rows of X in a distributed manner.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        do_all_gather: bool
            Whether to only store the final result in the main
            process (False) or to have a copy of it in all processes (True). In the
            former case, returns None except for the main process.

    Returns:

        Squared L2 norm of rows of X

    """
    xi_squared_norms = torch.linalg.vector_norm(Xi, dim=1) ** 2
    return gather_tensor(xi_squared_norms, do_all_gather)


def distributed_squared_euclidean_distance(
    X, Xi, Y, X_squared_norms, do_all_gather=False
):
    """
    Compute squared Euclidean distance between X and Y.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        Y: torch.tensor
            Same on all worker.
        X_squared_norms: torch.tensor of shape (n_samples, )
            Squared L2 norm of rows of X.
        do_all_gather: bool
            Whether to only store the final result in the main
            process (False) or to have a copy of it in all processes (True). In the
            former case, returns None except for the main process.

    Returns:

        Pairwise squared Euclidean distance between rows of X and Y.

    """
    XY = distributed_matmul(X, Xi, Y.T, do_all_gather)
    if do_all_gather:
        Y_squared_norms = torch.linalg.vector_norm(Y, dim=1) ** 2
        XY_dist = X_squared_norms[:, None] - 2 * XY + Y_squared_norms[None, :]
        return XY_dist
    else:
        if is_main_process():
            Y_squared_norms = torch.linalg.vector_norm(Y, dim=1) ** 2
            XY_dist = X_squared_norms[:, None] - 2 * XY + Y_squared_norms[None, :]
            return XY_dist
        else:
            return None


def select_best_candidate(
    X,
    Xi,
    xi_squared_norms,
    candidate_ids,
    closest_dist_sq,
    high_precision=torch.float32,
):
    """
    The selection sub-procedure of kmeans++ initialization.
    Given a list of candidates to select as the next centroid, it find 
    the candidate that would result in the smallest partial kmeans objective.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        candidate_ids: tensor
            List of indices of points to select as the next centroid.
        closest_dist_sq: torch.tensor of shape (n_samples,)
            Squared Euclidean distance to the closest selected centroid.
        high_precision: torch.float32 or torch.float64
            The precision used when high precision is required.

    Returns:

        int, best candidate in candidate_ids
        current_pot: the updated kmeans potential after adding the new centroid
        updated closest_dist_sq
    """

    if high_precision not in [torch.float32, torch.float64]:
        raise ValueError(
            "Only support high_precision value in [torch.float32, torch.float64]"
        )

    part_indices = get_part_indices(X.shape[0], get_global_size())
    rank = get_global_rank()

    # load features of the candidates from X
    Y_candidates = torch.tensor(
        np.array(X[candidate_ids.detach().cpu().numpy()]),
        device=Xi.device,
        dtype=Xi.dtype,
    )
    # compute squared Euclidean distance from candidates to data points
    distance_to_candidates = (
        xi_squared_norms[:, None]
        - 2 * torch.matmul(Xi, Y_candidates.T)
        + torch.linalg.vector_norm(Y_candidates, dim=1)[None, :] ** 2
    )
    distance_to_candidates = distance_to_candidates.type(high_precision).T

    # compute the kmeans potentials if adding each of the candidates
    torch.minimum(
        closest_dist_sq[part_indices[rank] : part_indices[rank + 1]],
        distance_to_candidates,
        out=distance_to_candidates,
    )
    candidates_pot = distance_to_candidates.sum(dim=1)
    dist.all_reduce(candidates_pot, op=dist.ReduceOp.SUM)

    # select the candidate that results in the smallest potential
    best_candidate = torch.argmin(candidates_pot)

    # gather closest_dist_sq
    new_closest_dist_sq = distance_to_candidates[best_candidate].contiguous()
    new_closest_dist_sq = gather_tensor(new_closest_dist_sq, do_all_gather=True)

    # Update potential
    current_pot = new_closest_dist_sq.sum()

    return candidate_ids[best_candidate].item(), current_pot, new_closest_dist_sq


def distributed_kmeans_plusplus_init(
    X,
    Xi,
    n_clusters,
    x_squared_norms,
    random_state=None,
    n_local_trials=None,
    high_precision=torch.float32,
    save_dir=None,
    checkpoint_period=-1,
    max_num_checkpoints=5,
    saving_checkpoint_pattern="kmpp_checkpoint_%d.pth",
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        n_clusters : int
            The number of seeds to choose.
        x_squared_norms : torch.tensor (n_samples,)
            Squared Euclidean norm of each data point.
        random_state : RandomState instance
            The generator used to initialize the centers.
        n_local_trials : int, default=None
            The number of seeding trials for each center (except the first),
            of which the one reducing inertia the most is greedily chosen.
            Set to None to make the number of trials depend logarithmically
            on the number of seeds (2+log(k)); this is the default.
        high_precision: torch.float32 or torch.float64
            The precision used when high precision is required
        save_dir: str or Path
            Location for saving checkpoints.
        checkpoint_period: int
            Save checkpoint after every 'checkpoint_period' iterations, put -1 if do
            not want checkpointing.
        max_num_checkpoints: int
            Maximum number of checkpoints to keep, if exceeded, the oldest checkpoint
            will be deleted.

    Returns

        centers : torch.tensor of shape (n_clusters, n_features)
            The initial centers for k-means.
        indices : ndarray of shape (n_clusters,)
            The index location of the chosen centers in the data array X. For a
            given index and center, X[index] = center.
    """
    if checkpoint_period > 0:
        assert save_dir

    # Common variables in devices
    n_samples, n_features = X.shape
    num_candidates = torch.tensor(0, device=Xi.device)
    candidate_ids = None
    Y_candidates = None

    xi_squared_norms = torch.linalg.vector_norm(Xi, dim=1) ** 2

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    if get_last_valid_checkpoint(save_dir, saving_checkpoint_pattern):
        # Load data from checkpoint if exists
        ckpt_path = get_last_valid_checkpoint(save_dir, saving_checkpoint_pattern)
        logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        begin_iter = ckpt["iter"] + 1
        centers = ckpt["centers"].to(Xi.device)
        pots = ckpt["pots"].to(Xi.device)
        current_pot = ckpt["current_pot"].to(Xi.device)
        closest_dist_sq = ckpt["closest_dist_sq"].to(Xi.device)
        indices = ckpt["indices"]
        random_state = ckpt["random_state"]
    else:
        logger.info("Initializing the first centroid")
        begin_iter = 1
        centers = torch.empty(
            (n_clusters, n_features), dtype=Xi.dtype, device=Xi.device
        )
        pots = torch.empty((n_clusters,), dtype=high_precision, device=Xi.device)
        indices = np.full(n_clusters, -1, dtype=int)

        if random_state is None:
            random_state = check_random_state(random_state)

        if is_main_process():
            # Pick first center randomly and track index of point
            center_id = random_state.randint(n_samples)
            centers[0] = torch.tensor(
                X[center_id], dtype=centers.dtype, device=centers.device
            )
            indices[0] = center_id

            Y_candidates = centers[
                [0]
            ]  # guarantee that Y_candidates have size (1, n_features)

        if not is_main_process():
            Y_candidates = torch.zeros(
                (1, n_features), device=Xi.device, dtype=Xi.dtype
            )
        dist.broadcast(Y_candidates, src=0)
        synchronize()

        # Initialize list of closest distances and calculate current potential
        closest_dist_sq = (
            distributed_squared_euclidean_distance(
                X, Xi, Y_candidates, x_squared_norms, do_all_gather=True
            )
            .ravel()
            .type(high_precision)
        )
        current_pot = closest_dist_sq.sum()
        pots[0] = current_pot

    synchronize()
    logger.info("Begin main loop")
    # Pick the remaining n_clusters-1 points
    if is_main_process():
        iterates = tqdm(
            range(begin_iter, n_clusters),
            desc="Distributed kmeans++ initialization",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(begin_iter, n_clusters)
    for c in iterates:
        if is_main_process():
            # Choose center candidates by sampling with probability proportional
            # to the squared distance to the closest existing center
            rand_vals = (
                torch.tensor(random_state.uniform(size=n_local_trials)).to(
                    current_pot.device
                )
                * current_pot
            )
            candidate_ids = torch.searchsorted(
                torch.cumsum(closest_dist_sq, dim=0), rand_vals
            )
            # numerical imprecision can result in a candidate_id out of range
            torch.clip(
                candidate_ids, None, closest_dist_sq.shape[0] - 1, out=candidate_ids
            )
            num_candidates = torch.tensor(len(candidate_ids), device=Xi.device)

        # broadcast candidate_ids candidates to all processes
        dist.broadcast(num_candidates, src=0)
        synchronize()
        if not is_main_process():
            candidate_ids = torch.zeros(
                (num_candidates,), device=Xi.device, dtype=torch.int64
            )
        dist.broadcast(candidate_ids, src=0)
        synchronize()

        best_candidate, current_pot, closest_dist_sq = select_best_candidate(
            X,
            Xi,
            xi_squared_norms,
            candidate_ids,
            closest_dist_sq,
            high_precision=high_precision,
        )
        pots[c] = current_pot

        if is_main_process():
            # Permanently add best center candidate found in local tries
            centers[c] = torch.tensor(
                X[best_candidate], dtype=Xi.dtype, device=Xi.device
            )
            indices[c] = best_candidate

            if (
                checkpoint_period > 0
                and save_dir
                and ((c + 1) % checkpoint_period == 0 or c + 1 == n_clusters)
            ):
                logger.info("Saving checkpoint to " + saving_checkpoint_pattern % c)
                torch.save(
                    {
                        "centers": centers.cpu(),
                        "indices": indices,
                        "iter": c,
                        "current_pot": current_pot.cpu(),
                        "pots": pots.cpu(),
                        "closest_dist_sq": closest_dist_sq.cpu(),
                        "random_state": random_state,
                    },
                    Path(save_dir, saving_checkpoint_pattern % c),
                    pickle_protocol=4,
                )
                _delete_old_checkpoint(
                    save_dir,
                    c,
                    checkpoint_period,
                    max_num_checkpoints,
                    saving_checkpoint_pattern,
                )

    indices = torch.tensor(indices, device=Xi.device)
    logger.info(f"Kmeans potential of kmeans++ initialization: {current_pot}")
    dist.broadcast(centers, src=0)
    dist.broadcast(indices, src=0)
    synchronize()

    return centers, indices


def distributed_assign_clusters(X, Xi, centroids, chunk_size, verbose=False):
    """
    The assignment sub-procedure of k-means. Given the centroids, assign data points to the index
    of the nearest centroids.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data. Though not used, still put X here to have a consistent function signature.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        centroids: torch.tensor of shape (n_clusters x n_features)
            Centroids of clusters.
        chunk_size: int
            Number of data points that are assigned at once.
            Use a small chunk_size if n_clusters is large to avoid
            out-of-memory error, e.g. chunk_size <= 1e9/n_clusters.
            Default is -1, meaning all data points are assigned at once.
        verbose: bool
            Whether to print progress.

    Returns:

        The assignment of points in X to centroids, each process has a copy of the final result.

    """

    cluster_assignment = kmg.assign_clusters(centroids, Xi, "l2", chunk_size, verbose=verbose)
    cluster_assignment = gather_tensor(cluster_assignment, do_all_gather=True)
    return cluster_assignment


def distributed_compute_centroids(
    X, Xi, n_clusters, centroids, cluster_assignment, high_precision=torch.float32
):
    """
    Compute centroids of each cluster given its data points.

    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data. Though not used, still put X here to have a consistent function signature.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        n_clusters: int
            Number of clusters.
        centroids: torch.tensor of shape (n_clusters, n_features)
            Previous centroids of the clusters.
        cluster_assignment: torch.tensor of shape (n_samples, )
            Cluster id of data points.
        high_precision: torch.float32 or torch.float64, to save GPU memory, one
            can use float32 or float16 for data 'X', 'high_precision' will be
            use in aggregation operation to avoid overflow.

    Returns:

        torch.tensor of shape (n_clusters, n_features), new centroids.

    """
    part_indices = get_part_indices(X.shape[0], get_global_size())
    rank = get_global_rank()
    cluster_assignment_i = cluster_assignment[
        part_indices[rank] : part_indices[rank + 1]
    ]
    clusters_i = kmg.create_clusters_from_cluster_assignment(
        cluster_assignment_i, n_clusters
    )

    in_cluster_sum = torch.zeros(
        (n_clusters, Xi.shape[1]), device=Xi.device, dtype=high_precision
    )
    for _cluster_idx in range(n_clusters):
        if len(clusters_i[_cluster_idx]) > 0:
            in_cluster_sum[_cluster_idx] = torch.sum(
                Xi[clusters_i[_cluster_idx]].type(high_precision), dim=0
            )
    dist.all_reduce(in_cluster_sum, op=dist.ReduceOp.SUM)

    cluster_size = collections.Counter(cluster_assignment)
    for _cluster_idx in range(n_clusters):
        if cluster_size[_cluster_idx] > 0:
            in_cluster_sum[_cluster_idx] = (
                in_cluster_sum[_cluster_idx] / cluster_size[_cluster_idx]
            )
        else:
            in_cluster_sum[_cluster_idx] = centroids[_cluster_idx]
    return in_cluster_sum.type(Xi.dtype)


def distributed_kmeans(
    X,
    Xi,
    n_clusters,
    n_iters=10,
    chunk_size=1000,
    init_method="kmeans++",
    random_state=None,
    save_dir=None,
    save_kmpp_results=True,
    kmpp_checkpoint_period=-1,
    high_precision=torch.float32,
    checkpoint_period=5,
    checkpoint_pattern="centroids_checkpoint_%d.npy",
):
    """
    Parameters:

        X: mem_map of an array of shape (n_samples, n_features) or the array itself
            Data.
        Xi: torch.tensor
            Part of data that is managed by the current device.
        n_clusters: int
            Number of clusters.
        chunk_size: int
            Number of data points that are assigned at once.
            Use a small chunk_size if n_clusters is large to avoid
            out-of-memory error, e.g. chunk_size <= 1e9/n_clusters.
            Default is -1, meaning all data points are assigned at once.
        init_method: str
            'kmeans++' or 'random'
        save_kmpp_results: bool
            Whether to save kmeans++ init results.
        save_dir: str or Path
            Where to save results.

    Returns:

        centroids:
        cluster_assignment: array containing the cluster index of each point.

    """
    assert save_dir or (
        not save_kmpp_results
    ), "provide save_dir to save kmeans++ init results"

    if get_last_valid_checkpoint(save_dir, checkpoint_pattern):
        ckpt_path = get_last_valid_checkpoint(save_dir, checkpoint_pattern)
        logger.info(f"Loading checkpoint from {ckpt_path}")
        begin_iter = int(Path(ckpt_path).stem.split("_")[-1])
        centroids = torch.tensor(np.load(ckpt_path), dtype=Xi.dtype, device=Xi.device)
        cluster_assignment = (
            distributed_assign_clusters(X, Xi, centroids, chunk_size).cpu().numpy()
        )
    else:
        if random_state is None:
            random_state = check_random_state(random_state)
        if init_method == "kmeans++":
            x_squared_norms = compute_data_squared_norms(X, Xi, do_all_gather=True)
            centroids, indices = distributed_kmeans_plusplus_init(
                X,
                Xi,
                n_clusters,
                x_squared_norms,
                save_dir=save_dir,
                high_precision=high_precision,
                checkpoint_period=kmpp_checkpoint_period,
                random_state=random_state,
            )
            if save_kmpp_results:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                np.save(Path(save_dir, "kmpp_centers.npy"), centroids.cpu().numpy())
                np.save(Path(save_dir, "kmpp_indices.npy"), indices.cpu().numpy())

        elif init_method == "random":
            indices = torch.tensor(
                np.sort(random_state.choice(range(len(X)), n_clusters, replace=False)),
                device=Xi.device,
            )
            dist.broadcast(indices, src=0)
            synchronize()
            centroids = torch.tensor(
                X[indices.cpu().numpy()], dtype=Xi.dtype, device=Xi.device
            )
        else:
            raise ValueError(f'Initialization method "{init_method}" not supported!')

        cluster_assignment = (
            distributed_assign_clusters(X, Xi, centroids, chunk_size).cpu().numpy()
        )
        begin_iter = 0

    for _iter in tqdm(
        range(begin_iter, n_iters),
        desc="Distributed kmeans interation",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        centroids = distributed_compute_centroids(
            X,
            Xi,
            n_clusters,
            centroids,
            cluster_assignment,
            high_precision=high_precision,
        )
        cluster_assignment = (
            distributed_assign_clusters(X, Xi, centroids, chunk_size).cpu().numpy()
        )
        if (
            checkpoint_period > 0
            and (_iter + 1) % checkpoint_period == 0
            and is_main_process()
        ):
            logger.info("Saving checkpoint to " + checkpoint_pattern % (_iter + 1))
            np.save(
                Path(save_dir, checkpoint_pattern % (_iter + 1)),
                centroids.cpu().numpy(),
            )
        synchronize()
    return centroids, cluster_assignment


def distributed_sort_cluster_by_distance(
    X,
    centroids,
    clusters,
    device="cuda",
    dtype=torch.float32,
    save_dir=None,
    checkpoint_period=-1,
):
    """
    Parameters:
        X: memory map of an array of shape (n_samples, n_features) or the array itself
            Data.
        centroids: torch.tensor of shape (n_clusters x dim)
            Centroids of clusters.
        clusters: (n_clusters,) array or list
            clusters[i] contains indices of points in cluster i

    Returns:

        sorted_clusters: list
            sorted_clusters[i] contains indices of points in cluster i in increasing order
            from the centroid.
    """

    n_clusters, n_dim = centroids.shape[0], centroids.shape[1]
    part_indices = get_part_indices(n_clusters, get_global_size())
    rank = get_global_rank()

    if checkpoint_period > 0 and Path(
        save_dir,
        f"sorted_clusters_checkpoint_{rank}.npy"
    ).exists():
        cluster_data = np.load(
            Path(
                save_dir,
                f"sorted_clusters_checkpoint_{rank}.npy"
            ),
            allow_pickle=True
        ).item()
        sorted_clusters = cluster_data["sorted_clusters"]
        prev_item = cluster_data["prev_item"]
    else:
        sorted_clusters = []
        prev_item = part_indices[rank] - 1

    for cluster_idx in tqdm(
        range(prev_item + 1, part_indices[rank + 1]),
        desc="Distributed sorting clusters by distance",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        point_indices = np.sort(clusters[cluster_idx])
        point_feats = torch.tensor(X[point_indices], device=device, dtype=dtype)
        _centroid = torch.tensor(
            centroids[cluster_idx],
            device=device,
            dtype=dtype
        ).reshape(1, n_dim)

        dist_to_centroid = torch.cdist(point_feats, _centroid).flatten()
        sorted_clusters.append(
            point_indices[torch.argsort(dist_to_centroid).cpu().numpy()]
        )
        del point_feats

        if(
            (
                checkpoint_period > 0 and
                cluster_idx % checkpoint_period == 0
            ) or
            cluster_idx == part_indices[rank + 1] - 1
        ):
            logger.info(f"Saving checkpoint to {save_dir}/sorted_clusters_checkpoint_{rank}.npy")
            np.save(
                Path(save_dir, f"sorted_clusters_checkpoint_{rank}.npy"),
                {
                    "sorted_clusters": sorted_clusters,
                    "prev_item": cluster_idx
                }
            )
    synchronize()
    if is_main_process():
        logger.info("Gathering clusters")
        sorted_clusters = []
        for i in tqdm(
            range(get_global_size()),
            desc="Distributed gathering sorted clusters",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            rank_data = np.load(
                Path(save_dir, f"sorted_clusters_checkpoint_{i}.npy"),
                allow_pickle=True,
            ).item()
            assert rank_data['prev_item'] == part_indices[i + 1] - 1
            sorted_clusters += rank_data["sorted_clusters"]
        sorted_clusters = np.array(sorted_clusters, dtype=object)
        np.save(
            Path(save_dir, "sorted_clusters.npy"),
            sorted_clusters
        )
        for i in range(get_global_size()):
            Path(save_dir, f"sorted_clusters_checkpoint_{i}.npy").unlink(missing_ok=True)

    synchronize()
    sorted_clusters = np.load(
        Path(save_dir, "sorted_clusters.npy"),
        allow_pickle=True
    )
    return sorted_clusters
