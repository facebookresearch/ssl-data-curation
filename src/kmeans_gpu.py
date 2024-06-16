# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import torch
from tqdm import tqdm

from .utils import create_clusters_from_cluster_assignment
from sklearn.utils import check_random_state


def matmul_transpose(X, Y):
    """
    Compute X . Y.T
    """
    return torch.matmul(X, Y.T)


def compute_distance(
    X, Y, Y_squared_norms, dist="l2", X_squared_norm=None, matmul_fn=matmul_transpose
):
    """
    Compute pairwise distance between rows of X and Y.

    Parameters:
        X: torch.tensor of shape (n_samples_x, n_features)
        Y: torch.tensor of shape (n_samples_y, n_features)
            Y is supposed to be larger than X.
        Y_squared_norms: torch.tensor of shape (n_samples_y, )
            Squared L2 norm of rows of Y.
            It can be  provided to avoid re-computation.
        dist: 'cos' or 'l2'
            If 'cos', assuming that rows of X are normalized
            to have L2 norm equal to 1.
        X_squared_norm: torch.tensor of shape (n_samples_x, )
            Squared L2 norm of rows of X.
        matmul_fn: matmul function.

    Returns:

        Pairwise distance between rows of X and Y.

    """

    if dist == "cos":
        return 2 - 2 * matmul_fn(X, Y)
    elif dist == "l2":
        if X_squared_norm is None:
            X_squared_norm = torch.linalg.vector_norm(X, dim=1) ** 2
        return X_squared_norm[:, None] - 2 * matmul_fn(X, Y) + Y_squared_norms[None, :]
    else:
        raise ValueError(f'dist = "{dist}" not supported!')


# A modified version of _kmeans_plusplus
# from https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/cluster/_kmeans.py#L63
def kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    dist,
    random_state=None,
    n_local_trials=None,
    high_precision=torch.float64,
    verbose=False,
):
    """
    Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
        X : torch.tensor of shape (n_samples, n_features)
            The data to pick seeds for.
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
        high_precision: torch.float32 or torch.float64, to save GPU memory, one
            can use float32 or float16 for data 'X', 'high_precision' will be
            use in aggregation operation to avoid overflow.

    Returns
        centers : torch.tensor of shape (n_clusters, n_features)
            The initial centers for k-means.
        indices : ndarray of shape (n_clusters,)
            The index location of the chosen centers in the data array X. For a
            given index and center, X[index] = center.

    """
    if random_state is None:
        random_state = check_random_state(random_state)

    n_samples, n_features = X.shape

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype).to(X.device)
    pots = torch.empty((n_clusters,), device=X.device, dtype=high_precision)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = compute_distance(X[center_id, None], X, x_squared_norms, dist)[
        0
    ].type(high_precision)
    current_pot = closest_dist_sq.sum()
    pots[0] = current_pot

    # Pick the remaining n_clusters-1 points
    if verbose:
        iterates = tqdm(
            range(1, n_clusters),
            desc="Kmeans++ initialization",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(1, n_clusters)
    for c in iterates:
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
        torch.clip(candidate_ids, None, closest_dist_sq.shape[0] - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = compute_distance(
            X[candidate_ids], X, x_squared_norms, dist
        ).type(high_precision)

        # update closest distances squared and potential for each candidate
        torch.minimum(
            closest_dist_sq, distance_to_candidates, out=distance_to_candidates
        )
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate
        pots[c] = current_pot

    return centers, indices


def assign_clusters(centroids, X, dist, chunk_size=-1, verbose=False):
    """
    Assign data points to their closest clusters.

    Parameters:

        centroids: torch.tensor of shape (n_clusters, n_features)
            Centroids of the clusters.
        X: torch.tensor of shape (n_samples, n_features)
            Data.
        dist: 'cos' or 'l2'
            If 'cos', assuming that rows of X are normalized
            to have L2 norm equal to 1.
        chunk_size: int
            Number of data points that are assigned at once.
            Use a small chunk_size if n_clusters is large to avoid
            out-of-memory error, e.g. chunk_size <= 1e9/n_clusters.
            Default is -1, meaning all data points are assigned at once.
        verbose: bool
            Whether to print progress bar.

    Returns:

        torch.tensor of shape (n_samples, ) containing the cluster id of
        each data point.

    """

    cluster_ids = []
    n_samples, _ = X.shape
    x_squared_norms = torch.linalg.vector_norm(X, dim=1) ** 2
    centroid_squared_norm = torch.linalg.vector_norm(centroids, dim=1) ** 2
    if chunk_size < 0:
        try:
            distance_from_centroids = compute_distance(
                centroids, X, x_squared_norms, dist, centroid_squared_norm
            )
        except Exception as e:
            raise MemoryError(
                f"matrices are too large, consider setting chunk_size ({chunk_size}) to a smaller number"
            ) from e
        cluster_ids = torch.argmin(distance_from_centroids, dim=0)
    else:
        n_iters = (n_samples + chunk_size - 1) // chunk_size
        if verbose:
            iterates = tqdm(
                range(n_iters),
                desc="Assigning data points to centroids",
                file=sys.stdout,
                bar_format="{l_bar}{bar}{r_bar}",
            )
        else:
            iterates = range(n_iters)

        for chunk_idx in iterates:
            begin_idx = chunk_idx * chunk_size
            end_idx = min(n_samples, (chunk_idx + 1) * chunk_size)
            distance_from_centroids = compute_distance(
                centroids,
                X[begin_idx:end_idx],
                x_squared_norms[begin_idx:end_idx],
                dist,
                centroid_squared_norm,
            )
            cluster_ids.append(torch.argmin(distance_from_centroids, dim=0))
            del distance_from_centroids
        cluster_ids = torch.cat(cluster_ids)
    return cluster_ids


def compute_centroids(
    centroids, cluster_assignment, n_clusters, X, high_precision=torch.float32
):
    """
    Compute centroids of each cluster given its data points.

    Parameters:

        centroids: torch.tensor of shape (n_clusters, n_features)
            Previous centroids of the clusters.
        cluster_assignment: torch.tensor of shape (n_samples, )
            Cluster id of data points.
        n_clusters: int
            Number of clusters.
        X: torch.tensor of shape (n_samples, n_features)
            Data.
        high_precision: torch.float32 or torch.float64, to save GPU memory, one
            can use float32 or float16 for data 'X', 'high_precision' will be
            use in aggregation operation to avoid overflow.

    Returns:

        torch.tensor of shape (n_clusters, n_features), new centroids
    """
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    new_centroids = torch.zeros_like(centroids)
    for i in range(n_clusters):
        if len(clusters[i]) > 0:
            new_centroids[i] = torch.mean(
                X[clusters[i].astype(int)].type(high_precision), dim=0
            )
        else:
            new_centroids[i] = centroids[i]
    return new_centroids


def _kmeans(
    X,
    n_clusters,
    n_iters,
    chunk_size=-1,
    init_method="kmeans++",
    dist="l2",
    high_precision=torch.float32,
    random_state=None,
    verbose=False,
):
    """
    Run kmeans once.

    Parameters: See above.

    Returns:

        centroids:
        clusters: np.array of np.array
            Indices of points in each cluster. A subarray corresponds to a cluster.
        cluster_assignment:
        pot: float, kmeans objective

    """
    if random_state is None:
        random_state = check_random_state(random_state)

    x_squared_norms = torch.linalg.vector_norm(X, dim=1) ** 2
    if init_method == "kmeans++":
        centroids, _ = kmeans_plusplus(
            X,
            n_clusters,
            x_squared_norms,
            dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
    else:
        centroids = torch.tensor(
            X[np.sort(random_state.choice(range(len(X)), n_clusters, replace=False))],
            device=X.device,
            dtype=X.dtype,
        )

    cluster_assignment = assign_clusters(centroids, X, dist, chunk_size).cpu().numpy()
    for _iter in range(n_iters):
        centroids = compute_centroids(
            centroids, cluster_assignment, n_clusters, X, high_precision
        )
        cluster_assignment = (
            assign_clusters(centroids, X, dist, chunk_size).cpu().numpy()
        )
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, n_clusters)
    pot = np.sum(
        [
            torch.sum(
                torch.cdist(
                    X[el.astype(int)], X[el.astype(int)].mean(dim=0, keepdim=True)
                )
                ** 2
            ).item()
            for el in clusters
        ]
    )
    return centroids, clusters, cluster_assignment, pot


def kmeans(
    X,
    n_clusters,
    n_iters,
    chunk_size=-1,
    num_init=10,
    init_method="kmeans++",
    dist="l2",
    high_precision=torch.float32,
    random_state=None,
    verbose=False,
):
    """
    Run kmeans multiple times and return the clustering with the best objective.

    Parameters: See above and

        num_init: int
            Number of kmeans runs.

    Returns:

        Same as _kmeans

    """

    n_clusters = min(X.shape[0], n_clusters)
    best_centroids, best_clusters, best_cluster_assignment, best_pot = (
        None,
        None,
        None,
        np.Inf,
    )
    for _ in range(num_init):
        centroids, clusters, cluster_assignment, pot = _kmeans(
            X,
            n_clusters,
            n_iters,
            chunk_size=chunk_size,
            init_method=init_method,
            dist=dist,
            high_precision=high_precision,
            random_state=random_state,
            verbose=verbose,
        )
        if pot < best_pot:
            best_centroids, best_clusters, best_cluster_assignment, best_pot = (
                centroids,
                clusters,
                cluster_assignment,
                pot,
            )
    return best_centroids, best_clusters, best_cluster_assignment, best_pot


def sort_cluster_by_distance(
    X, centroids, clusters, device="cuda", dtype=torch.float32, verbose=False,
):
    """
    Sort data points in each cluster in increasing order of distance to the centroid.

    Parameters:

        X: data
        centroids:
        clusters:

    Returns:

        sorted_clusters: np.array of np.array
            Indices of points in each cluster. A subarray corresponds to a cluster.

    """

    n_clusters, n_dim = centroids.shape[0], centroids.shape[1]

    sorted_clusters = []
    if verbose:
        iterates = tqdm(
            range(n_clusters),
            desc="Sorting clusters by distance",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(n_clusters)
    for cluster_idx in iterates:
        if len(clusters[cluster_idx]) > 0:
            point_indices = np.sort(clusters[cluster_idx]).astype(int)
            point_feats = torch.tensor(X[point_indices], device=device, dtype=dtype)
            _centroid = centroids[cluster_idx].reshape(1, n_dim).type(dtype)

            dist_to_centroid = torch.cdist(point_feats, _centroid).flatten()
            sorted_clusters.append(
                point_indices[torch.argsort(dist_to_centroid).cpu().numpy()]
            )
            del point_feats
        else:
            sorted_clusters.append(np.array([]).astype(int))
    return np.array(sorted_clusters, dtype=object)
