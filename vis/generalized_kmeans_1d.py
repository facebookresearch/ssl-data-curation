# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys 

import numpy as np
import torch
from tqdm import tqdm
from sklearn.utils import check_random_state

from src import kmeans_gpu as kmg
from src.utils import create_clusters_from_cluster_assignment


def l2_squared_power(x, xi, n):
    """
    Compute L_2 ^ (2 * n) distance
    """
    return (x - xi) ** (2 * n)


def l2_squared_power_der(x, xi, n):
    """
    Compute the derivative of L_2 ^ (2 * n) distance
    """
    return 2 * n * (x - xi) ** (2 * n - 1)


def l2_squared_power_der2(x, xi, n):
    """
    Compute second-order derivative of L_2 ^ (2 * n) distance
    """
    return 2 * n * (2 * n - 1) * (x - xi) ** (2 * n - 2)


def kmeans_plusplus(
    X,
    n_clusters,
    x_squared_norms,
    dist,
    power=1,
    random_state=None,
    n_local_trials=None,
    save_running_results=False,
    high_precision=torch.float32,
    verbose=False,
):
    """
    Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : torch.tensor of shape (n_samples, n_features)
        The data to pick seeds for.
    n_clusters : int
        The number of seeds to choose.
    x_squared_norms : torch.tensor (n_samples,)
        Squared Euclidean norm of each data point.
    dist: str
        Type of distance function. Options are "l2" or "cos".
    power: int
        Distance is L_2 ^ (2 * power).
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    save_running_results: bool, default=False
        Whether to save temporary results during execution.
    high_precision: torch.Type
        type for high-precision computations.
    verbose: bool, default=False

    Returns
    -------
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
    closest_dist_sq = (
        kmg.compute_distance(X[center_id, None], X, x_squared_norms, dist)[0].type(
            high_precision
        )
        ** power
    )
    current_pot = closest_dist_sq.sum()
    pots[0] = current_pot

    # Pick the remaining n_clusters-1 points
    if verbose:
        iterates = tqdm(
            range(1, n_clusters),
            desc="Genralized kmeans++ initialization",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        )
    else:
        iterates = range(1, n_clusters)
    for c in iterates:
        # Choose center candidates by sampling with probability proportional
        # to the distance to the closest existing center
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
        distance_to_candidates = (
            kmg.compute_distance(X[candidate_ids], X, x_squared_norms, dist).type(
                high_precision
            )
            ** power
        )

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

        if save_running_results and c % 1000 == 0:
            np.save(
                "kmpp_running_results.npy",
                {"centers": centers.cpu().numpy(), "indices": indices, "iter": c},
            )

    return centers, indices


def compute_centroids(X, n, n_iters=5, method="newton", verbose=False):
    """
    Compute k-means centroids given a set of points, according to distortion
    function L_2 ^ (2 * n), with Newton method.
    """
    if method == "newton":
        # Initialize the centroid with L_2^2 means.
        c = X.mean()
        if len(X) == 1:
            return c
        for _ in range(n_iters):
            if verbose:
                f = torch.sum(l2_squared_power(c, X, n))
                print(f, end=", ")
            der_f = torch.sum(l2_squared_power_der(c, X, n))
            der2_f = torch.sum(l2_squared_power_der2(c, X, n))
            if der_f == 0:
                break
            c -= der_f / der2_f
        return c
    else:
        raise ValueError("Method not supported!")


def assign_clusters(X, centers, chunk_size=-1):
    """
    Assign points to centroids.
    """
    cluster_assignment = (
        kmg.assign_clusters(centers, X, "l2", chunk_size=chunk_size, verbose=False)
        .cpu()
        .numpy()
    )
    clusters = create_clusters_from_cluster_assignment(cluster_assignment, len(centers))
    return clusters


def update_centroids(X, clusters, n):
    """
    Update centroids based on the new clusters after reassignment.
    """
    n_clusters = len(clusters)
    centers = torch.zeros((n_clusters, 1), device=X.device, dtype=X.dtype)
    for cid in range(n_clusters):
        if len(clusters[cid]) > 0:
            centers[cid, 0] = compute_centroids(X[clusters[cid]], n).item()
    return centers


def generalized_kmeans_1d(
    X, n_clusters, n, n_iters=50, init_method="k-means++", chunk_size=-1
):
    """
    Run generalized k-means with distance L_2 ^ (2 * n)
    """
    assert X.ndim == 2
    # initialize
    if init_method == "k-means++":
        x_squared_norms = torch.linalg.vector_norm(X, dim=1) ** 2
        centers, _ = kmeans_plusplus(X, n_clusters, x_squared_norms, "l2", n)
    else:
        centers = X[np.random.choice(len(X), n_clusters, replace=False), :]
    clusters = assign_clusters(X, centers, chunk_size=chunk_size)
    for _ in tqdm(
        range(n_iters),
        desc="Generalized kmeans iterations",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        centers = update_centroids(X, clusters, n)
        clusters = assign_clusters(X, centers, chunk_size=chunk_size)
    return centers, clusters
