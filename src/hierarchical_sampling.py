# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import random

import numpy as np
from tqdm import tqdm

from src.clusters import HierarchicalCluster


logger = logging.getLogger("hkmeans")

def random_selection(clusters, valid_clusters, num_per_cluster):
    """
    Parameters:
        clusters: (num_cluster, ) np.array
            clusters[i] contain indices of points in cluster i
        valid_clusters: list or np.array
            indices of clusters that are considered
        num_per_cluster: int
            number of points selected from each cluster

    Returns:
        array containing indices of selected points
    """
    num_clusters = len(clusters)
    selected = [[]] * num_clusters
    for cluster_id in tqdm(
        valid_clusters,
        desc="Random sampling from clusters",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        selected[cluster_id] = random.sample(
            list(clusters[cluster_id]), min(num_per_cluster, len(clusters[cluster_id]))
        )
    return np.concatenate(selected).astype(np.int64)


def closest_to_centroid_selection(sorted_clusters, valid_clusters, num_per_cluster):
    """
    Parameters:
        sorted_clusters: (num_cluster, ) np.array
            clusters[i] contain indices of points in cluster i
            indices in clusters[i] are sorted in increasing distance from the centroid i
        valid_clusters: list or np.array
            indices of clusters that are considered
        num_per_cluster: int, number of points selected from each cluster

    Returns:
        array containing indices of selected points
    """
    num_clusters = len(sorted_clusters)
    selected = [[]] * num_clusters
    for cluster_id in tqdm(
        valid_clusters,
        desc="Closest-to-centroid sampling from clusters",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        selected[cluster_id] = sorted_clusters[cluster_id][:num_per_cluster]
    return np.concatenate(selected).astype(np.int64)


def _find_best_cut_left(arr, target):
    """
    Find integers x such that sum(min(x, arr)) best approximates target
    """
    if target < 0:
        raise ValueError(f"target {target} must be non-negative!")
    if np.min(arr) < 0:
        raise ValueError("arr has negative elements!")
    if np.sum(arr) <= target:
        return np.max(arr)
    left = 0
    right = np.max(arr)
    while right - left > 1:
        mid = (left + right) // 2
        sum_with_mid = np.sum(np.minimum(mid, arr))
        if sum_with_mid > target:
            right = mid
        elif sum_with_mid < target:
            left = mid
        else:
            return mid
    if np.sum(np.minimum(right, arr)) <= target:
        return right
    return left


def find_subcluster_target_size(
    subcluster_sizes,
    target_size,
    multiplier,
):
    """
    Given the target number of points to sample from a clusters,
    find number of points to sample from its subclusters.
    """
    if isinstance(subcluster_sizes, np.ndarray):
        arr = subcluster_sizes * multiplier
    else:
        arr = np.array(subcluster_sizes) * multiplier
    best_cut_left =  _find_best_cut_left(arr, target_size)
    if best_cut_left == np.max(arr):
        return arr
    else:
        subcluster_target_sizes = np.minimum(best_cut_left, arr)
        remainder = target_size - subcluster_target_sizes.sum()
        candidates = np.where(arr > best_cut_left)[0]
        subcluster_target_sizes[np.random.choice(candidates, remainder, replace=False)] = best_cut_left + 1
        assert subcluster_target_sizes.sum() == target_size
        assert np.all(subcluster_target_sizes <= arr)
        return subcluster_target_sizes


def recursive_hierarchical_sampling(
    clusters: HierarchicalCluster,
    level: int,
    target_size: int,
    cl_index: int,
    multiplier: int,
    sampling_strategy: str = "r",
):
    """
    Given a target number of points to sample from a cluster, return
    the a set of sampled points.
    """
    if level == 1:
        current_cluster = clusters.clusters[1][cl_index]
        current_cluster_size = clusters.clusters_size[1][cl_index]
        if current_cluster_size * multiplier <= target_size:
            return np.tile(current_cluster, multiplier)
        else:
            n_replicates = target_size // current_cluster_size
            replicates = np.tile(current_cluster, n_replicates)
            remaining_target = target_size - n_replicates * current_cluster_size
            if sampling_strategy == "r":  # random
                remaining_samples = np.random.choice(
                    current_cluster,
                    remaining_target,
                    replace=False,
                )
            elif sampling_strategy == "c":  # "closest"
                remaining_samples = current_cluster[:remaining_target]
            else:
                raise ValueError(f"sampling_strategy={sampling_strategy} is not supported")
            return np.concatenate([replicates, remaining_samples])
    else:
        subcl_indices = clusters.clusters[level][cl_index]
        subcluster_sizes = clusters.flat_clusters_size[level - 1][subcl_indices]
        subcluster_target_sizes = find_subcluster_target_size(
            subcluster_sizes,
            target_size,
            multiplier,
        )
        samples = []
        for i, subcl_index in enumerate(subcl_indices):
            samples.append(
                recursive_hierarchical_sampling(
                    clusters,
                    level - 1,
                    subcluster_target_sizes[i],
                    subcl_index,
                    multiplier,
                    sampling_strategy,
                )
            )
        return np.concatenate(samples)


def hierarchical_sampling(
    clusters: HierarchicalCluster,
    target_size: int,
    multiplier: int = 1,
    sampling_strategy: str = "r",
):
    """
    Method for sample hierarchically from a hierarchy of clusters.
    """
    if (not clusters.is_loaded) or (not clusters.is_processed):
        raise RuntimeError("HierarchicalCluster is not loaded or processed.")
    n_levels = clusters.n_levels
    cluster_target_sizes = find_subcluster_target_size(
        clusters.flat_clusters_size[n_levels],
        target_size,
        multiplier,
    )
    samples = []
    for cl_index in tqdm(
        range(len(clusters.clusters[n_levels])),
        desc="Hierarchical sampling from clusters",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        samples.append(
            recursive_hierarchical_sampling(
                clusters,
                n_levels,
                cluster_target_sizes[cl_index],
                cl_index,
                multiplier,
                sampling_strategy,
            )
        )
    samples = np.concatenate(samples)
    return samples
