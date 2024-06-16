# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
import pickle
from typing import Dict, List

import numpy as np


logger = logging.getLogger("hkmeans")


def load_clusters_from_file(fpath):
    """
    Utility to load clusters fromj different file formats.
    """
    if Path(fpath).suffix == ".pkl":
        with open(fpath, "rb") as f:
            return np.array(pickle.load(f), dtype=object)
    else:
        return np.load(Path(fpath), allow_pickle=True)

class HierarchicalCluster:
    """
    Class representing a hierarchy of clusters returned by hierarchical k-means.
    """
    def __init__(self):
        self.cluster_path = None
        self.n_levels = None
        self.cluster_fname = None
        self.is_loaded = False
        self.is_processed = False
        self.n_clusters = {}
        self.clusters = {}
        self.flat_clusters = {}
        self.clusters_size = {}
        self.flat_clusters_size = {}
        self.size_order = {}
        self.flat_size_order = {}

    def load_clusters_from_file(self):
        for level in range(1, 1 + self.n_levels):
            self.clusters[level] = load_clusters_from_file(
                Path(
                    self.cluster_path,
                    f"level{level}",
                    self.cluster_fname
                )
            )
            self.n_clusters[level] = len(self.clusters[level])
        self.is_loaded = True

    def process_clusters(self):
        if not self.is_loaded:
            raise RuntimeError("Clusters must be loaded before being processed")
        logger.info("Computing flat clusters")
        self.flat_clusters[1] = self.clusters[1]
        for level in range(2, 1 + self.n_levels):
            current_non_flat = self.clusters[level]
            prev_flat = self.flat_clusters[level - 1]
            self.flat_clusters[level] = np.array(
                [
                    np.concatenate([prev_flat[el] for el in clus])
                    if len(clus) > 0 else np.array([])
                    for clus in current_non_flat
                ],
                dtype=object,
            )

        logger.info("Computing cluster length")
        for level, clus in self.clusters.items():
            self.clusters_size[level] = np.array([len(el) for el in clus])

        for level, clus in self.flat_clusters.items():
            self.flat_clusters_size[level] = np.array([len(el) for el in clus])

        logger.info("Sorting clusters by length")
        for level, clsize in self.clusters_size.items():
            self.size_order[level] = np.argsort(clsize)[::-1]

        for level, flat_clsize in self.flat_clusters_size.items():
            self.flat_size_order[level] = np.argsort(flat_clsize)[::-1]

        self.is_processed = True

    @staticmethod
    def from_file(
        cluster_path,
        cluster_fname="sorted_clusters.npy",
    ):
        """
        Method for reading hierarchical clusters from files
        """
        logger.info("Loading hierarchical clusters from file.")
        cl = HierarchicalCluster()
        cl.cluster_path = cluster_path
        cl.cluster_fname = cluster_fname
        cl.n_levels = 0
        while True:
            if Path(cl.cluster_path, f"level{cl.n_levels + 1}").exists():
                cl.n_levels += 1
            else:
                break
        cl.load_clusters_from_file()
        cl.process_clusters()
        return cl

    @staticmethod
    def from_dict(clusters: List[Dict]):
        """
        Read hierarchical clusters from a list of dictionaries.

        Parameters:
            clusters: List[Dict]
                Each element is a dictionary containing a field name "clusters".
                An example is the output of hierarchical_kmeans_gpu.hierarchical_kmeans

        Return:
            A instance of HierarchicalCluster.
        """
        logger.info("Loading hierarchical clusters from dictionaries.")
        cl = HierarchicalCluster()
        cl.n_levels = len(clusters)
        for level in range(1, 1 + cl.n_levels):
            cl.clusters[level] = clusters[level - 1]["clusters"]
            cl.n_clusters[level] = len(cl.clusters[level])
        cl.is_loaded = True
        cl.process_clusters()
        return cl
