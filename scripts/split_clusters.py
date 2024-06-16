# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import ArgumentParser
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch

from src.utils import setup_logging

from src.dist_comm import (
    enable_distributed,
    get_global_rank,
    get_global_size,
    is_main_process,
    synchronize,
)
from src import distributed_kmeans_gpu as dkmg, kmeans_gpu as kmg


logger = logging.getLogger("hkmeans")


def split_clusters(
    data_path,
    subset_indices_path,
    clusters_path,
    n_splits,
    n_iters,
    dtype,
    high_precision,
    save_path,
    device="cuda",
    use_torchrun=False,
    checkpoint_period=10,
    verbose=False,
):
    enable_distributed(
        use_torchrun=use_torchrun,
        overwrite=True,
    )
    X = np.load(data_path, mmap_mode="r")
    if subset_indices_path is not None:
        logger.info(f"Using subset with indices in {subset_indices_path}")
        subset_indices = np.load(subset_indices_path)
        X = dkmg.ExtendedNumpyMemMap(X, subset_indices)
    clusters = np.load(clusters_path, allow_pickle=True)
    n_clusters = len(clusters)

    part_indices = dkmg.get_part_indices(n_clusters, get_global_size())
    rank = get_global_rank()

    # load checkpoints if exist
    if Path(save_path, f"split_checkpoint_{rank}.npy").exists():
        ckpt = np.load(
            Path(save_path, f"split_checkpoint_{rank}.npy"), allow_pickle=True
        ).item()
        small_centroids = list(ckpt["small_centroids"])
        small_clusters = list(ckpt["small_clusters"])
        last_index = ckpt["last_index"]
        assert last_index - part_indices[rank] + 1 == len(small_centroids)
    else:
        small_centroids = []
        small_clusters = []
        last_index = part_indices[rank] - 1

    # run kmeans++ on clusters
    for cluster_idx in tqdm(
        range(last_index + 1, part_indices[rank + 1]),
        desc="Splitting pre-clusters",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}",
    ):
        if verbose:
            logger.info(f"Processing cluster {cluster_idx}")
        point_indices = np.sort(clusters[cluster_idx])
        if len(point_indices) > 0:
            point_feats = torch.tensor(X[point_indices], device=device, dtype=dtype)
            _small_centroids, _small_clusters, _, _ = kmg.kmeans(
                point_feats,
                min(n_splits, len(point_indices)),
                n_iters,
                chunk_size=-1,
                init_method="kmeans++",
                dist="l2",
                high_precision=high_precision,
            )

            _small_clusters = kmg.sort_cluster_by_distance(
                point_feats,
                _small_centroids,
                _small_clusters,
                device="cuda",
                dtype=dtype,
            )
            _small_clusters = [point_indices[el.astype(int)] for el in _small_clusters]

            non_empty_clusters = [len(el) > 0 for el in _small_clusters]
            _small_clusters = [el for el in _small_clusters if len(el) > 0]
            _small_centroids = _small_centroids[non_empty_clusters]

            small_centroids.append(_small_centroids.cpu().numpy())
            small_clusters += _small_clusters

            del point_feats
        if(
            cluster_idx % checkpoint_period == 0 or
            cluster_idx == part_indices[rank + 1] - 1
        ):
            np.save(
                Path(save_path, f"split_checkpoint_{rank}.npy"),
                {
                    "small_centroids": small_centroids,
                    "small_clusters": small_clusters,
                    "last_index": cluster_idx,
                },
            )
    synchronize()
    logger.info("Gathering clusters")
    if is_main_process():
        centroids = []
        clusters = []
        for i in tqdm(
            range(get_global_size()),
            desc="Gathering splitted clusters",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            split_data = np.load(
                Path(save_path, f"split_checkpoint_{i}.npy"),
                allow_pickle=True
            ).item()
            small_centroids = np.concatenate(split_data["small_centroids"])
            small_clusters = split_data["small_clusters"]
            assert(
                len(small_centroids) == len(small_clusters)
            ), f"Inconsistent shape in split_checkpoint_{i}.npy"
            assert split_data["last_index"] == part_indices[i + 1] - 1
            centroids.append(small_centroids)
            clusters += small_clusters
        centroids = np.concatenate(centroids)
        clusters = np.array(clusters, dtype=object)

        logger.info("Saving centroids and clusters")
        np.save(Path(save_path, "centroids.npy"), centroids)
        np.save(Path(save_path, "sorted_clusters.npy"), clusters)
        logger.info("Cleaning checkpoints")
        for i in range(get_global_size()):
            Path(save_path, f"split_checkpoint_{i}.npy").unlink(missing_ok=True)
    logger.info("Finished split_clusters!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subset_indices_path", type=str, default=None)
    parser.add_argument("--clusters_path", type=str, required=True)
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--n_iters", type=int, required=True)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--high_precision", type=str, default="float32")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--use_torchrun", action="store_true")

    args = parser.parse_args()
    setup_logging()

    def parse_dtype(dtype):
        if dtype == "float32":
            return torch.float32
        elif dtype == "float64":
            return torch.float64
        elif dtype == "float16":
            return torch.float16
        else:
            raise ValueError(f"Value of args.dtype ({args.dtype}) not regconised")

    args.dtype = parse_dtype(args.dtype)
    args.high_precision = parse_dtype(args.high_precision)

    split_clusters(
        args.data_path,
        args.subset_indices_path,
        args.clusters_path,
        args.n_splits,
        args.n_iters,
        args.dtype,
        args.high_precision,
        args.save_path,
        "cuda",
        args.use_torchrun,
    )
