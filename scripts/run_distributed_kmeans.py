# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import subprocess

import numpy as np
import torch

from src import (
    distributed_kmeans_gpu as dkmg,
    kmeans_gpu as kmg,
    hierarchical_sampling as hs,
)
from src.dist_comm import enable_distributed, is_main_process, synchronize
from src.utils import get_last_valid_checkpoint, setup_logging


logger = logging.getLogger("hkmeans")

def check_and_load_npy(load_path, allow_pickle=False, data_name=None):
    if load_path.exists():
        if data_name is not None:
            logger.info(f"Loading {data_name} from {str(load_path)}")
        else:
            logger.info(f"Loading from {str(load_path)}")
        data = np.load(load_path, allow_pickle=allow_pickle)
        return data
    else:
        return None


def check_and_save(save_path, save_data):
    if is_main_process():
        if not save_path.exists():
            np.save(save_path, save_data)
    synchronize()


def main(args):
    enable_distributed(
        use_torchrun=args.use_torchrun,
        overwrite=True,
    )

    X_ori = np.load(args.data_path, mmap_mode="r")
    if args.subset_indices_path is not None:
        logger.info(f"Using subset with indices in {args.subset_indices_path}")
        subset_indices = np.load(args.subset_indices_path)
        X_ori = dkmg.ExtendedNumpyMemMap(X_ori, subset_indices)
    Xi_ori = dkmg.load_data_to_worker(X_ori, dtype=args.dtype)

    for step_id in range(args.n_steps):
        step_dir = Path(args.exp_dir, f"step{step_id}")
        sorted_clusters_path = Path(step_dir, "sorted_clusters.npy")
        if sorted_clusters_path.exists():
            logger.info(
                f"Step {step_id}: sorted clusters exist ({sorted_clusters_path}), skipping"
            )
            continue
        logger.info(f"Running step {step_id}")
        step_dir.mkdir(exist_ok=True)

        # Load resampled points and run kmeans
        if step_id == 0:
            X = X_ori
            Xi = Xi_ori
        else:
            if is_main_process():
                if not Path(step_dir, "sampled_indices.npy").exists():
                    logger.info(f"Sampling points for step {step_id}")
                    prev_sorted_clusters = np.load(
                        Path(args.exp_dir, f"step{step_id-1}", "sorted_clusters.npy"),
                        allow_pickle=True,
                    )
                    logger.info(
                        f"Sampling from {len(prev_sorted_clusters)} clusters using "
                        f"'{args.sampling_strategy}' sampling strategy, "
                        f"{args.sample_size} samples per cluster"
                    )
                    if args.sampling_strategy == "c":
                        sampler = hs.closest_to_centroid_selection
                    elif args.sampling_strategy == "r":
                        sampler = hs.random_selection
                    else:
                        raise ValueError(
                            f"sampling_strategy={args.sampling_strategy} not recognized!"
                        )
                    sampled_indices = sampler(
                        prev_sorted_clusters,
                        list(range(len(prev_sorted_clusters))),
                        args.sample_size,
                    )
                    sampled_indices = np.sort(sampled_indices)
                    logger.info(f"Selected {len(sampled_indices)} images")
                    np.save(Path(step_dir, "sampled_indices.npy"), sampled_indices)
                else:
                    logger.info(
                        f"sampled_indices.npy exists at "
                        f"{str(Path(step_dir, 'sampled_indices.npy'))}"
                    )
            synchronize()
            sampled_indices = np.load(Path(step_dir, "sampled_indices.npy"))
            X = dkmg.ExtendedNumpyMemMap(X_ori, indices=sampled_indices)
            Xi = dkmg.load_data_to_worker(X, dtype=args.dtype)

        # Compute centroids
        centroids = check_and_load_npy(
            Path(step_dir, "centroids.npy"),
            allow_pickle=False,
            data_name="centroids"
        )
        if centroids is not None:
            centroids = torch.tensor(centroids, device="cuda", dtype=args.dtype)
            synchronize()
        else:
            logger.info("Begin distributed kmeans")
            centroids, _ = dkmg.distributed_kmeans(
                X,
                Xi,
                args.n_clusters,
                n_iters=args.n_iters,
                chunk_size=args.chunk_size,
                init_method="kmeans++",
                save_kmpp_results=True,
                save_dir=step_dir,
                kmpp_checkpoint_period=args.checkpoint_period,
                high_precision=args.high_precision,
            )
        check_and_save(
            Path(step_dir, "centroids.npy"),
            centroids.cpu().numpy()
        )

        # Compute cluster_assignment
        cluster_assignment = check_and_load_npy(
            Path(step_dir, "cluster_assignment.npy"),
            allow_pickle=False,
            data_name="cluster_assignment"
        )
        if cluster_assignment is None:
            logger.info("Assign points to clusters")
            cluster_assignment = (
                dkmg.distributed_assign_clusters(
                    X_ori,
                    Xi_ori,
                    centroids,
                    args.chunk_size,
                    verbose=True
                )
                .cpu()
                .numpy()
            )
        check_and_save(
            Path(step_dir, "cluster_assignment.npy"),
            cluster_assignment
        )

        # Compute clusters
        clusters = check_and_load_npy(
            Path(step_dir, "clusters.npy"),
            allow_pickle=True,
            data_name="clusters"
        )
        if clusters is None:
            logger.info("Create clusters from cluster_assignment")
            clusters = kmg.create_clusters_from_cluster_assignment(
                cluster_assignment, args.n_clusters
            )
        check_and_save(
            Path(step_dir, "clusters.npy"),
            clusters
        )

        if not Path(step_dir, "sorted_clusters.npy").exists():
            centroids = centroids.cpu().numpy()
            del X, Xi
            if step_id == args.n_steps - 1:
                del Xi_ori
            torch.cuda.empty_cache()
            logger.info("Sort points in each cluster by distance to centroid")
            _ = dkmg.distributed_sort_cluster_by_distance(
                X_ori,
                centroids,
                clusters,
                dtype=torch.float32,
                save_dir=step_dir,
                checkpoint_period=args.sort_cluster_checkpoint_period,
            )

        # remove checkpoints
        if is_main_process():
            while get_last_valid_checkpoint(step_dir, "kmpp_checkpoint_%d.pth"):
                get_last_valid_checkpoint(step_dir, "kmpp_checkpoint_%d.pth").unlink(
                    missing_ok=True
                )
            while get_last_valid_checkpoint(step_dir, "centroids_checkpoint_%d.npy"):
                get_last_valid_checkpoint(
                    step_dir, "centroids_checkpoint_%d.npy"
                ).unlink(missing_ok=True)

    if is_main_process():
        last_sorted_clusters_path = str(
            Path(args.exp_dir, f"step{args.n_steps-1}", "sorted_clusters.npy").resolve()
        )
        last_centroids_path = str(
            Path(args.exp_dir, f"step{args.n_steps-1}", "centroids.npy").resolve()
        )

        link_command = f'ln -s {last_sorted_clusters_path} {str(Path(args.exp_dir, "sorted_clusters.npy").resolve())}'
        process = subprocess.Popen(link_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

        link_command = f'ln -s {last_centroids_path} {str(Path(args.exp_dir, "centroids.npy").resolve())}'
        process = subprocess.Popen(link_command.split(), stdout=subprocess.PIPE)
        _, _ = process.communicate()

    logger.info("Finished all steps!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--subset_indices_path", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--high_precision", type=str, default="float32")
    parser.add_argument("--checkpoint_period", type=int, default=1000)
    parser.add_argument(
        "--sort_cluster_checkpoint_period",
        type=int,
        default=-1
    )
    parser.add_argument("--exp_dir", type=str, default="tmp")
    parser.add_argument("--n_iters", type=int, default=10)
    parser.add_argument("--use_torchrun", action="store_true")

    parser.add_argument(
        "--n_steps", type=int, default=1, help="Number of resampling step"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Number of samples per cluster in resampling",
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="c",
        help="resampling with closest (c) or random (r) strategy",
    )

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
    if args.dtype == torch.float64:
        args.high_precision = torch.float64
    assert args.high_precision in [torch.float32, torch.float64]

    logger.info(f"Args: {args}")

    main(args)
    synchronize()
