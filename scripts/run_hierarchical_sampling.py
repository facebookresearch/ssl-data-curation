# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from src.clusters import HierarchicalCluster
from src.utils import setup_logging
from src.hierarchical_sampling import hierarchical_sampling

logger = logging.getLogger("hkmeans")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--clustering_path", "-clus", type=str, required=True)
    parser.add_argument(
        "--target_size",
        type=int,
        required=True,
        help="Target size of the sampled set"
    )
    parser.add_argument(
        "--multiplier",
        "-m",
        type=int,
        default=1,
        help="Maximum number of times an image is selected"
    )
    parser.add_argument(
        "--sampling_strategy",
        "-ss",
        type=str,
        default="r",
        help='"r" for random, "c" for closest',
    )
    parser.add_argument(
        "--sort_indices",
        action="store_true",
        help="If true, sort indices in increasing order",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Suffix to add to the indice file name",
    )
    parser.add_argument(
        "--valid_indices_path",
        type=str,
        default=None,
        help=(
            "Path to .npy file containing valid indices of the base dataset. "
            "The clustering is computed only on these valid images."
        ),
    )
    parser.add_argument(
        "--cluster_fname",
        type=str,
        default="sorted_clusters.npy",
        help="name of files containing clusters",
    )
    parser.add_argument("--save_dir_name", type=str, default="curated_datasets")

    args = parser.parse_args()
    args.clustering_path = Path(args.clustering_path).resolve()
    setup_logging()
    logger.info(f"args: {args}")

    cl = HierarchicalCluster.from_file(
        cluster_path=args.clustering_path,
        cluster_fname=args.cluster_fname
    )

    sampled_indices = hierarchical_sampling(
        cl,
        args.target_size,
        args.multiplier,
        args.sampling_strategy,
    )
    if args.valid_indices_path is not None:
        valid_indices = np.load(args.valid_indices_path)
        assert len(valid_indices) == np.sum(
            [len(el) for el in cl.clusters[1]]
        ), "Number of images is not equal to valid_indices size"
        sampled_indices = valid_indices[sampled_indices]

    if args.sort_indices:
        sampled_indices = np.sort(sampled_indices)

    num_images = len(sampled_indices)
    logger.info(f"Number of selected data points: {num_images}")

    save_indices_path = Path(
        args.clustering_path,
        args.save_dir_name,
        f'{cl.n_levels}{args.sampling_strategy}_mul{args.multiplier}_'
        f'{args.target_size}_balanced_selection.npy'
    )
    if len(args.name_suffix) > 0:
        save_indices_path = Path(
            str(save_indices_path).replace(".npy", f"_{args.name_suffix}.npy")
        )
    logger.info(f"Indices will be saved to {str(save_indices_path.resolve())}")
    if args.save:
        Path(args.clustering_path, args.save_dir_name).mkdir(exist_ok=True)
        np.save(save_indices_path, sampled_indices)
        logger.info("Indices are saved!")
