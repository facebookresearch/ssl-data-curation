# eta: 1m
import sys
sys.path.insert(0, "..")

import torch
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import os

from src.clusters import HierarchicalCluster
from src import (
  hierarchical_kmeans_gpu as hkmg,
  hierarchical_sampling as hs
)

def load_embeddings(directory):
    embeddings = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)
            loaded_data = torch.load(file_path)
            # Ensure the loaded tensor is 3D
            if loaded_data.ndim == 3:
                loaded_data = loaded_data.reshape(loaded_data.shape[0], -1)  # Flatten the middle dimension
            embeddings.append(loaded_data)
    # Convert all embeddings to torch tensors if they are not already
    embeddings = [torch.tensor(e) if not isinstance(e, torch.Tensor) else e for e in embeddings]
    return torch.cat(embeddings)

#disable backpropagation (and other stuff)
with torch.inference_mode(): 
    # Load data
    # data_directory = '/mnt/ceph/tco/TCO-Students/Homes/schreibpa/project/embeddings/endoViT/Cholec80/1/02/'
    data_directory = '/mnt/ceph/tco/TCO-Students/Homes/schreibpa/project/test_embeddings/'
    data = load_embeddings(data_directory)


    # Print shape to verify it's 2D
    print("Data shape after flattening:", data.shape)

    # Ensure data is 2D
    if data.ndim != 2:
        raise ValueError(f"Data should be 2D but got shape {data.shape}")
    
    # Convert data to torch tensor and move to CUDA
    data_tensor = torch.tensor(data, device="cuda", dtype=torch.float32)

    print("Loaded data on GPU")

    clusters = hkmg.hierarchical_kmeans_with_resampling(
        data=data_tensor,
        n_clusters=[1000, 300],
        n_levels=2,
        sample_sizes=[1, 1],
        verbose=False,
    )

    # Free unused memory
    torch.cuda.empty_cache()

"""     # Convert data to torch tensor and move to CUDA
    data_tensor = torch.tensor(data, device="cuda", dtype=torch.float32) """
""" 
print("Loaded data")

clusters = hkmg.hierarchical_kmeans(
            data=data,
            n_clusters=[1000, 300],
            n_levels=2,
            # sample_sizes=[15, 2],
            verbose=False,
        ) """

"""     # Free unused memory
    torch.cuda.empty_cache()

    try:
        clusters = hkmg.hierarchical_kmeans_with_resampling(
            data=data_tensor,
            n_clusters=[1000, 300],
            n_levels=2,
            sample_sizes=[15, 2],
            verbose=False,
        )
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Reducing batch size or switching to CPU for some operations.")
            # Free unused memory
            del data_tensor
            torch.cuda.empty_cache()
            # Consider processing data in smaller batches or using CPU for intermediate steps
            data_tensor = torch.tensor(data, dtype=torch.float32)  # Keep data on CPU
            clusters = hkmg.hierarchical_kmeans_with_resampling(
                data=data_tensor,
                n_clusters=[1000, 300],
                n_levels=2,
                sample_sizes=[15, 2],
                verbose=False,
            )
        else:
            raise e """

print("Calculated Clusters")
print(clusters)

cl = HierarchicalCluster.from_dict(clusters)
sampled_indices = hs.hierarchical_sampling(cl, target_size=1000)
sampled_indices_as_int = sampled_indices.astype(int)
sampled_points = data[sampled_indices_as_int]
print("Sampled points")
print(sampled_points.shape)