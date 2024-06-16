# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import re
import socket
from typing import Dict, List

import torch
import torch.distributed as dist


_LOCAL_RANK = -1
_LOCAL_WORLD_SIZE = -1


def is_distributed_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    """
    Returns:
        The number of processes in the process group
    """
    return dist.get_world_size() if is_distributed_enabled() else 1


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_distributed_enabled() else 0


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not is_distributed_enabled():
        return 0
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_RANK


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not is_distributed_enabled():
        return 1
    assert 0 <= _LOCAL_RANK < _LOCAL_WORLD_SIZE
    return _LOCAL_WORLD_SIZE


def is_main_process() -> bool:
    """
    Returns:
        True if the current process is the main one.
    """
    return get_global_rank() == 0


def save_in_main_process(*args, **kwargs) -> None:
    """Utility function to save only from the main process"""
    if not is_main_process():
        return
    torch.save(*args, **kwargs)


def _get_master_port(seed: int = 0) -> int:
    MIN_MASTER_PORT, MAX_MASTER_PORT = (20_000, 60_000)

    master_port_str = os.environ.get("MASTER_PORT")
    if master_port_str is None:
        rng = random.Random(seed)
        return rng.randint(MIN_MASTER_PORT, MAX_MASTER_PORT)

    return int(master_port_str)


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # A "" host address means INADDR_ANY i.e. binding to all interfaces.
        # Note this is not compatible with IPv6.
        s.bind(("", 0))
        port = s.getsockname()[1]
        return port


_TORCH_DISTRIBUTED_ENV_VARS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
)


def _collect_env_vars() -> Dict[str, str]:
    return {
        env_var: os.environ[env_var]
        for env_var in _TORCH_DISTRIBUTED_ENV_VARS
        if env_var in os.environ
    }


def _is_slurm_job_process() -> bool:
    return "SLURM_JOB_ID" in os.environ


def _parse_slurm_node_list(s: str) -> List[str]:
    nodes = []
    # Extract "hostname", "hostname[1-2,3,4-5]," substrings
    p = re.compile(r"(([^\[]+)(?:\[([^\]]+)\])?),?")
    for m in p.finditer(s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                width = len(span[0])
                start, end = int(span[0]), int(span[1]) + 1
                nodes.extend([prefix + f"{i:0{width}}" for i in range(start, end)])
    return nodes


class _TorchDistributedEnvironment:
    def __init__(self, use_torchrun):
        self.master_addr = "127.0.0.1"
        self.master_port = 0
        self.rank = -1
        self.world_size = -1
        self.local_rank = -1
        self.local_world_size = -1

        if _is_slurm_job_process() and not use_torchrun:
            # use_torchrun is set to True to launch computation on nodes with srun or salloc
            return self._set_from_slurm_env()

        env_vars = _collect_env_vars()
        if not env_vars:
            # Environment is not set
            pass
        elif len(env_vars) == len(_TORCH_DISTRIBUTED_ENV_VARS):
            # Environment is fully set
            return self._set_from_preset_env()
        else:
            # Environment is partially set
            collected_env_vars = ", ".join(env_vars.keys())
            raise RuntimeError(f"Partially set environment: {collected_env_vars}")

        if torch.cuda.device_count() > 0:
            return self._set_from_local()

        raise RuntimeError("Can't initialize PyTorch distributed environment")

    # Slurm job created with sbatch, submitit, etc...
    def _set_from_slurm_env(self):
        job_id = int(os.environ["SLURM_JOB_ID"])
        node_count = int(os.environ["SLURM_JOB_NUM_NODES"])
        nodes = _parse_slurm_node_list(os.environ["SLURM_JOB_NODELIST"])
        assert len(nodes) == node_count

        self.master_addr = nodes[0]
        self.master_port = _get_master_port(seed=job_id)
        print(f"using {self.master_addr}:{self.master_port}")
        self.rank = int(os.environ["SLURM_PROCID"])
        self.world_size = int(os.environ["SLURM_NTASKS"])
        assert self.rank < self.world_size
        self.local_rank = int(os.environ["SLURM_LOCALID"])
        self.local_world_size = self.world_size // node_count
        assert self.local_rank < self.local_world_size

    # Single node job with preset environment (i.e. torchrun)
    def _set_from_preset_env(self):
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        assert self.rank < self.world_size
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        assert self.local_rank < self.local_world_size

    # Single node and GPU job (i.e. local script run)
    def _set_from_local(self):
        self.master_addr = "127.0.0.1"
        self.master_port = _get_available_port()
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.local_world_size = 1

    def export(self, *, overwrite: bool) -> "_TorchDistributedEnvironment":
        # See the "Environment variable initialization" section from
        # https://pytorch.org/docs/stable/distributed.html for the complete list of
        # environment variables required for the env:// initialization method.
        env_vars = {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_RANK": str(self.local_rank),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
        }
        print(env_vars)

        if not overwrite:
            for key in env_vars:
                # Only check for difference with preset environment variables
                if key in os.environ and os.environ[key] != env_vars[key]:
                    raise RuntimeError(
                        f"Cannot export environment variables as {key} is already set"
                    )

        os.environ.update(env_vars)
        return self


def enable_distributed(
    *, use_torchrun=False, set_cuda_current_device: bool = True, overwrite: bool = False
):
    """Enable distributed mode
    Args:
        set_cuda_current_device: If True, call torch.cuda.set_device() to set the
            current PyTorch CUDA device to the one matching the local rank.
        overwrite: If True, overwrites already set variables. Else fails.
    """

    global _LOCAL_RANK, _LOCAL_WORLD_SIZE
    if _LOCAL_RANK >= 0 or _LOCAL_WORLD_SIZE >= 0:
        raise RuntimeError("Distributed mode has already been enabled")
    torch_env = _TorchDistributedEnvironment(use_torchrun)
    torch_env.export(overwrite=overwrite)

    if set_cuda_current_device:
        torch.cuda.set_device(torch_env.local_rank)

    dist.init_process_group(backend="nccl")
    dist.barrier()


def gather_tensor(x, do_all_gather=False):
    """
    Gather tensors from all ranks to the main rank.
    There is an option to all_gather.
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(x.size(), device=x.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *x.size()[1:])
        padding = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
        x = torch.cat((x, padding))

    if do_all_gather:
        all_tensors_padded = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, x)
        synchronize()
        all_tensors = []
        for tensor_, size in zip(all_tensors_padded, all_sizes):
            all_tensors.append(tensor_[: size[0]])
        return torch.cat(all_tensors)
    else:
        if is_main_process():
            all_tensors_padded = [torch.zeros_like(x) for _ in range(world_size)]
            dist.gather(x, all_tensors_padded)
        else:
            dist.gather(x, dst=0)
        synchronize()
        if is_main_process():
            all_tensors = []
            for tensor_, size in zip(all_tensors_padded, all_sizes):
                all_tensors.append(tensor_[: size[0]])
            return torch.cat(all_tensors)
        else:
            return None


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_global_size()
    if world_size == 1:
        return
    dist.barrier()
