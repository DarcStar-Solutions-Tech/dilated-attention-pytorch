#!/usr/bin/env python3
"""
Test ring utilities to find where the hang occurs.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_utils.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_attention_utils import all_ring_pass, split_by_rank


def test_ring_utils():
    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_utils.py")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing ring utilities...")

    # Create simple tensor
    full_tensor = torch.arange(16, device=device).reshape(1, 16, 1, 1)
    print(f"[Rank {rank}] Full tensor shape: {full_tensor.shape}")

    # Split by rank
    local_chunk = split_by_rank(full_tensor, rank, world_size)
    print(f"[Rank {rank}] Local chunk shape: {local_chunk.shape}")
    print(f"[Rank {rank}] Local chunk values: {local_chunk.squeeze().tolist()}")

    # Test ring pass
    print(f"[Rank {rank}] Starting ring pass...")

    for i, (ring_info, (chunk,)) in enumerate(
        all_ring_pass(local_chunk, ring_size=world_size)
    ):
        print(f"[Rank {rank}] Ring iteration {i}:")
        print(f"[Rank {rank}]   Ring info: {ring_info}")
        print(
            f"[Rank {rank}]   Chunk shape: {chunk.shape if chunk is not None else 'None'}"
        )
        if chunk is not None:
            print(f"[Rank {rank}]   Chunk values: {chunk.squeeze().tolist()}")

    print(f"[Rank {rank}] Ring pass completed!")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_ring_utils()
