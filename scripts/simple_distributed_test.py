#!/usr/bin/env python3
"""Test simple distributed operations."""

import os
import torch
import torch.distributed as dist

if "WORLD_SIZE" in os.environ:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # CRITICAL: Set the correct GPU for each process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(
        f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}, Device: {device}"
    )

    # Test all_gather
    local_tensor = torch.tensor([rank], device=device)
    gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]

    dist.all_gather(gathered, local_tensor)

    print(f"[Rank {rank}] All-gather result: {[t.item() for t in gathered]}")

    dist.destroy_process_group()
else:
    print("Run with: torchrun --nproc_per_node=2 test_simple_distributed.py")
