#!/usr/bin/env python3
"""Simple multi-GPU test without ring attention."""

import torch
import torch.distributed as dist
import os


def main():
    # Initialize distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    print(f"[Rank {rank}] Initialized")

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    print(f"[Rank {rank}] Device: {device}")

    # Create a simple tensor
    tensor = torch.ones(100, 100, device=device) * (rank + 1)
    print(f"[Rank {rank}] Created tensor with value {rank + 1}")

    if world_size > 1:
        # Simple all-reduce
        print(f"[Rank {rank}] Starting all-reduce...")
        dist.all_reduce(tensor)
        print(f"[Rank {rank}] All-reduce complete. Sum: {tensor[0, 0].item()}")

        # Barrier to sync
        dist.barrier()
        print(f"[Rank {rank}] Barrier complete")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"[Rank {rank}] Done!")


if __name__ == "__main__":
    main()
