#!/usr/bin/env python3
"""
Verify GPU assignment in distributed mode.
"""

import torch
import torch.distributed as dist
import os


def main():
    """Check GPU assignment."""

    # Initialize distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    print(f"\n[Rank {rank}] Process information:")
    print(f"  World size: {world_size}")
    print(f"  Global rank: {rank}")
    print(f"  Local rank: {local_rank}")

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {num_gpus}")

    # Default device selection (this is the problem!)
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Default device: {default_device}")

    # Correct device selection using local_rank
    correct_device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    print(f"  Correct device: {correct_device}")

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"  Current CUDA device: {current_device}")
        print(f"  Device name: {device_name}")

        # Create a test tensor
        x = torch.ones(10, device=correct_device)
        print(f"  Test tensor device: {x.device}")

    # Synchronize
    if dist.is_initialized():
        dist.barrier()
        print(f"\n[Rank {rank}] Barrier passed - all processes synchronized")

        # Test communication
        test_tensor = torch.tensor([rank], device=correct_device, dtype=torch.float32)
        dist.all_reduce(test_tensor)
        print(
            f"[Rank {rank}] All-reduce result: {test_tensor.item()} (expected: {sum(range(world_size))})"
        )

        dist.destroy_process_group()

    print(f"\n[Rank {rank}] Complete!")


if __name__ == "__main__":
    main()
