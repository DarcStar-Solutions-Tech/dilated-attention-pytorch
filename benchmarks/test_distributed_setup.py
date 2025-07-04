#!/usr/bin/env python3
"""Test distributed setup with 2 GPUs."""

import os
import torch
import torch.distributed as dist


def test_distributed():
    """Test if distributed is properly initialized."""

    # Print environment
    print("Process started with:")
    print(f"  RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("\nNo GPUs available!")
        return

    # Try to initialize distributed
    try:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            print("\nDistributed initialized successfully!")
            print(f"  Rank: {dist.get_rank()}")
            print(f"  World size: {dist.get_world_size()}")
            print(f"  Backend: {dist.get_backend()}")

            # Simple all-reduce test
            rank = dist.get_rank()
            tensor = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
            print(f"\nBefore all-reduce: {tensor.item()}")

            dist.all_reduce(tensor)
            print(
                f"After all-reduce: {tensor.item()} (should be {sum(range(1, dist.get_world_size() + 1))})"
            )

            dist.destroy_process_group()
            print("\nDistributed destroyed successfully!")
        else:
            print("\nDistributed already initialized!")

    except Exception as e:
        print(f"\nError initializing distributed: {e}")
        print(f"Exception type: {type(e).__name__}")


if __name__ == "__main__":
    test_distributed()
