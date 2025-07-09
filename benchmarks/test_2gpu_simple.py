#!/usr/bin/env python3
"""
Simple test to verify 2-GPU distributed setup works.

Launch with:
torchrun --nproc_per_node=2 test_2gpu_simple.py
"""

import torch
import torch.distributed as dist
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Simple distributed test."""
    # Setup
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")

    print(f"Rank {rank}/{world_size} on device {device}")

    # Test 1: Simple tensor communication
    tensor = torch.ones(10, device=device) * (rank + 1)
    print(f"Rank {rank} initial tensor: {tensor[0].item()}")

    if world_size > 1:
        # All-reduce sum
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"Rank {rank} after all-reduce: {tensor[0].item()}")

        # Expected: rank 0 = 3.0, rank 1 = 3.0 (1+2)
        expected = sum(range(1, world_size + 1))
        assert tensor[0].item() == expected, (
            f"Expected {expected}, got {tensor[0].item()}"
        )

    # Test 2: Ring pass simulation (without the model)
    if world_size > 1:
        # Create buffer
        send_tensor = torch.ones(100, 100, device=device) * (rank + 1)
        recv_tensor = torch.empty_like(send_tensor)

        src = (rank - 1) % world_size
        dst = (rank + 1) % world_size

        print(f"Rank {rank}: sending to {dst}, receiving from {src}")

        # Use P2P operations
        if rank == 0:
            # Rank 0: send first, then receive
            dist.send(send_tensor, dst)
            dist.recv(recv_tensor, src)
        else:
            # Rank 1: receive first, then send
            dist.recv(recv_tensor, src)
            dist.send(send_tensor, dst)

        print(f"Rank {rank}: received value {recv_tensor[0, 0].item()} from rank {src}")

    print(f"Rank {rank}: All tests passed!")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
