#!/usr/bin/env python3
"""
Test simplest possible ring communication.

Launch with:
torchrun --nproc_per_node=2 test_ring_simple.py
"""

import torch
import torch.distributed as dist
import os


def test_ring_simple():
    """Test the simplest ring communication."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}")
    print(f"Rank {rank}: Using {device}")

    # Create a simple tensor
    x = torch.ones(10, 10, device=device) * rank
    print(f"Rank {rank}: Created tensor, sum = {x.sum().item()}")

    if world_size > 1:
        # Ring communication
        src = (rank - 1) % world_size
        dst = (rank + 1) % world_size

        # Allocate receive buffer
        x_recv = torch.empty_like(x)

        print(f"Rank {rank}: Sending to {dst}, receiving from {src}")

        # Even ranks send first, odd ranks receive first to avoid deadlock
        if rank % 2 == 0:
            # Send first
            dist.send(x, dst=dst)
            print(f"Rank {rank}: Sent")
            # Then receive
            dist.recv(x_recv, src=src)
            print(f"Rank {rank}: Received, sum = {x_recv.sum().item()}")
        else:
            # Receive first
            dist.recv(x_recv, src=src)
            print(f"Rank {rank}: Received, sum = {x_recv.sum().item()}")
            # Then send
            dist.send(x, dst=dst)
            print(f"Rank {rank}: Sent")

        print(f"Rank {rank}: Ring communication successful!")

        # Cleanup
        dist.destroy_process_group()

    print(f"Rank {rank}: Test completed!")


if __name__ == "__main__":
    test_ring_simple()
