#!/usr/bin/env python3
"""
Test basic P2P communication with proper device setup.

Launch with:
torchrun --nproc_per_node=2 test_basic_p2p.py
"""

import torch
import torch.distributed as dist
import os


def main():
    """Test basic P2P communication."""
    # Get rank info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Process {rank}: local_rank={local_rank}, world_size={world_size}")

    if world_size > 1:
        # Set device BEFORE init_process_group
        torch.cuda.set_device(local_rank)

        # Initialize with explicit device
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f"cuda:{local_rank}"),
        )

        print(f"Rank {rank}: Initialized on cuda:{local_rank}")

    # Create small tensor on the correct device
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(10, device=device) * (rank + 1)

    print(f"Rank {rank}: Created tensor on {device}, value={tensor[0].item()}")

    if world_size > 1:
        # Simple P2P test
        if rank == 0:
            # Rank 0 sends to rank 1
            print("Rank 0: Sending tensor to rank 1...")
            dist.send(tensor, dst=1)
            print("Rank 0: Send complete")

            # Receive from rank 1
            recv_tensor = torch.empty_like(tensor)
            print("Rank 0: Waiting to receive from rank 1...")
            dist.recv(recv_tensor, src=1)
            print(f"Rank 0: Received tensor with value={recv_tensor[0].item()}")

        else:
            # Rank 1 receives from rank 0
            recv_tensor = torch.empty_like(tensor)
            print("Rank 1: Waiting to receive from rank 0...")
            dist.recv(recv_tensor, src=0)
            print(f"Rank 1: Received tensor with value={recv_tensor[0].item()}")

            # Send to rank 0
            print("Rank 1: Sending tensor to rank 0...")
            dist.send(tensor, dst=0)
            print("Rank 1: Send complete")

        print(f"Rank {rank}: P2P test successful!")

        # Cleanup
        dist.destroy_process_group()

    print(f"Rank {rank}: Test completed!")


if __name__ == "__main__":
    main()
