#!/usr/bin/env python3
"""
Test if basic ring communication works with PyTorch distributed.
Run with: torchrun --nproc_per_node=2 test_ring_communication.py
"""

import os
import torch
import torch.distributed as dist


def test_basic_ring():
    """Test basic ring passing pattern."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting ring communication test")

    # Create a tensor with rank value
    tensor = torch.full((4, 4), float(rank), device=device)
    recv_tensor = torch.empty_like(tensor)

    print(f"[Rank {rank}] Initial tensor: {tensor[0, 0].item()}")

    # Ring pass
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Try different approaches

    # Approach 1: Using send/recv (blocking)
    if rank == 0:
        print("\nTesting blocking send/recv...")

    dist.barrier()

    if rank == 0:
        dist.send(tensor, next_rank)
        dist.recv(recv_tensor, prev_rank)
    else:
        dist.recv(recv_tensor, prev_rank)
        dist.send(tensor, next_rank)

    print(f"[Rank {rank}] After blocking ring: received {recv_tensor[0, 0].item()}")

    # Approach 2: Using isend/irecv (non-blocking)
    if rank == 0:
        print("\nTesting non-blocking isend/irecv...")

    dist.barrier()

    # Reset
    tensor.fill_(float(rank))
    recv_tensor.zero_()

    send_op = dist.isend(tensor, next_rank)
    recv_op = dist.irecv(recv_tensor, prev_rank)

    send_op.wait()
    recv_op.wait()

    print(f"[Rank {rank}] After non-blocking ring: received {recv_tensor[0, 0].item()}")

    # Approach 3: Using all_to_all
    if rank == 0:
        print("\nTesting all_to_all approach...")

    dist.barrier()

    # Create list of tensors
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    input_list = [tensor.clone() for _ in range(world_size)]

    dist.all_to_all(output_list, input_list)

    print(
        f"[Rank {rank}] After all_to_all: received from rank 0: {output_list[0][0, 0].item()}"
    )


def main():
    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 test_ring_communication.py")
        return

    dist.init_process_group(backend="nccl")

    try:
        test_basic_ring()
        print(f"\n[Rank {dist.get_rank()}] ✅ All communication patterns work!")
    except Exception as e:
        print(f"\n[Rank {dist.get_rank()}] ❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
