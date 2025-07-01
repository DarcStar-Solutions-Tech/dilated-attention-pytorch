#!/usr/bin/env python3
"""
Debug ring communication pattern.
"""

import os
import torch
import torch.distributed as dist


def test_ring_communication():
    """Test basic ring communication pattern."""
    if "WORLD_SIZE" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 debug_ring_communication.py")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}] Initialized")

    # Create test tensor
    tensor_size = (1, 1024, 8, 64)  # Small tensor
    my_tensor = torch.full(tensor_size, rank, device=device, dtype=torch.float32)

    print(f"[Rank {rank}] Created tensor with value {rank}")

    # Test 1: Basic send/recv
    print(f"\n[Rank {rank}] Test 1: Basic send/recv")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    recv_tensor = torch.empty_like(my_tensor)

    # Everyone sends to next and receives from previous
    if rank == 0:
        # Rank 0 sends first, then receives
        print(f"[Rank {rank}] Sending to rank {next_rank}")
        dist.send(my_tensor, next_rank)
        print(f"[Rank {rank}] Receiving from rank {prev_rank}")
        dist.recv(recv_tensor, prev_rank)
    else:
        # Other ranks receive first, then send
        print(f"[Rank {rank}] Receiving from rank {prev_rank}")
        dist.recv(recv_tensor, prev_rank)
        print(f"[Rank {rank}] Sending to rank {next_rank}")
        dist.send(my_tensor, next_rank)

    print(f"[Rank {rank}] Received tensor with value {recv_tensor[0, 0, 0, 0].item()}")

    # Test 2: Async send/recv
    dist.barrier()
    print(f"\n[Rank {rank}] Test 2: Async send/recv")

    send_req = dist.isend(my_tensor, next_rank)
    recv_req = dist.irecv(recv_tensor, prev_rank)

    print(f"[Rank {rank}] Waiting for async operations...")
    send_req.wait()
    recv_req.wait()

    print(
        f"[Rank {rank}] Async complete, received value {recv_tensor[0, 0, 0, 0].item()}"
    )

    # Test 3: Ring loop
    dist.barrier()
    print(f"\n[Rank {rank}] Test 3: Ring loop")

    current_tensor = my_tensor.clone()

    for step in range(world_size):
        source_rank = (rank - step) % world_size
        print(
            f"[Rank {rank}] Step {step}: current value = {current_tensor[0, 0, 0, 0].item()} (from rank {source_rank})"
        )

        if step < world_size - 1:
            # Pass to next
            recv_buffer = torch.empty_like(current_tensor)

            # Use different approach to avoid deadlock
            reqs = []
            reqs.append(dist.isend(current_tensor, next_rank))
            reqs.append(dist.irecv(recv_buffer, prev_rank))

            for req in reqs:
                req.wait()

            current_tensor = recv_buffer

    print(f"\n[Rank {rank}] All tests completed successfully!")
    dist.destroy_process_group()


if __name__ == "__main__":
    test_ring_communication()
