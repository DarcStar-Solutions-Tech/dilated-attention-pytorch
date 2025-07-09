"""
Fixed ring attention communication to avoid deadlocks.
"""

import torch
import torch.distributed as dist
from typing import Tuple
from torch import Tensor


def ring_pass_kv_no_deadlock(
    k: Tensor, v: Tensor, rank: int, world_size: int, step: int
) -> Tuple[Tensor, Tensor]:
    """
    Ring pass that avoids deadlock by using unique tags per step.

    Key fixes:
    1. Use unique tags based on step to avoid tag conflicts
    2. Ensure proper ordering of operations
    3. Handle 2-GPU case specially
    """
    if world_size <= 1:
        return k, v

    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    # Ensure contiguous
    k = k.contiguous()
    v = v.contiguous()

    # Allocate receive buffers
    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    # Use unique tags based on step to avoid conflicts
    base_tag = step * 10
    k_tag = base_tag + 0
    v_tag = base_tag + 1

    if world_size == 2:
        # Special handling for 2 GPUs to avoid deadlock
        # Each rank does operations in opposite order
        if rank == 0:
            # Rank 0: Send K, Recv K, Send V, Recv V
            k_send_req = dist.isend(k, dst=dst, tag=k_tag)
            k_recv_req = dist.irecv(k_recv, src=src, tag=k_tag)
            k_send_req.wait()
            k_recv_req.wait()

            v_send_req = dist.isend(v, dst=dst, tag=v_tag)
            v_recv_req = dist.irecv(v_recv, src=src, tag=v_tag)
            v_send_req.wait()
            v_recv_req.wait()
        else:
            # Rank 1: Recv K, Send K, Recv V, Send V (opposite order)
            k_recv_req = dist.irecv(k_recv, src=src, tag=k_tag)
            k_send_req = dist.isend(k, dst=dst, tag=k_tag)
            k_recv_req.wait()
            k_send_req.wait()

            v_recv_req = dist.irecv(v_recv, src=src, tag=v_tag)
            v_send_req = dist.isend(v, dst=dst, tag=v_tag)
            v_recv_req.wait()
            v_send_req.wait()
    else:
        # For >2 GPUs, standard non-blocking should work
        reqs = []
        reqs.append(dist.isend(k, dst=dst, tag=k_tag))
        reqs.append(dist.irecv(k_recv, src=src, tag=k_tag))
        reqs.append(dist.isend(v, dst=dst, tag=v_tag))
        reqs.append(dist.irecv(v_recv, src=src, tag=v_tag))

        # Wait for all
        for req in reqs:
            req.wait()

    return k_recv, v_recv


def test_ring_communication():
    """Test ring communication to ensure no deadlock."""
    if not dist.is_initialized():
        print("Not in distributed mode")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    print(f"Rank {rank}: Testing ring communication...")

    # Create test tensors
    k = torch.ones(2, 4, 100, 64, device=f"cuda:{device}") * rank
    v = torch.ones(2, 4, 100, 64, device=f"cuda:{device}") * (rank + 10)

    # Test multiple ring passes
    for step in range(world_size):
        print(
            f"Rank {rank}: Step {step}, K[0,0,0,0]={k[0, 0, 0, 0].item():.0f}, "
            f"V[0,0,0,0]={v[0, 0, 0, 0].item():.0f}"
        )

        if step < world_size - 1:
            k, v = ring_pass_kv_no_deadlock(k, v, rank, world_size, step)

    print(f"Rank {rank}: Communication test completed successfully!")


if __name__ == "__main__":
    # This would be run with torchrun
    test_ring_communication()
