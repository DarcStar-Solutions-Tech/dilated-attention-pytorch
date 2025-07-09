#!/usr/bin/env python3
"""
Debug ring communication deadlock.

Launch with:
torchrun --nproc_per_node=2 test_ring_comm_debug.py
"""

import torch
import torch.distributed as dist
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_attention_fixed_deadlock import (
    ring_pass_kv_no_deadlock,
)


def test_original_ring_pass():
    """Test the original ring pass that might deadlock."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    print(f"Rank {rank}: Testing ORIGINAL ring pass...")

    # Create test tensors
    k = torch.ones(1, 4, 100, 64, device=f"cuda:{device}") * rank
    v = torch.ones(1, 4, 100, 64, device=f"cuda:{device}") * (rank + 10)

    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size

    k_recv = torch.empty_like(k)
    v_recv = torch.empty_like(v)

    try:
        # Original approach - might deadlock
        print(f"Rank {rank}: Starting 4 async operations...")
        reqs = []
        reqs.append(dist.isend(k.contiguous(), dst=dst, tag=0))
        reqs.append(dist.irecv(k_recv, src=src, tag=0))
        reqs.append(dist.isend(v.contiguous(), dst=dst, tag=1))
        reqs.append(dist.irecv(v_recv, src=src, tag=1))

        print(f"Rank {rank}: Waiting for operations...")
        for i, req in enumerate(reqs):
            print(f"Rank {rank}: Waiting for request {i}...")
            req.wait()
            print(f"Rank {rank}: Request {i} completed")

        print(f"Rank {rank}: ORIGINAL test passed!")

    except Exception as e:
        print(f"Rank {rank}: ORIGINAL test failed: {e}")


def test_fixed_ring_pass():
    """Test the fixed ring pass."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    print(f"\nRank {rank}: Testing FIXED ring pass...")

    # Create test tensors
    k = torch.ones(1, 4, 100, 64, device=f"cuda:{device}") * rank
    v = torch.ones(1, 4, 100, 64, device=f"cuda:{device}") * (rank + 10)

    try:
        # Fixed approach
        for step in range(3):  # Test multiple steps
            print(f"Rank {rank}: Step {step}")
            k_new, v_new = ring_pass_kv_no_deadlock(k, v, rank, world_size, step)
            print(
                f"Rank {rank}: Step {step} completed. "
                f"K changed: {not torch.allclose(k, k_new)}, "
                f"V changed: {not torch.allclose(v, v_new)}"
            )
            k, v = k_new, v_new

        print(f"Rank {rank}: FIXED test passed!")

    except Exception as e:
        print(f"Rank {rank}: FIXED test failed: {e}")


def main():
    """Main test function."""
    # Setup distributed
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    if rank == 0:
        print("=" * 60)
        print("Testing Ring Communication Deadlock")
        print("=" * 60)

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    # Test original (might timeout/deadlock)
    start = time.time()
    test_original_ring_pass()
    print(f"Rank {rank}: Original test took {time.time() - start:.2f}s")

    # Synchronize
    if dist.is_initialized():
        dist.barrier()

    # Test fixed version
    start = time.time()
    test_fixed_ring_pass()
    print(f"Rank {rank}: Fixed test took {time.time() - start:.2f}s")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"Rank {rank}: All tests completed!")


if __name__ == "__main__":
    main()
