#!/usr/bin/env python3
"""
Simple test for Ring V3 to debug issues.
Run with: torchrun --nproc_per_node=2 test_ring_v3_simple.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_simple():
    """Very simple test to isolate issues."""

    # Initialize distributed if available
    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    print(f"[Rank {rank}] Starting simple test, world_size={world_size}")

    # Medium test case
    seq_len = 512  # Must be divisible by world_size
    batch_size = 1
    num_heads = 4
    head_dim = 32

    try:
        # Create model with medium segments and bucketing
        model = RingDilatedAttentionV3(
            segment_lengths=[256],
            dilation_rates=[1],
            bucket_size=128,  # Medium buckets
            use_bucketed=True,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Create small inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        print(f"[Rank {rank}] Inputs created, running forward...")

        # Forward pass
        output = model(q, k, v, is_causal=False)

        print(f"[Rank {rank}] ✅ Forward pass completed!")
        print(f"[Rank {rank}] Output shape: {output.shape}")

        # Synchronize
        if world_size > 1:
            dist.barrier()
            print(f"[Rank {rank}] ✅ Barrier passed!")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    test_simple()
