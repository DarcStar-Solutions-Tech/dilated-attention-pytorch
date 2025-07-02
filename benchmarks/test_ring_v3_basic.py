#!/usr/bin/env python3
"""
Basic test of Ring V3 without bucketing to isolate the issue.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_basic.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_basic():
    """Test basic functionality without bucketing."""

    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_basic.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting basic test")

    # Test 1: Very small sequence without bucketing
    print(f"\n[Rank {rank}] Test 1: Small sequence without bucketing")

    model = RingDilatedAttentionV3(
        segment_lengths=[128],
        dilation_rates=[1],
        use_bucketed=False,  # Disable bucketing
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    # Small inputs
    seq_len = 256
    q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
    k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
    v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

    try:
        output = model(q, k, v, is_causal=False)
        has_nan = torch.isnan(output).any().item()
        print(f"[Rank {rank}] Test 1: Success! NaN={has_nan}, shape={output.shape}")
    except Exception as e:
        print(f"[Rank {rank}] Test 1 failed: {e}")

    # Test 2: With bucketing enabled
    print(f"\n[Rank {rank}] Test 2: Small sequence with bucketing")

    model = RingDilatedAttentionV3(
        segment_lengths=[128],
        dilation_rates=[1],
        bucket_size=64,
        use_bucketed=True,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    try:
        output = model(q, k, v, is_causal=False)
        has_nan = torch.isnan(output).any().item()
        print(f"[Rank {rank}] Test 2: Success! NaN={has_nan}")
    except Exception as e:
        print(f"[Rank {rank}] Test 2 failed: {e}")

    # Test 3: Single GPU comparison
    print(f"\n[Rank {rank}] Test 3: Compare single vs multi GPU")

    # Force single GPU mode
    model_single = RingDilatedAttentionV3(
        segment_lengths=[128],
        dilation_rates=[1],
        use_bucketed=False,
        device=device,
        dtype=torch.float32,
        ring_size=1,  # Force single device
    )

    # Test on rank 0 only
    if rank == 0:
        output_single = model_single(q, k, v, is_causal=False)
        print(f"[Rank {rank}] Single GPU: mean={output_single.mean().item():.6f}")

    # Multi-GPU
    dist.barrier()
    output_multi = model(q, k, v, is_causal=False)
    print(f"[Rank {rank}] Multi GPU: mean={output_multi.mean().item():.6f}")

    print(f"\n[Rank {rank}] Cleaning up")
    dist.destroy_process_group()


if __name__ == "__main__":
    test_basic()
