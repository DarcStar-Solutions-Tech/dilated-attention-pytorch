#!/usr/bin/env python3
"""
Test hybrid implementation with exactly 1024 tokens to debug the timeout.
Run with: torchrun --nproc_per_node=2 benchmarks/test_hybrid_1024_tokens.py
"""

import os
import torch
import torch.distributed as dist
import time


def test_1024_tokens():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_hybrid_1024_tokens.py"
        )
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing with 1024 tokens...")

    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Create model with segment_length = seq_len // 2 = 512
    print(f"[Rank {rank}] Creating model with segment_length=512...")
    model = RingDilatedAttentionHybrid(
        segment_lengths=[512],
        dilation_rates=[1],
        ring_size=world_size,
        device=device,
    )
    print(f"[Rank {rank}] Model created, dtype={model.dtype}")

    # Create inputs
    seq_len = 1024
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(f"[Rank {rank}] Creating inputs...")
    q = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )
    k = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )
    v = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )
    print(f"[Rank {rank}] Inputs created")

    # Test forward pass
    print(f"[Rank {rank}] Starting forward pass...")
    start = time.time()

    try:
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        elapsed = time.time() - start
        print(f"[Rank {rank}] Forward pass completed in {elapsed:.3f}s")
        print(f"[Rank {rank}] Output shape: {output.shape}")

        # Check for NaN/Inf
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        print(f"[Rank {rank}] NaN: {has_nan}, Inf: {has_inf}")

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback

        traceback.print_exc()

    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_1024_tokens()
