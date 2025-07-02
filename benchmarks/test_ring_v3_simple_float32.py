#!/usr/bin/env python3
"""
Simple test of Ring V3 with float32 to debug issues.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_simple_float32.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_simple_float32():
    """Simple test with float32 and detailed logging."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_simple_float32.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Initialized, world_size={world_size}")

    # Start with small sequence
    seq_len = 512

    print(f"[Rank {rank}] Creating model with seq_len={seq_len}, float32...")

    model = RingDilatedAttentionV3(
        segment_lengths=[256],
        dilation_rates=[1],
        bucket_size=128,
        use_bucketed=True,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    print(f"[Rank {rank}] Model created")

    # Create small inputs
    torch.manual_seed(42)
    q = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * 0.1
    k = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * 0.1
    v = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * 0.1

    print(f"[Rank {rank}] Inputs created, starting forward pass...")

    try:
        output = model(q, k, v, is_causal=False)
        print(f"[Rank {rank}] ✅ Forward pass completed!")

        # Check output
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        output_mean = output.mean().item()

        print(
            f"[Rank {rank}] Output: NaN={has_nan}, Inf={has_inf}, mean={output_mean:.6f}"
        )

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")
        import traceback

        traceback.print_exc()

    # Now try 1024 tokens
    print(f"\n[Rank {rank}] Testing 1024 tokens...")

    seq_len = 1024
    model = RingDilatedAttentionV3(
        segment_lengths=[512],
        dilation_rates=[1],
        bucket_size=256,
        use_bucketed=True,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    # Scale inputs more aggressively
    scale = 0.05  # Even smaller values
    q = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * scale
    k = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * scale
    v = torch.randn(1, seq_len, 4, 32, device=device, dtype=torch.float32) * scale

    print(f"[Rank {rank}] Created inputs with scale={scale}")

    try:
        output = model(q, k, v, is_causal=False)

        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        output_mean = output.mean().item()
        output_max = output.abs().max().item()

        if has_nan or has_inf:
            print(f"[Rank {rank}] ❌ 1024 tokens: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"[Rank {rank}] ✅ 1024 tokens: Success!")
            print(f"[Rank {rank}]    mean={output_mean:.6f}, max={output_max:.6f}")

    except Exception as e:
        print(f"[Rank {rank}] ❌ 1024 token error: {e}")

    print(f"[Rank {rank}] Cleaning up...")
    dist.destroy_process_group()
    print(f"[Rank {rank}] Done")


if __name__ == "__main__":
    test_simple_float32()
