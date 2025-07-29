#!/usr/bin/env python3
"""Test impact of different block sizes on Triton performance."""

import torch
import triton
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    hilbert_attention_kernel,
    standard_attention_kernel,
    create_hilbert_mapping,
)


def benchmark_block_sizes(seq_len=2048, num_heads=12, head_dim=64):
    """Benchmark different block size configurations."""
    device = "cuda"
    B = 1  # batch size

    # Create tensors
    q = torch.randn(B, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(B, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(B, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    out = torch.zeros_like(q)

    # Create Hilbert mapping
    hilbert_map = create_hilbert_mapping(seq_len).to(device)

    # Test configurations
    block_configs = [
        (32, 32, 32),  # Current default
        (64, 64, 64),  # Moderate
        (128, 128, 64),  # Aggressive
        (64, 128, 64),  # Asymmetric
    ]

    print(f"Block Size Performance Test (seq_len={seq_len})")
    print("=" * 60)
    print(
        f"{'BLOCK_M':<10} {'BLOCK_N':<10} {'BLOCK_D':<10} {'Grid Size':<15} {'Time (ms)':<15}"
    )
    print("-" * 60)

    for BLOCK_M, BLOCK_N, BLOCK_D in block_configs:
        # Skip invalid configurations
        if BLOCK_M > seq_len or BLOCK_N > seq_len or BLOCK_D > head_dim:
            continue

        grid_size = triton.cdiv(seq_len, BLOCK_M) * B * num_heads

        # Warmup
        grid = (grid_size,)
        hilbert_attention_kernel[grid](
            q,
            k,
            v,
            out,
            hilbert_map,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            B,
            num_heads,
            seq_len,
            head_dim,
            scale=head_dim**-0.5,
            segment_size=128,
            dilation_rate=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        hilbert_attention_kernel[grid](
            q,
            k,
            v,
            out,
            hilbert_map,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            B,
            num_heads,
            seq_len,
            head_dim,
            scale=head_dim**-0.5,
            segment_size=128,
            dilation_rate=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        print(
            f"{BLOCK_M:<10} {BLOCK_N:<10} {BLOCK_D:<10} {grid_size:<15} {elapsed:<15.2f}"
        )

    # Also test without Hilbert
    print("\nStandard attention (no Hilbert):")

    for BLOCK_M, BLOCK_N, BLOCK_D in [(64, 64, 64), (128, 128, 64)]:
        if BLOCK_M > seq_len or BLOCK_N > seq_len or BLOCK_D > head_dim:
            continue

        grid_size = triton.cdiv(seq_len, BLOCK_M) * B * num_heads
        grid = (grid_size,)

        # Warmup
        standard_attention_kernel[grid](
            q,
            k,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            B,
            num_heads,
            seq_len,
            head_dim,
            scale=head_dim**-0.5,
            segment_size=128,
            dilation_rate=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        torch.cuda.synchronize()
        start = time.perf_counter()

        standard_attention_kernel[grid](
            q,
            k,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out.stride(),
            B,
            num_heads,
            seq_len,
            head_dim,
            scale=head_dim**-0.5,
            segment_size=128,
            dilation_rate=1,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        print(
            f"{BLOCK_M:<10} {BLOCK_N:<10} {BLOCK_D:<10} {grid_size:<15} {elapsed:<15.2f}"
        )


if __name__ == "__main__":
    print("Testing different sequence lengths:\n")

    for seq_len in [1024, 2048, 4096]:
        benchmark_block_sizes(seq_len=seq_len)
        print("\n")
