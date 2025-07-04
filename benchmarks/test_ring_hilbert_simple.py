#!/usr/bin/env python3
"""
Simple test of Ring Hilbert attention without distributed setup.
Tests the basic concept on a single GPU.
"""

import torch
import time
from dilated_attention_pytorch.ring_dilated_attention_true import (
    TrueRingDilatedAttention,
)
import sys

sys.path.append(".")
from benchmark_true_ring_hilbert import HilbertTrueRingDilatedAttention


def test_simple():
    """Test basic functionality."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Small test case
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    # Create models
    standard = TrueRingDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[1],
        ring_size=1,  # Single GPU
        device=device,
        dtype=torch.float16,
    )

    hilbert = HilbertTrueRingDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[1],
        ring_size=1,  # Single GPU
        device=device,
        dtype=torch.float16,
        use_hilbert=True,
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        out_standard = standard(q, k, v)
        out_hilbert = hilbert(q, k, v)

    print(f"Standard output shape: {out_standard.shape}")
    print(f"Hilbert output shape: {out_hilbert.shape}")

    # Benchmark
    print("\nBenchmarking...")
    warmup = 3
    iterations = 10

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = standard(q, k, v)
            _ = hilbert(q, k, v)

    torch.cuda.synchronize()

    # Standard timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = standard(q, k, v)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / iterations * 1000

    # Hilbert timing
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = hilbert(q, k, v)
    torch.cuda.synchronize()
    hilbert_time = (time.perf_counter() - start) / iterations * 1000

    print("\nResults:")
    print(f"Standard: {standard_time:.2f} ms")
    print(f"Hilbert:  {hilbert_time:.2f} ms")
    print(f"Speedup:  {standard_time / hilbert_time:.2f}x")

    # Test with dilation
    print("\n\nTesting with dilation rate = 2...")
    standard_dilated = TrueRingDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[2],
        ring_size=1,
        device=device,
        dtype=torch.float16,
    )

    hilbert_dilated = HilbertTrueRingDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[2],
        ring_size=1,
        device=device,
        dtype=torch.float16,
        use_hilbert=True,
    )

    with torch.no_grad():
        out_standard_d = standard_dilated(q, k, v)
        out_hilbert_d = hilbert_dilated(q, k, v)

    print(f"Dilated standard shape: {out_standard_d.shape}")
    print(f"Dilated hilbert shape: {out_hilbert_d.shape}")


if __name__ == "__main__":
    test_simple()
