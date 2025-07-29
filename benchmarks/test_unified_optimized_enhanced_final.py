#!/usr/bin/env python3
"""
Final test of the Unified Optimized Enhanced implementation.
"""

import torch
import torch.nn as nn
import time
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels.hilbert_attention_unified import (
    UnifiedHilbertAttention as Original,
)
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced import (
    UnifiedHilbertAttentionOptimizedEnhanced as Enhanced,
)


def benchmark_implementation(
    module: nn.Module, x: torch.Tensor, name: str, warmup: int = 3, runs: int = 10
):
    """Benchmark a single implementation."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)
        torch.cuda.synchronize()

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            out = module(x)
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) / runs * 1000  # ms
    return out, avg_time


def main():
    # Test parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Test different sequence lengths and patterns
    test_configs = [
        (1024, 1, "Dense 1K"),
        (2048, 1, "Dense 2K"),
        (4096, 1, "Dense 4K"),
        (8192, 1, "Dense 8K"),
        (2048, 2, "Sparse 2K (dilation=2)"),
        (4096, 2, "Sparse 4K (dilation=2)"),
        (4096, 4, "Sparse 4K (dilation=4)"),
    ]

    print("=== Unified Optimized Enhanced vs Original ===")
    print(f"Batch: {batch_size}, Hidden: {hidden_dim}, Heads: {num_heads}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()

    for seq_len, dilation_rate, desc in test_configs:
        # Create modules
        original = Original(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            hilbert_threshold=1024,
        ).cuda()

        enhanced = Enhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            hilbert_threshold=1024,
            enable_8k_optimization=True,
        ).cuda()

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        # Benchmark
        out_orig, time_orig = benchmark_implementation(original, x, "Original")
        out_enh, time_enh = benchmark_implementation(enhanced, x, "Enhanced")

        # Verify correctness
        if torch.allclose(out_orig, out_enh, rtol=1e-3, atol=1e-3):
            correctness = "✓ Match"
        else:
            max_diff = (out_orig - out_enh).abs().max().item()
            correctness = f"✗ Diff: {max_diff:.6f}"

        # Print results
        speedup = time_orig / time_enh
        print(
            f"{desc:25} | Original: {time_orig:7.2f}ms | Enhanced: {time_enh:7.2f}ms | "
            f"Speedup: {speedup:5.2f}x | {correctness}"
        )

    # Special test for 8K optimization
    print("\n=== 8K Optimization Test ===")
    seq_len = 8192
    x_8k = torch.randn(1, seq_len, hidden_dim).cuda()

    # Test with and without 8K optimization
    enhanced_no_8k = Enhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=1,
        hilbert_threshold=1024,
        enable_8k_optimization=False,
    ).cuda()

    enhanced_8k = Enhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=1,
        hilbert_threshold=1024,
        enable_8k_optimization=True,
    ).cuda()

    _, time_no_8k = benchmark_implementation(enhanced_no_8k, x_8k, "No 8K opt")
    _, time_8k = benchmark_implementation(enhanced_8k, x_8k, "With 8K opt")

    print(f"8K without optimization: {time_no_8k:7.2f}ms")
    print(f"8K with optimization:    {time_8k:7.2f}ms")
    print(f"8K optimization speedup: {time_no_8k / time_8k:5.2f}x")


if __name__ == "__main__":
    main()
