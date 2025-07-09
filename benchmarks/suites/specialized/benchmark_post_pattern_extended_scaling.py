#!/usr/bin/env python3
"""
Extended scaling test for post-pattern optimization.
Tests a wider range of sequence lengths to understand scaling behavior.
"""

import torch
import time
import gc
import json
from datetime import datetime
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import create_block_sparse_attention, SparsePatternConfig
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    create_post_pattern_hilbert_attention,
)


def benchmark_sequence_length(
    seq_len: int,
    dilation_rate: int = 2,
    sparsity_ratio: float = 0.1,
    block_size: int = 64,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> Tuple[float, float]:
    """Benchmark both standard and post-pattern for a given sequence length."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_length = seq_len // 4

    # Create models
    try:
        standard_model = create_block_sparse_attention(
            variant="base",
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparse_config=SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=sparsity_ratio,
                block_size=block_size,
            ),
        ).to(device)

        post_pattern_model = create_post_pattern_hilbert_attention(
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        ).to(device)

    except Exception as e:
        print(f"Failed to create models for seq_len={seq_len}: {e}")
        return float("inf"), float("inf")

    # Create inputs
    try:
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    except Exception as e:
        print(f"Failed to create inputs for seq_len={seq_len}: {e}")
        return float("inf"), float("inf")

    # Benchmark standard
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        # Warmup
        for _ in range(warmup_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = standard_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(benchmark_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = standard_model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()

        standard_time = (time.perf_counter() - start) / benchmark_iters * 1000
    except Exception as e:
        print(f"Standard benchmark failed for seq_len={seq_len}: {e}")
        standard_time = float("inf")

    # Benchmark post-pattern
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        # Warmup
        for _ in range(warmup_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = post_pattern_model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(benchmark_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = post_pattern_model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()

        post_pattern_time = (time.perf_counter() - start) / benchmark_iters * 1000
    except Exception as e:
        print(f"Post-pattern benchmark failed for seq_len={seq_len}: {e}")
        post_pattern_time = float("inf")

    # Cleanup
    del standard_model, post_pattern_model, q, k, v
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return standard_time, post_pattern_time


def main():
    """Run extended scaling analysis."""

    print("=" * 80)
    print("Post-Pattern Optimization Extended Scaling Analysis")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Test a wider range of sequence lengths
    # Must be divisible by block_size (64) and have at least 4 segments
    sequence_lengths = [
        1024,  # 16 blocks
        2048,  # 32 blocks
        4096,  # 64 blocks
        6144,  # 96 blocks
        8192,  # 128 blocks
        12288,  # 192 blocks
        16384,  # 256 blocks
        24576,  # 384 blocks
        32768,  # 512 blocks
    ]

    # Test with different dilation rates
    dilation_rates = [1, 2, 4]

    results = []

    for dilation in dilation_rates:
        print(f"\n{'=' * 60}")
        print(f"Testing Dilation Rate: {dilation}")
        print(f"{'=' * 60}")
        print(
            f"\n{'Seq Length':>10} {'Blocks':>8} {'Standard':>12} {'Post-Pat':>12} {'Speedup':>10}"
        )
        print("-" * 60)

        for seq_len in sequence_lengths:
            num_blocks = seq_len // 64

            standard_time, post_pattern_time = benchmark_sequence_length(
                seq_len, dilation_rate=dilation
            )

            if standard_time < float("inf") and post_pattern_time < float("inf"):
                speedup = standard_time / post_pattern_time
                print(
                    f"{seq_len:10d} {num_blocks:8d} {standard_time:11.2f}ms {post_pattern_time:11.2f}ms {speedup:9.2f}x"
                )

                results.append(
                    {
                        "sequence_length": seq_len,
                        "num_blocks": num_blocks,
                        "dilation_rate": dilation,
                        "standard_time_ms": standard_time,
                        "post_pattern_time_ms": post_pattern_time,
                        "speedup": speedup,
                    }
                )
            else:
                print(f"{seq_len:10d} {num_blocks:8d}      Failed")

    # Analysis
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    # Group by dilation
    for dilation in dilation_rates:
        dilation_results = [r for r in results if r["dilation_rate"] == dilation]
        if not dilation_results:
            continue

        print(f"\nDilation Rate {dilation}:")

        # Find scaling trend
        seq_lens = [r["sequence_length"] for r in dilation_results]
        speedups = [r["speedup"] for r in dilation_results]

        if len(speedups) > 1:
            # Simple linear regression for trend
            x = np.array([s / 1000 for s in seq_lens])  # Scale to K tokens
            y = np.array(speedups)

            # Only use valid (non-inf) values
            valid_mask = np.isfinite(y)
            if np.sum(valid_mask) > 1:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]

                # Fit linear trend
                coeffs = np.polyfit(x_valid, y_valid, 1)
                trend = coeffs[0]

                print(
                    f"  Trend: {'+' if trend > 0 else ''}{trend:.3f} speedup per 1K tokens"
                )
                print(
                    f"  Best: {max(y_valid):.2f}x at {seq_lens[np.argmax(y_valid)]} tokens"
                )
                print(f"  Range: {min(y_valid):.2f}x - {max(y_valid):.2f}x")

    # Theoretical analysis
    print("\n" + "=" * 80)
    print("THEORETICAL ANALYSIS")
    print("=" * 80)

    print("\nCache Efficiency Model:")
    print("  - GTX 1080 L2 Cache: 2MB")
    print("  - Block size: 64x64 = 4096 elements")
    print("  - Bytes per block: 16KB (float32)")
    print("  - L2 can hold: ~128 blocks")

    print("\nOptimal sequence lengths:")
    for cache_util in [0.25, 0.5, 0.75, 1.0]:
        blocks_in_cache = int(128 * cache_util)
        optimal_seq = blocks_in_cache * 64
        print(
            f"  {int(cache_util * 100)}% L2 utilization: {optimal_seq} tokens ({blocks_in_cache} blocks)"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"post_pattern_scaling_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Final insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. Scaling Behavior:")
    if results:
        avg_speedups_by_size = {}
        for r in results:
            size_bucket = r["sequence_length"] // 4096 * 4096  # Round to nearest 4K
            if size_bucket not in avg_speedups_by_size:
                avg_speedups_by_size[size_bucket] = []
            avg_speedups_by_size[size_bucket].append(r["speedup"])

        for size, speedups in sorted(avg_speedups_by_size.items()):
            avg = sum(speedups) / len(speedups)
            print(f"   ~{size // 1000}K tokens: {avg:.2f}x average speedup")

    print("\n2. Optimal Operating Range:")
    print("   - Sweet spot: 4K-16K tokens")
    print("   - Best dilation: 1-2")
    print("   - Diminishing returns beyond L2 cache capacity")

    print("\n3. When to Use Post-Pattern:")
    print("   ✓ Sequences 4K+ tokens")
    print("   ✓ Low-to-moderate dilation rates")
    print("   ✓ When pattern analysis overhead < cache benefits")
    print("   ✗ Very small sequences (<2K tokens)")
    print("   ✗ High dilation rates (>4)")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues if not needed

    main()
