#!/usr/bin/env python3
"""
Benchmark the fully optimized Hilbert implementation with zero forward-pass overhead.
"""

import torch
import time
import gc
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_optimized import (
    create_optimized_hilbert_attention,
)


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_sequence_length(
    model,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
    device: torch.device = None,
) -> float:
    """Benchmark a single sequence length."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(warmup_iters):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            _ = model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(benchmark_iters):
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            output = model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed_time = (time.perf_counter() - start_time) / benchmark_iters * 1000

    del q, k, v, output
    clear_gpu_memory()

    return elapsed_time


def main():
    """Run benchmarks comparing standard vs optimized Hilbert."""

    print("=" * 80)
    print("Optimized Hilbert Implementation Benchmark")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_lengths = [2048, 4096, 8192, 16384, 32768]

    # Configuration
    sparsity_ratio = 0.05  # 95% sparse
    block_size = 64
    num_heads = 8
    head_dim = 64

    # Create models
    print("\nCreating models...")

    # Standard model
    standard_model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[max(seq_lengths) // 2],
        dilation_rates=[1],
        sparse_config=SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        ),
    ).to(device)

    # Optimized Hilbert model with pre-computation
    print("Pre-computing Hilbert patterns...")
    precompute_start = time.perf_counter()

    hilbert_model = create_optimized_hilbert_attention(
        segment_lengths=[max(seq_lengths) // 2],
        dilation_rates=[1],
        sparsity_ratio=sparsity_ratio,
        pattern_type="dilated_sparse",
        block_size=block_size,
        precompute_seq_lengths=seq_lengths,  # Pre-compute all test lengths
    ).to(device)

    precompute_time = (time.perf_counter() - precompute_start) * 1000
    print(f"Pre-computation completed in {precompute_time:.2f}ms")

    cache_stats = hilbert_model.get_cache_stats()
    print(f"Cache stats: {cache_stats}")

    # Benchmark
    print("\n" + "-" * 80)
    print(
        f"{'Seq Length':>10} {'Standard (ms)':>15} {'Hilbert (ms)':>15} {'Speedup':>10}"
    )
    print("-" * 80)

    results = []
    for seq_len in seq_lengths:
        # Standard
        standard_time = benchmark_sequence_length(
            standard_model,
            seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )

        # Hilbert
        hilbert_time = benchmark_sequence_length(
            hilbert_model,
            seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )

        speedup = standard_time / hilbert_time
        print(
            f"{seq_len:10d} {standard_time:15.2f} {hilbert_time:15.2f} {speedup:10.2f}x"
        )

        results.append(
            {
                "seq_len": seq_len,
                "standard_ms": standard_time,
                "hilbert_ms": hilbert_time,
                "speedup": speedup,
            }
        )

    # Summary
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print("-" * 80)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(
        f"Amortized precompute cost per sequence: {precompute_time / len(seq_lengths):.2f}ms"
    )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if avg_speedup > 1.0:
        print("\n✓ Optimized Hilbert FASTER than standard!")
        print("  - Pre-computation eliminates forward-pass overhead")
        print("  - Direct indexing with no sorting required")
        print("  - Cache-friendly access patterns")
    else:
        print("\n✗ Optimized Hilbert still slower than standard")
        print("  - GPU memory access patterns favor sequential blocks")
        print("  - Hilbert reordering disrupts coalesced memory access")
        print("  - Overhead still exceeds cache benefits")

    print("\nKey Insights:")
    print("1. Pre-computation removes Hilbert computation from critical path")
    print("2. Performance depends heavily on GPU architecture")
    print("3. Benefits may emerge at larger scales or different hardware")

    # Test without pre-computation
    print("\n" + "=" * 80)
    print("TESTING WITHOUT PRE-COMPUTATION")
    print("=" * 80)

    # Clear cache to test on-the-fly computation
    hilbert_model.clear_cache()
    print("Cache cleared. Testing on-the-fly Hilbert computation...")

    test_seq = 8192
    no_cache_time = benchmark_sequence_length(
        hilbert_model, test_seq, num_heads=num_heads, head_dim=head_dim, device=device
    )

    # Now it should be cached
    with_cache_time = benchmark_sequence_length(
        hilbert_model, test_seq, num_heads=num_heads, head_dim=head_dim, device=device
    )

    print(f"\nSequence {test_seq}:")
    print(f"  First run (no cache): {no_cache_time:.2f}ms")
    print(f"  Second run (cached): {with_cache_time:.2f}ms")
    print(f"  Cache speedup: {no_cache_time / with_cache_time:.2f}x")


if __name__ == "__main__":
    main()
