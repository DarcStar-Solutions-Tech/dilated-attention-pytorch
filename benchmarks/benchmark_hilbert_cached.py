#!/usr/bin/env python3
"""
Benchmark comparing standard, original Hilbert V1, and cached Hilbert implementations.

This tests whether caching Hilbert orderings improves performance.
"""

import torch
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_cached import (
    create_cached_block_sparse_hilbert,
)


@dataclass
class CachedHilbertResult:
    """Results from benchmarking cached Hilbert implementation."""

    implementation: str  # "standard", "hilbert_v1", or "hilbert_cached"
    sequence_length: int
    forward_time_ms: float
    memory_mb: float
    preprocessing_time_ms: Optional[float] = None
    cache_stats: Optional[Dict] = None


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_implementation(
    implementation: str,
    seq_lengths: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 64,
    sparsity_ratio: float = 0.05,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> List[CachedHilbertResult]:
    """Benchmark a specific implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"\n{implementation.upper()} Implementation:")
    print("-" * 60)

    # Create model once for cached version to benefit from pre-computation
    if implementation == "cached":
        # Pre-create model with all sequence lengths
        model = create_cached_block_sparse_hilbert(
            segment_lengths=[max(seq_lengths) // 2],
            dilation_rates=[1],
            sparsity_ratio=sparsity_ratio,
            pattern_type="dilated_sparse",
            block_size=block_size,
            use_hilbert=True,
            precompute_seq_lengths=seq_lengths,  # Pre-compute all test lengths
        ).to(device)

        # Measure preprocessing time
        preprocess_start = time.perf_counter()
        # Force pre-computation by accessing cache stats
        cache_stats = (
            model.get_cache_stats() if hasattr(model, "get_cache_stats") else {}
        )
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000
        print(f"  Preprocessing time: {preprocess_time:.2f}ms")
        print(f"  Cache stats: {cache_stats}")

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model for standard and V1
            if implementation == "standard":
                model = create_block_sparse_attention(
                    variant="base",
                    segment_lengths=[seq_len // 2],
                    dilation_rates=[1],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=sparsity_ratio,
                        block_size=block_size,
                    ),
                ).to(device)
            elif implementation == "hilbert_v1":
                model = create_block_sparse_attention(
                    variant="hilbert",
                    segment_lengths=[seq_len // 2],
                    dilation_rates=[1],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=sparsity_ratio,
                        block_size=block_size,
                    ),
                    hilbert_block_level=True,
                    hilbert_within_blocks=False,
                ).to(device)

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup
            for _ in range(warmup_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Measure memory
            mem_before = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

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

            # Measure memory
            mem_after = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )
            memory_used = mem_after - mem_before

            print(f"  Seq {seq_len:6d}: {elapsed_time:7.2f}ms, {memory_used:7.1f}MB")

            result = CachedHilbertResult(
                implementation=implementation,
                sequence_length=seq_len,
                forward_time_ms=elapsed_time,
                memory_mb=memory_used,
            )

            if implementation == "cached":
                result.preprocessing_time_ms = preprocess_time / len(
                    seq_lengths
                )  # Amortized
                result.cache_stats = (
                    model.get_cache_stats() if hasattr(model, "get_cache_stats") else {}
                )

            results.append(result)

            # Cleanup for non-cached versions
            if implementation != "cached":
                del model
            del q, k, v, output

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Seq {seq_len:6d}: OOM")
                break
            else:
                print(f"  Seq {seq_len:6d}: Error - {str(e)[:50]}")
                break

    return results


def main():
    """Run comprehensive benchmarks comparing all implementations."""

    print("=" * 80)
    print("Cached Hilbert Implementation Benchmarks")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Test configurations
    seq_lengths = [2048, 4096, 8192, 16384, 32768]

    # Run benchmarks
    standard_results = benchmark_implementation("standard", seq_lengths)
    hilbert_v1_results = benchmark_implementation("hilbert_v1", seq_lengths)
    cached_results = benchmark_implementation("cached", seq_lengths)

    # Analysis
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print("\nSpeedups vs Standard:")
    print(f"{'Seq Length':>10} {'Hilbert V1':>12} {'Cached':>12} {'Improvement':>12}")
    print("-" * 48)

    for i, std in enumerate(standard_results):
        if i < len(hilbert_v1_results) and i < len(cached_results):
            v1 = hilbert_v1_results[i]
            cached = cached_results[i]

            v1_speedup = std.forward_time_ms / v1.forward_time_ms
            cached_speedup = std.forward_time_ms / cached.forward_time_ms
            improvement = cached.forward_time_ms / v1.forward_time_ms

            print(
                f"{std.sequence_length:10d} {v1_speedup:12.2f}x {cached_speedup:12.2f}x {improvement:12.2f}x"
            )

    # Calculate averages
    v1_speedups = []
    cached_speedups = []
    improvements = []

    for i, std in enumerate(standard_results):
        if i < len(hilbert_v1_results) and i < len(cached_results):
            v1_speedups.append(
                std.forward_time_ms / hilbert_v1_results[i].forward_time_ms
            )
            cached_speedups.append(
                std.forward_time_ms / cached_results[i].forward_time_ms
            )
            improvements.append(
                cached_results[i].forward_time_ms
                / hilbert_v1_results[i].forward_time_ms
            )

    if v1_speedups:
        print("\nAverages:")
        print(f"  Hilbert V1 vs Standard: {sum(v1_speedups) / len(v1_speedups):.2f}x")
        print(
            f"  Cached vs Standard: {sum(cached_speedups) / len(cached_speedups):.2f}x"
        )
        print(f"  Cached vs Hilbert V1: {sum(improvements) / len(improvements):.2f}x")

    # Memory comparison
    print("\n" + "=" * 80)
    print("MEMORY USAGE COMPARISON")
    print("=" * 80)

    print(f"\n{'Seq Length':>10} {'Standard':>12} {'Hilbert V1':>12} {'Cached':>12}")
    print("-" * 48)

    for i, std in enumerate(standard_results):
        if i < len(hilbert_v1_results) and i < len(cached_results):
            print(
                f"{std.sequence_length:10d} {std.memory_mb:11.1f}MB {hilbert_v1_results[i].memory_mb:11.1f}MB {cached_results[i].memory_mb:11.1f}MB"
            )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"hilbert_cached_benchmark_{timestamp}.json"

    all_results = []
    for results in [standard_results, hilbert_v1_results, cached_results]:
        for r in results:
            all_results.append(
                {
                    "implementation": r.implementation,
                    "sequence_length": r.sequence_length,
                    "forward_time_ms": r.forward_time_ms,
                    "memory_mb": r.memory_mb,
                    "preprocessing_time_ms": r.preprocessing_time_ms,
                    "cache_stats": r.cache_stats,
                }
            )

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. Caching Benefits:")
    print("   - Eliminates repeated Hilbert index computation")
    print("   - Amortizes preprocessing cost across forward passes")
    print("   - Most effective for repeated inference on same sequence lengths")

    print("\n2. When to Use Cached Hilbert:")
    print("   - Production inference with known sequence lengths")
    print("   - Repeated forward passes on same model")
    print("   - When memory for cache is available")

    print("\n3. Trade-offs:")
    print("   - Small memory overhead for storing orderings")
    print("   - Initial preprocessing time")
    print("   - Best gains when sequence lengths are known in advance")


if __name__ == "__main__":
    main()
