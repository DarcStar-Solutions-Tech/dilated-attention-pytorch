#!/usr/bin/env python3
"""
Benchmark optimized Block-Sparse implementation against original.

This script compares:
1. Original BlockSparseRingDilatedAttention
2. BlockSparseOptimized with enhanced caching and batching
3. Baseline ImprovedDilatedAttention for reference
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_optimized import BlockSparseOptimized
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_implementation(
    name,
    model,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    device,
    dtype,
    runs=10,
    warmup=3,
):
    """Benchmark a single implementation."""
    print(f"\n{name}:")

    # Create inputs
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    try:
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Measure forward pass
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(query, key, value, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Get memory usage
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        # Calculate statistics
        mean_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000

        print(f"  Mean time: {mean_time:.2f} ms (±{std_time:.2f})")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  Peak memory: {peak_memory:.2f} MB")

        # Get optimization stats if available
        if hasattr(model, "get_optimization_stats"):
            stats = model.get_optimization_stats()
            if "pattern_cache" in stats:
                cache_stats = stats["pattern_cache"]
                print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
                print(f"  Cached patterns: {cache_stats.get('total_patterns', 0)}")

        return {
            "name": name,
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "peak_memory_mb": peak_memory,
            "success": True,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "name": name,
            "success": False,
            "error": str(e),
        }
    finally:
        # Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optimized Block-Sparse implementation"
    )
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity ratio")
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Block-Sparse Optimization Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Sparsity: {args.sparsity}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Adjust segment lengths based on sequence length
    if args.seq_len <= 2048:
        segment_lengths = [512, 1024, 2048]
    elif args.seq_len <= 4096:
        segment_lengths = [1024, 2048, 4096]
    elif args.seq_len <= 8192:
        segment_lengths = [2048, 4096, 8192]
    else:
        segment_lengths = [4096, 8192, 16384]
    dilation_rates = [1, 2, 4]

    print(f"\nSegment lengths: {segment_lengths}")
    print(f"Dilation rates: {dilation_rates}")

    # Sparse configuration
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=args.sparsity,
        block_size=64,
        local_window_size=256,
    )

    results = []

    # 1. Baseline: ImprovedDilatedAttention
    print("\n" + "=" * 80)
    print("BASELINE")
    print("=" * 80)

    baseline = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
    )
    baseline_result = benchmark_implementation(
        "ImprovedDilatedAttention",
        baseline,
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        device,
        dtype,
        args.runs,
    )
    results.append(baseline_result)
    del baseline

    # 2. Original Block-Sparse
    print("\n" + "=" * 80)
    print("ORIGINAL BLOCK-SPARSE")
    print("=" * 80)

    original = BlockSparseRingDilatedAttention(
        ring_size=1,
        device=device,
        dtype=dtype,
        sparse_config=sparse_config,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
    )
    original_result = benchmark_implementation(
        "BlockSparse_Original",
        original,
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        device,
        dtype,
        args.runs,
    )
    results.append(original_result)
    del original

    # 3. Optimized Block-Sparse (with batching)
    print("\n" + "=" * 80)
    print("OPTIMIZED BLOCK-SPARSE")
    print("=" * 80)

    optimized = BlockSparseOptimized(
        ring_size=1,
        device=device,
        dtype=dtype,
        sparse_config=sparse_config,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_batched_ops=True,
        cache_size=100,
    )
    optimized_result = benchmark_implementation(
        "BlockSparse_Optimized",
        optimized,
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        device,
        dtype,
        args.runs,
    )
    results.append(optimized_result)

    # Test cache effectiveness
    print("\nTesting cache effectiveness...")
    _ = optimized.get_optimization_stats()["pattern_cache"]

    # Run multiple forward passes with same sequence length
    for _ in range(10):
        with torch.no_grad():
            _ = optimized(
                torch.randn(
                    args.batch_size,
                    args.seq_len,
                    args.num_heads,
                    args.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    args.batch_size,
                    args.seq_len,
                    args.num_heads,
                    args.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                torch.randn(
                    args.batch_size,
                    args.seq_len,
                    args.num_heads,
                    args.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                is_causal=False,
            )

    cache_stats_after = optimized.get_optimization_stats()["pattern_cache"]
    print(f"Cache hit rate after multiple runs: {cache_stats_after['hit_rate']:.2%}")
    print(f"Total cache accesses: {cache_stats_after['total_accesses']}")

    del optimized

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"block-sparse-optimized-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Block-Sparse Optimization Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Sequence length: {args.seq_len}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Sparsity: {args.sparsity}\n")
        f.write(f"- Segment lengths: {segment_lengths}\n")
        f.write(f"- Dilation rates: {dilation_rates}\n\n")

        f.write("## Results\n\n")

        f.write(
            "| Implementation | Mean Time (ms) | Std Dev | Memory (MB) | Speedup vs Original |\n"
        )
        f.write(
            "|---------------|----------------|---------|-------------|--------------------|\n"
        )

        if all(r["success"] for r in results):
            baseline_time = results[0]["mean_time_ms"]
            original_time = results[1]["mean_time_ms"]

            for r in results:
                speedup = original_time / r["mean_time_ms"]
                f.write(
                    f"| {r['name']} | {r['mean_time_ms']:.2f} | "
                    f"±{r['std_time_ms']:.2f} | {r['peak_memory_mb']:.2f} | "
                    f"{speedup:.2f}x |\n"
                )

        f.write("\n## Analysis\n\n")

        if all(r["success"] for r in results):
            baseline_time = results[0]["mean_time_ms"]
            original_time = results[1]["mean_time_ms"]
            optimized_time = results[2]["mean_time_ms"]

            improvement = (original_time - optimized_time) / original_time * 100
            f.write(f"- Optimization improvement: {improvement:.1f}%\n")
            f.write(f"- Optimized vs baseline: {baseline_time / optimized_time:.2f}x\n")
            f.write(f"- Original vs baseline: {baseline_time / original_time:.2f}x\n")

        f.write("\n## Cache Statistics\n\n")
        f.write(f"- Final cache hit rate: {cache_stats_after['hit_rate']:.2%}\n")
        f.write(f"- Total accesses: {cache_stats_after['total_accesses']}\n")
        f.write(f"- Cached patterns: {cache_stats_after['total_patterns']}\n")

    print(f"\n\nResults saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all(r["success"] for r in results):
        for r in results:
            print(f"{r['name']:30} {r['mean_time_ms']:8.2f} ms")

        improvement = (original_time - optimized_time) / original_time * 100
        print(f"\nOptimization improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
