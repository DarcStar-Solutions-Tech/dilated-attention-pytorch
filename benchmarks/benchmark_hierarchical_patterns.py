#!/usr/bin/env python3
"""
Benchmark hierarchical attention patterns against other sparse patterns.

This script compares:
1. Original Block-Sparse (local window)
2. Optimized Block-Sparse (local window)
3. Hierarchical Block-Sparse (multi-scale)
4. Dense baseline (ImprovedDilatedAttention)
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
from dilated_attention_pytorch.block_sparse_hierarchical import (
    BlockSparseHierarchical,
    get_hierarchical_presets,
)
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

        # Benchmark
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

        print(f"  Time: {mean_time:.2f} ms (±{std_time:.2f})")
        print(f"  Memory: {peak_memory:.2f} MB")

        # Get pattern stats if available
        if hasattr(model, "get_pattern_stats"):
            stats = model.get_pattern_stats(seq_len)
            print(f"  Sparsity: {stats['sparsity']:.1%}")
            print(f"  Active blocks: {stats['active_blocks']}/{stats['total_blocks']}")

        return {
            "name": name,
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
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
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def compare_hierarchical_presets(
    seq_len,
    batch_size,
    num_heads,
    head_dim,
    device,
    dtype,
    segment_lengths,
    dilation_rates,
):
    """Compare different hierarchical preset configurations."""
    print("\n" + "=" * 80)
    print("HIERARCHICAL PRESET COMPARISON")
    print("=" * 80)

    presets = get_hierarchical_presets()
    results = []

    for preset_name, hierarchical_config in presets.items():
        model = BlockSparseHierarchical(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            hierarchical_config=hierarchical_config,
            device=device,
            dtype=dtype,
        )

        result = benchmark_implementation(
            f"Hierarchical-{preset_name}",
            model,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device,
            dtype,
            runs=5,
        )

        if result["success"]:
            results.append(result)

        del model

    return results


def visualize_patterns(seq_len=1024):
    """Visualize different attention patterns for comparison."""
    print("\n" + "=" * 80)
    print("ATTENTION PATTERN VISUALIZATIONS")
    print("=" * 80)

    # Local window pattern
    print("\n1. Local Window Pattern (BlockSparseOptimized):")
    local_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.1,
        block_size=64,
        local_window_size=256,
    )
    _ = BlockSparseOptimized(
        segment_lengths=[2048],
        dilation_rates=[1],
        sparse_config=local_config,
        device=torch.device("cpu"),
    )
    # Simple visualization for local window
    print("Pattern: Each position attends to fixed local window")
    print(f"Window size: {local_config.local_window_size}")
    print(f"Block size: {local_config.block_size}")

    # Hierarchical pattern
    print("\n2. Hierarchical Pattern (BlockSparseHierarchical):")
    hierarchical_model = BlockSparseHierarchical(
        segment_lengths=[2048],
        dilation_rates=[1],
        device=torch.device("cpu"),
    )
    print(hierarchical_model.visualize_pattern(seq_len=min(seq_len, 512)))


def main():
    parser = argparse.ArgumentParser(description="Benchmark hierarchical patterns")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument(
        "--visualize", action="store_true", help="Show pattern visualizations"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Hierarchical Attention Pattern Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Adjust segment lengths based on sequence length
    if args.seq_len <= 2048:
        segment_lengths = [512, 1024, 2048]
    elif args.seq_len <= 4096:
        segment_lengths = [1024, 2048, 4096]
    else:
        segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Show pattern visualizations if requested
    if args.visualize:
        visualize_patterns(args.seq_len)

    # Main comparison
    print("\n" + "=" * 80)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 80)

    results = []

    # 1. Dense baseline
    baseline = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
    )
    results.append(
        benchmark_implementation(
            "Dense Baseline",
            baseline,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del baseline

    # 2. Original Block-Sparse (local window)
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.1,
        block_size=64,
        local_window_size=256,
    )

    original = BlockSparseRingDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        ring_size=1,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Original Block-Sparse",
            original,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del original

    # 3. Optimized Block-Sparse (local window)
    optimized = BlockSparseOptimized(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        ring_size=1,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Optimized Block-Sparse",
            optimized,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del optimized

    # 4. Hierarchical Block-Sparse (default config)
    hierarchical = BlockSparseHierarchical(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Hierarchical (default)",
            hierarchical,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del hierarchical

    # Compare hierarchical presets
    preset_results = compare_hierarchical_presets(
        args.seq_len,
        args.batch_size,
        args.num_heads,
        args.head_dim,
        device,
        dtype,
        segment_lengths,
        dilation_rates,
    )
    results.extend(preset_results)

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"hierarchical-patterns-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Hierarchical Attention Patterns Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Data type: {dtype}\n")
        f.write(f"- Sequence length: {args.seq_len}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n\n")

        f.write("## Results\n\n")
        f.write("| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |\n")
        f.write("|----------------|-----------|-------------|--------------------|\n")

        if all(r["success"] for r in results):
            baseline_time = results[0]["mean_time_ms"]

            for r in results:
                speedup = baseline_time / r["mean_time_ms"]
                f.write(
                    f"| {r['name']} | {r['mean_time_ms']:.2f} | "
                    f"{r['peak_memory_mb']:.2f} | {speedup:.2f}x |\n"
                )

        f.write("\n## Key Findings\n\n")

        if all(r["success"] for r in results):
            # Find best performers
            best_time = min(results, key=lambda x: x["mean_time_ms"])
            best_memory = min(results, key=lambda x: x["peak_memory_mb"])

            f.write(
                f"- Fastest: {best_time['name']} ({best_time['mean_time_ms']:.2f} ms)\n"
            )
            f.write(
                f"- Most memory efficient: {best_memory['name']} ({best_memory['peak_memory_mb']:.2f} MB)\n"
            )

            # Compare hierarchical vs local window
            local_optimized = next(
                r for r in results if r["name"] == "Optimized Block-Sparse"
            )
            hierarchical_default = next(
                r for r in results if r["name"] == "Hierarchical (default)"
            )

            perf_diff = (
                (local_optimized["mean_time_ms"] - hierarchical_default["mean_time_ms"])
                / local_optimized["mean_time_ms"]
                * 100
            )

            if perf_diff > 0:
                f.write(
                    f"- Hierarchical is {perf_diff:.1f}% faster than local window\n"
                )
            else:
                f.write(
                    f"- Hierarchical is {-perf_diff:.1f}% slower than local window\n"
                )

    print(f"\n\nResults saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all(r["success"] for r in results):
        baseline_time = results[0]["mean_time_ms"]

        for r in results:
            speedup = baseline_time / r["mean_time_ms"]
            print(f"{r['name']:30} {r['mean_time_ms']:8.2f} ms ({speedup:5.2f}x)")


if __name__ == "__main__":
    main()
