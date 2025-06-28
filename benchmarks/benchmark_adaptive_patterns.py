#!/usr/bin/env python3
"""
Benchmark adaptive attention patterns against fixed patterns.

This script compares:
1. Fixed local window pattern
2. Fixed hierarchical pattern
3. Adaptive learned pattern
4. Dense baseline
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
from dilated_attention_pytorch.block_sparse_hierarchical import (
    BlockSparseHierarchical,
)
from dilated_attention_pytorch.block_sparse_adaptive import (
    AdaptiveConfig,
    create_adaptive_block_sparse,
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
    test_backward=False,
):
    """Benchmark a single implementation."""
    print(f"\n{name}:")

    # Create inputs
    query = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=test_backward,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    try:
        for _ in range(warmup):
            if test_backward:
                output = model(query, key, value, is_causal=False)
                loss = output.mean()
                loss.backward()
            else:
                with torch.no_grad():
                    _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        forward_times = []
        backward_times = []

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Forward pass
            start = time.perf_counter()
            if test_backward:
                output = model(query, key, value, is_causal=False)
            else:
                with torch.no_grad():
                    output = model(query, key, value, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            forward_time = time.perf_counter() - start
            forward_times.append(forward_time)

            # Backward pass if requested
            if test_backward:
                start = time.perf_counter()
                loss = output.mean()
                loss.backward()

                if device.type == "cuda":
                    torch.cuda.synchronize()

                backward_time = time.perf_counter() - start
                backward_times.append(backward_time)

        # Get memory usage
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        # Calculate statistics
        forward_mean = np.mean(forward_times) * 1000  # ms
        forward_std = np.std(forward_times) * 1000

        print(f"  Forward: {forward_mean:.2f} ms (±{forward_std:.2f})")

        if test_backward:
            backward_mean = np.mean(backward_times) * 1000
            backward_std = np.std(backward_times) * 1000
            total_mean = (np.mean(forward_times) + np.mean(backward_times)) * 1000
            print(f"  Backward: {backward_mean:.2f} ms (±{backward_std:.2f})")
            print(f"  Total: {total_mean:.2f} ms")

        print(f"  Peak memory: {peak_memory:.2f} MB")

        # Get pattern stats if available
        if hasattr(model, "get_pattern_stats"):
            stats = model.get_pattern_stats(seq_len)
            print(f"  Sparsity: {stats['sparsity']:.1%}")

        return {
            "name": name,
            "forward_mean_ms": forward_mean,
            "forward_std_ms": forward_std,
            "backward_mean_ms": backward_mean if test_backward else None,
            "backward_std_ms": backward_std if test_backward else None,
            "total_mean_ms": total_mean if test_backward else forward_mean,
            "peak_memory_mb": peak_memory,
            "success": True,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {
            "name": name,
            "success": False,
            "error": str(e),
        }
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def test_pattern_adaptation(seq_len, batch_size, num_heads, head_dim, device, dtype):
    """Test how adaptive patterns change with different inputs."""
    print("\n" + "=" * 80)
    print("PATTERN ADAPTATION TEST")
    print("=" * 80)

    # Create adaptive model
    adaptive_config = AdaptiveConfig(
        base_sparsity=0.9,
        temperature=0.5,
        hard_sparsity=True,
    )

    model = create_adaptive_block_sparse(
        embed_dim=num_heads * head_dim,
        num_heads=num_heads,
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        adaptive_config=adaptive_config,
        device=device,
        dtype=dtype,
    )

    # Test with different input patterns
    test_cases = [
        (
            "Random",
            lambda: torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            ),
        ),
        (
            "Periodic",
            lambda: torch.sin(
                torch.arange(seq_len, device=device, dtype=dtype).view(1, -1, 1, 1)
                * 0.1
            ).expand(batch_size, -1, num_heads, head_dim),
        ),
        (
            "Sparse",
            lambda: torch.zeros(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            .scatter_(
                1,
                torch.randint(
                    0, seq_len, (batch_size, seq_len // 10, 1, 1), device=device
                ),
                1.0,
            )
            .expand(-1, -1, num_heads, head_dim),
        ),
    ]

    pattern_stats = []

    for name, input_fn in test_cases:
        print(f"\n{name} input:")

        # Generate input
        q = input_fn()
        k = q.clone()
        v = q.clone()

        # Get pattern
        with torch.no_grad():
            _, pattern_info = model(q, k, v, return_pattern=True)

        # Analyze pattern for first head
        row_idx, col_idx = pattern_info["patterns"][0]
        num_connections = len(row_idx)

        # Calculate actual sparsity
        num_blocks = seq_len // model.block_size
        total_possible = num_blocks * num_blocks
        actual_sparsity = 1.0 - (num_connections / total_possible)

        print(f"  Connections: {num_connections}")
        print(f"  Sparsity: {actual_sparsity:.1%}")

        pattern_stats.append(
            {
                "input_type": name,
                "num_connections": num_connections,
                "sparsity": actual_sparsity,
            }
        )

    return pattern_stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive patterns")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument(
        "--test_backward", action="store_true", help="Test backward pass"
    )
    parser.add_argument(
        "--test_adaptation", action="store_true", help="Test pattern adaptation"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Adaptive Attention Patterns Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Test backward: {args.test_backward}")

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

    # Test pattern adaptation if requested
    if args.test_adaptation:
        pattern_stats = test_pattern_adaptation(
            args.seq_len, args.batch_size, args.num_heads, args.head_dim, device, dtype
        )

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
            test_backward=args.test_backward,
        )
    )
    del baseline

    # 2. Fixed local window
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.1,
        block_size=64,
        local_window_size=256,
    )

    local_window = BlockSparseRingDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        ring_size=1,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Fixed Local Window",
            local_window,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
            test_backward=args.test_backward,
        )
    )
    del local_window

    # 3. Fixed hierarchical
    hierarchical = BlockSparseHierarchical(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Fixed Hierarchical",
            hierarchical,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
            test_backward=args.test_backward,
        )
    )
    del hierarchical

    # 4. Adaptive (default config)
    adaptive_default = create_adaptive_block_sparse(
        embed_dim=args.num_heads * args.head_dim,
        num_heads=args.num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Adaptive (default)",
            adaptive_default,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
            test_backward=args.test_backward,
        )
    )
    del adaptive_default

    # 5. Adaptive (optimized)
    adaptive_config = AdaptiveConfig(
        base_sparsity=0.95,
        temperature=0.5,
        hard_sparsity=True,
        hidden_dim=64,
        num_layers=1,
        share_across_heads=True,
    )

    adaptive_optimized = create_adaptive_block_sparse(
        embed_dim=args.num_heads * args.head_dim,
        num_heads=args.num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        adaptive_config=adaptive_config,
        device=device,
        dtype=dtype,
    )
    results.append(
        benchmark_implementation(
            "Adaptive (optimized)",
            adaptive_optimized,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
            test_backward=args.test_backward,
        )
    )
    del adaptive_optimized

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"adaptive-patterns-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Adaptive Attention Patterns Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Data type: {dtype}\n")
        f.write(f"- Sequence length: {args.seq_len}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n\n")

        if args.test_adaptation and "pattern_stats" in locals():
            f.write("## Pattern Adaptation Results\n\n")
            f.write("| Input Type | Connections | Sparsity |\n")
            f.write("|------------|-------------|----------|\n")
            for stat in pattern_stats:
                f.write(
                    f"| {stat['input_type']} | {stat['num_connections']} | "
                    f"{stat['sparsity']:.1%} |\n"
                )
            f.write("\n")

        f.write("## Performance Results\n\n")

        if args.test_backward:
            f.write(
                "| Implementation | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) |\n"
            )
            f.write(
                "|----------------|--------------|---------------|------------|-------------|\n"
            )
        else:
            f.write(
                "| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |\n"
            )
            f.write(
                "|----------------|-----------|-------------|--------------------|\n"
            )

        if all(r["success"] for r in results):
            baseline_time = results[0]["total_mean_ms"]

            for r in results:
                speedup = baseline_time / r["total_mean_ms"]

                if args.test_backward:
                    f.write(
                        f"| {r['name']} | {r['forward_mean_ms']:.2f} | "
                        f"{r['backward_mean_ms']:.2f} | {r['total_mean_ms']:.2f} | "
                        f"{r['peak_memory_mb']:.2f} |\n"
                    )
                else:
                    f.write(
                        f"| {r['name']} | {r['total_mean_ms']:.2f} | "
                        f"{r['peak_memory_mb']:.2f} | {speedup:.2f}x |\n"
                    )

        f.write("\n## Key Findings\n\n")

        if all(r["success"] for r in results):
            # Find best performers
            best_time = min(results, key=lambda x: x["total_mean_ms"])
            best_memory = min(results, key=lambda x: x["peak_memory_mb"])

            f.write(
                f"- Fastest: {best_time['name']} ({best_time['total_mean_ms']:.2f} ms)\n"
            )
            f.write(
                f"- Most memory efficient: {best_memory['name']} "
                f"({best_memory['peak_memory_mb']:.2f} MB)\n"
            )

            # Compare adaptive vs fixed
            fixed_local = next(r for r in results if "Fixed Local" in r["name"])
            adaptive_default = next(
                r for r in results if "Adaptive (default)" in r["name"]
            )

            perf_diff = (
                (fixed_local["total_mean_ms"] - adaptive_default["total_mean_ms"])
                / fixed_local["total_mean_ms"]
                * 100
            )

            if perf_diff > 0:
                f.write(
                    f"- Adaptive is {perf_diff:.1f}% faster than fixed local window\n"
                )
            else:
                f.write(
                    f"- Adaptive is {-perf_diff:.1f}% slower than fixed local window\n"
                )

            f.write("\n### Adaptive Pattern Advantages:\n")
            f.write("- Content-aware sparsity adapts to input characteristics\n")
            f.write("- Learnable patterns can capture task-specific dependencies\n")
            f.write("- Flexible sparsity ratio based on sequence complexity\n")

    print(f"\n\nResults saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all(r["success"] for r in results):
        baseline_time = results[0]["total_mean_ms"]

        for r in results:
            speedup = baseline_time / r["total_mean_ms"]
            print(f"{r['name']:25} {r['total_mean_ms']:8.2f} ms ({speedup:5.2f}x)")


if __name__ == "__main__":
    main()
