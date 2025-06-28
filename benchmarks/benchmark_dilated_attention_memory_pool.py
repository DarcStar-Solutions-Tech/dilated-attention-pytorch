#!/usr/bin/env python3
"""
Benchmark DilatedAttention memory pool integration.

This script tests the core DilatedAttention implementation with enhanced memory pool
to verify performance improvements and memory efficiency.
"""

import argparse
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.dilated_attention import DilatedAttention


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


def benchmark_dilated_attention_memory_pool(
    batch_size: int = 2,
    seq_len: int = 8192,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 10,
    device: torch.device = torch.device("cpu"),
    enable_profiling: bool = False,
) -> dict:
    """
    Benchmark DilatedAttention with and without memory pool.

    Args:
        batch_size: Batch size for attention
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        num_iterations: Number of iterations to run
        device: Device to run on
        enable_profiling: Enable memory profiling

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking DilatedAttention on {device}")
    print(
        f"Configuration: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}"
    )

    results = {
        "configuration": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_iterations": num_iterations,
            "device": str(device),
        },
        "without_memory_pool": {},
        "with_memory_pool": {},
        "with_lightweight_pool": {},
        "improvements": {},
    }

    # Test configuration
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Generate test data
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )

    # Test WITHOUT memory pool
    print("  Testing without memory pool...")
    attention_no_pool = DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=False,
    )

    # Warmup
    for _ in range(3):
        _ = attention_no_pool(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start_memory = get_memory_usage()
    start_time = time.perf_counter()

    for i in range(num_iterations):
        output = attention_no_pool(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    end_memory = get_memory_usage()
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    )
    tracemalloc.stop()

    results["without_memory_pool"] = {
        "total_time": end_time - start_time,
        "avg_time_per_iteration": (end_time - start_time) / num_iterations,
        "start_memory_mb": start_memory,
        "end_memory_mb": end_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": end_memory - start_memory,
    }

    del attention_no_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH full memory pool
    print("  Testing with full memory pool...")
    attention_with_pool = DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=True,
        enable_profiling=enable_profiling,
        lightweight_pool=False,
    )

    # Warmup
    for _ in range(3):
        _ = attention_with_pool(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start_memory = get_memory_usage()
    start_time = time.perf_counter()

    for i in range(num_iterations):
        output = attention_with_pool(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    end_memory = get_memory_usage()
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    )
    tracemalloc.stop()

    results["with_memory_pool"] = {
        "total_time": end_time - start_time,
        "avg_time_per_iteration": (end_time - start_time) / num_iterations,
        "start_memory_mb": start_memory,
        "end_memory_mb": end_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": end_memory - start_memory,
    }

    # Get memory pool statistics if profiling enabled
    if enable_profiling and attention_with_pool._memory_pool:
        pool_stats = attention_with_pool._memory_pool.get_stats()
        profiling_report = attention_with_pool._memory_pool.get_profiling_report()
        results["memory_pool_stats"] = pool_stats
        results["profiling_summary"] = {
            "report_available": "Total Allocations:" in profiling_report,
            "pool_efficiency": "bucketed" in profiling_report.lower(),
        }

    del attention_with_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH lightweight memory pool
    print("  Testing with lightweight memory pool...")
    attention_lightweight = DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=True,
        enable_profiling=False,  # Disable profiling for performance
        lightweight_pool=True,
    )

    # Warmup
    for _ in range(3):
        _ = attention_lightweight(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    tracemalloc.start()
    start_memory = get_memory_usage()
    start_time = time.perf_counter()

    for i in range(num_iterations):
        output = attention_lightweight(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    end_memory = get_memory_usage()
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else tracemalloc.get_traced_memory()[1] / (1024 * 1024)
    )
    tracemalloc.stop()

    results["with_lightweight_pool"] = {
        "total_time": end_time - start_time,
        "avg_time_per_iteration": (end_time - start_time) / num_iterations,
        "start_memory_mb": start_memory,
        "end_memory_mb": end_memory,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": end_memory - start_memory,
    }

    # Calculate improvements
    no_pool = results["without_memory_pool"]
    with_pool = results["with_memory_pool"]
    lightweight = results["with_lightweight_pool"]

    time_improvement_full = (
        (no_pool["avg_time_per_iteration"] - with_pool["avg_time_per_iteration"])
        / no_pool["avg_time_per_iteration"]
    ) * 100
    memory_improvement_full = (
        (no_pool["peak_memory_mb"] - with_pool["peak_memory_mb"])
        / no_pool["peak_memory_mb"]
    ) * 100

    time_improvement_light = (
        (no_pool["avg_time_per_iteration"] - lightweight["avg_time_per_iteration"])
        / no_pool["avg_time_per_iteration"]
    ) * 100
    memory_improvement_light = (
        (no_pool["peak_memory_mb"] - lightweight["peak_memory_mb"])
        / no_pool["peak_memory_mb"]
    ) * 100

    results["improvements"] = {
        "full_pool": {
            "time_improvement_percent": time_improvement_full,
            "memory_improvement_percent": memory_improvement_full,
            "memory_reduction_mb": no_pool["peak_memory_mb"]
            - with_pool["peak_memory_mb"],
        },
        "lightweight_pool": {
            "time_improvement_percent": time_improvement_light,
            "memory_improvement_percent": memory_improvement_light,
            "memory_reduction_mb": no_pool["peak_memory_mb"]
            - lightweight["peak_memory_mb"],
        },
    }

    # Print results
    print(
        f"    Without pool:     {no_pool['avg_time_per_iteration']:.4f}s/iter, {no_pool['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    With full pool:   {with_pool['avg_time_per_iteration']:.4f}s/iter, {with_pool['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    With light pool:  {lightweight['avg_time_per_iteration']:.4f}s/iter, {lightweight['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    Full pool improvements: {time_improvement_full:.1f}% time, {memory_improvement_full:.1f}% memory"
    )
    print(
        f"    Light pool improvements: {time_improvement_light:.1f}% time, {memory_improvement_light:.1f}% memory"
    )

    del attention_lightweight, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DilatedAttention memory pool integration"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--enable-profiling", action="store_true", help="Enable memory profiling"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("DilatedAttention Memory Pool Integration Benchmark")
    print("=" * 50)
    print(f"Device: {device}")

    # Run benchmark
    results = benchmark_dilated_attention_memory_pool(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_iterations=args.iterations,
        device=device,
        enable_profiling=args.enable_profiling,
    )

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"dilated-attention-memory-pool-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# DilatedAttention Memory Pool Integration Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Num Heads: {args.num_heads}\n")
        f.write(f"- Head Dim: {args.head_dim}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        # Results table
        f.write("## Performance Comparison\n\n")
        f.write(
            "| Configuration | Time per iteration | Peak Memory | Time Improvement | Memory Improvement |\n"
        )
        f.write(
            "|---------------|-------------------|-------------|------------------|--------------------||\n"
        )
        f.write(
            f"| Without Pool | {results['without_memory_pool']['avg_time_per_iteration']:.4f}s | {results['without_memory_pool']['peak_memory_mb']:.1f}MB | - | - |\n"
        )
        f.write(
            f"| Full Pool | {results['with_memory_pool']['avg_time_per_iteration']:.4f}s | {results['with_memory_pool']['peak_memory_mb']:.1f}MB | {results['improvements']['full_pool']['time_improvement_percent']:.1f}% | {results['improvements']['full_pool']['memory_improvement_percent']:.1f}% |\n"
        )
        f.write(
            f"| Lightweight Pool | {results['with_lightweight_pool']['avg_time_per_iteration']:.4f}s | {results['with_lightweight_pool']['peak_memory_mb']:.1f}MB | {results['improvements']['lightweight_pool']['time_improvement_percent']:.1f}% | {results['improvements']['lightweight_pool']['memory_improvement_percent']:.1f}% |\n\n"
        )

        f.write("## Key Findings\n\n")

        light_time_imp = results["improvements"]["lightweight_pool"][
            "time_improvement_percent"
        ]
        light_memory_imp = results["improvements"]["lightweight_pool"][
            "memory_improvement_percent"
        ]

        if light_time_imp > 5:
            f.write(
                f"- ✅ **Performance improvement**: {light_time_imp:.1f}% faster with lightweight pool\n"
            )
        elif light_time_imp > -5:
            f.write(
                f"- ✅ **Negligible overhead**: Only {abs(light_time_imp):.1f}% impact with lightweight pool\n"
            )
        else:
            f.write(
                f"- ⚠️ **Performance cost**: {abs(light_time_imp):.1f}% slower with lightweight pool\n"
            )

        if light_memory_imp > 0:
            f.write(
                f"- ✅ **Memory efficiency**: {light_memory_imp:.1f}% less memory usage\n"
            )
        else:
            f.write(
                f"- ⚠️ **Memory overhead**: {abs(light_memory_imp):.1f}% more memory usage\n"
            )

        f.write("\n### Memory Pool Features:\n")
        f.write("- Enhanced memory pool integration with DilatedAttention core\n")
        f.write("- Configurable pool strategies (full vs lightweight)\n")
        f.write("- Automatic strategy selection for tensor allocation\n")
        f.write("- Optional memory profiling and monitoring\n")
        f.write("- Based on lessons learned from ImprovedDilatedAttention\n")

    print(f"\nBenchmark results saved to: {report_path}")

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 20)
    light_imp = results["improvements"]["lightweight_pool"]
    print(
        f"Lightweight pool time improvement: {light_imp['time_improvement_percent']:.1f}%"
    )
    print(
        f"Lightweight pool memory improvement: {light_imp['memory_improvement_percent']:.1f}%"
    )
    print(f"Memory reduction: {light_imp['memory_reduction_mb']:.1f}MB")


if __name__ == "__main__":
    main()
