#!/usr/bin/env python3
"""
Benchmark Ring Attention memory pool integration.

This script tests memory pool benefits specifically for Ring Attention,
focusing on the communication buffer allocation patterns and large
sequence length scenarios where Ring Attention provides the most value.
"""

import argparse
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


def benchmark_ring_attention_memory_pool(
    batch_size: int = 2,
    seq_len: int = 16384,
    num_heads: int = 8,
    head_dim: int = 64,
    ring_size: int = 4,
    num_iterations: int = 10,
    device: torch.device = torch.device("cpu"),
    enable_profiling: bool = False,
) -> dict:
    """
    Benchmark Ring Attention with and without memory pool.

    Args:
        batch_size: Batch size for attention
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Head dimension
        ring_size: Ring size for distributed simulation
        num_iterations: Number of iterations to run
        device: Device to run on
        enable_profiling: Enable memory profiling

    Returns:
        Benchmark results dictionary
    """
    print(f"\nBenchmarking RingDilatedAttentionV2 on {device}")
    print(
        f"Configuration: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, head_dim={head_dim}, ring_size={ring_size}"
    )

    results = {
        "configuration": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "ring_size": ring_size,
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
    attention_no_pool = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        ring_size=ring_size,
        device=device,
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

    # Cleanup
    attention_no_pool.cleanup_buffers()
    del attention_no_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH memory pool (full features)
    print("  Testing with full memory pool...")
    attention_with_pool = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        ring_size=ring_size,
        device=device,
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

    # Cleanup
    attention_with_pool.cleanup_buffers()
    del attention_with_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH lightweight memory pool
    print("  Testing with lightweight memory pool...")
    attention_lightweight = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        ring_size=ring_size,
        device=device,
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

    # Cleanup
    attention_lightweight.cleanup_buffers()
    del attention_lightweight, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_ring_size_scaling(
    device: torch.device = torch.device("cpu"),
    enable_profiling: bool = False,
) -> dict:
    """
    Benchmark Ring Attention scaling with different ring sizes.

    Args:
        device: Device to test on
        enable_profiling: Enable memory profiling

    Returns:
        Ring size scaling results
    """
    print(f"\nBenchmarking ring size scaling on {device}")

    results = {
        "device": str(device),
        "ring_sizes": [],
        "without_pool": [],
        "with_pool": [],
    }

    # Test configuration
    batch_size = 2
    seq_len = 16384
    num_heads = 8
    head_dim = 64
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Test different ring sizes
    ring_sizes = [1, 2, 4, 8]

    for ring_size in ring_sizes:
        print(f"  Testing ring_size={ring_size}")

        try:
            # Generate test data
            query = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            key = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            value = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )

            # Test without memory pool
            attention_no_pool = RingDilatedAttentionV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                device=device,
                enable_memory_pool=False,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            output = attention_no_pool(query, key, value)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            no_pool_time = end_time - start_time
            no_pool_memory = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0
            )

            attention_no_pool.cleanup_buffers()
            del attention_no_pool, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Test with lightweight memory pool
            attention_with_pool = RingDilatedAttentionV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                device=device,
                enable_memory_pool=True,
                enable_profiling=enable_profiling,
                lightweight_pool=True,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            output = attention_with_pool(query, key, value)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            with_pool_time = end_time - start_time
            with_pool_memory = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0
            )

            attention_with_pool.cleanup_buffers()
            del attention_with_pool, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # Store results
            results["ring_sizes"].append(ring_size)
            results["without_pool"].append(
                {
                    "time": no_pool_time,
                    "memory_mb": no_pool_memory,
                    "success": True,
                }
            )
            results["with_pool"].append(
                {
                    "time": with_pool_time,
                    "memory_mb": with_pool_memory,
                    "success": True,
                }
            )

            print(
                f"    Results: no_pool={no_pool_time:.3f}s/{no_pool_memory:.1f}MB, with_pool={with_pool_time:.3f}s/{with_pool_memory:.1f}MB"
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"    Failed with ring_size={ring_size}: {e}")
            results["ring_sizes"].append(ring_size)
            results["without_pool"].append({"success": False, "error": str(e)})
            results["with_pool"].append({"success": False, "error": str(e)})

        # Clean up
        if "query" in locals():
            del query, key, value
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ring Attention memory pool integration"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16384, help="Sequence length")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--ring-size", type=int, default=4, help="Ring size")
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
        "--test-scaling", action="store_true", help="Test ring size scaling"
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

    print("Ring Attention Memory Pool Integration Benchmark")
    print("=" * 50)
    print(f"Device: {device}")

    all_results = {}

    # Basic memory pool benchmark
    all_results["basic"] = benchmark_ring_attention_memory_pool(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        ring_size=args.ring_size,
        num_iterations=args.iterations,
        device=device,
        enable_profiling=args.enable_profiling,
    )

    # Ring size scaling
    if args.test_scaling:
        all_results["scaling"] = benchmark_ring_size_scaling(
            device=device,
            enable_profiling=args.enable_profiling,
        )

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"ring-attention-memory-pool-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Ring Attention Memory Pool Integration Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Num Heads: {args.num_heads}\n")
        f.write(f"- Head Dim: {args.head_dim}\n")
        f.write(f"- Ring Size: {args.ring_size}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        # Basic results
        if "basic" in all_results:
            basic = all_results["basic"]
            f.write("## Basic Performance Comparison\n\n")
            f.write(
                "| Configuration | Time per iteration | Peak Memory | Time Improvement | Memory Improvement |\n"
            )
            f.write(
                "|---------------|-------------------|-------------|------------------|--------------------|\n"
            )
            f.write(
                f"| Without Pool | {basic['without_memory_pool']['avg_time_per_iteration']:.4f}s | {basic['without_memory_pool']['peak_memory_mb']:.1f}MB | - | - |\n"
            )
            f.write(
                f"| Full Pool | {basic['with_memory_pool']['avg_time_per_iteration']:.4f}s | {basic['with_memory_pool']['peak_memory_mb']:.1f}MB | {basic['improvements']['full_pool']['time_improvement_percent']:.1f}% | {basic['improvements']['full_pool']['memory_improvement_percent']:.1f}% |\n"
            )
            f.write(
                f"| Lightweight Pool | {basic['with_lightweight_pool']['avg_time_per_iteration']:.4f}s | {basic['with_lightweight_pool']['peak_memory_mb']:.1f}MB | {basic['improvements']['lightweight_pool']['time_improvement_percent']:.1f}% | {basic['improvements']['lightweight_pool']['memory_improvement_percent']:.1f}% |\n\n"
            )

        # Scaling results
        if "scaling" in all_results:
            scaling = all_results["scaling"]
            f.write("## Ring Size Scaling\n\n")
            f.write(
                "| Ring Size | Without Pool (ms) | With Pool (ms) | Memory Without | Memory With |\n"
            )
            f.write(
                "|-----------|-------------------|----------------|----------------|-------------|\n"
            )

            for i, ring_size in enumerate(scaling["ring_sizes"]):
                no_pool = scaling["without_pool"][i]
                with_pool = scaling["with_pool"][i]
                if no_pool.get("success") and with_pool.get("success"):
                    f.write(
                        f"| {ring_size} | {no_pool['time'] * 1000:.2f} | {with_pool['time'] * 1000:.2f} | {no_pool['memory_mb']:.1f}MB | {with_pool['memory_mb']:.1f}MB |\n"
                    )
            f.write("\n")

        f.write("## Key Findings\n\n")

        if "basic" in all_results:
            light_time_imp = all_results["basic"]["improvements"]["lightweight_pool"][
                "time_improvement_percent"
            ]
            light_memory_imp = all_results["basic"]["improvements"]["lightweight_pool"][
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
                    f"- ⚠️ **Memory overhead**: {abs(light_memory_imp):.1f}% more memory usage initially\n"
                )

        f.write("\n### Ring Attention Memory Pool Benefits:\n")
        f.write("- Enhanced communication buffer management\n")
        f.write("- Optimized allocation for output accumulators\n")
        f.write("- Configurable pool strategies (full vs lightweight)\n")
        f.write("- Automatic buffer cleanup and reuse\n")
        f.write("- Optional memory profiling for distributed scenarios\n")
        f.write("- Reduced allocation overhead for large sequences\n")

    print(f"\nBenchmark results saved to: {report_path}")

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 20)

    if "basic" in all_results:
        basic = all_results["basic"]
        light_imp = basic["improvements"]["lightweight_pool"]
        print(
            f"Lightweight pool time improvement: {light_imp['time_improvement_percent']:.1f}%"
        )
        print(
            f"Lightweight pool memory improvement: {light_imp['memory_improvement_percent']:.1f}%"
        )
        print(f"Memory reduction: {light_imp['memory_reduction_mb']:.1f}MB")


if __name__ == "__main__":
    main()
