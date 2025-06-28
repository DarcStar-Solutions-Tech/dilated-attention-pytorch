#!/usr/bin/env python3
"""
Benchmark ImprovedDilatedAttention with enhanced memory pool integration.

This script measures performance improvements from memory pool integration:
- Memory usage reduction
- Allocation overhead reduction
- Processing speed improvements
- Maximum sequence length capabilities
- Multi-GPU scaling performance
"""

import argparse
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        # For CPU, use tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)


def benchmark_attention_memory_pool(
    batch_size: int = 2,
    seq_len: int = 8192,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 10,
    device: torch.device = torch.device("cpu"),
    enable_profiling: bool = False,
) -> dict:
    """
    Benchmark attention with and without memory pool.

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
    print(f"\nBenchmarking ImprovedDilatedAttention on {device}")
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
        "improvements": {},
    }

    # Test configuration (use smaller segments for better pool utilization)
    segment_lengths = [1024, 2048, 4096]
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
    attention_no_pool = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=False,
    ).to(device)

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

    # Test WITH memory pool
    print("  Testing with memory pool...")
    attention_with_pool = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=True,
        enable_profiling=enable_profiling,
    ).to(device)

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
            "total_allocations": profiling_report.count("Total Allocations:"),
            "pool_efficiency": "Efficient"
            if "bucketed" in profiling_report
            else "Standard",
        }

    # Calculate improvements
    no_pool = results["without_memory_pool"]
    with_pool = results["with_memory_pool"]

    time_improvement = (
        (no_pool["avg_time_per_iteration"] - with_pool["avg_time_per_iteration"])
        / no_pool["avg_time_per_iteration"]
    ) * 100
    memory_improvement = (
        (no_pool["peak_memory_mb"] - with_pool["peak_memory_mb"])
        / no_pool["peak_memory_mb"]
    ) * 100

    results["improvements"] = {
        "time_improvement_percent": time_improvement,
        "memory_improvement_percent": memory_improvement,
        "memory_reduction_mb": no_pool["peak_memory_mb"] - with_pool["peak_memory_mb"],
    }

    # Print results
    print(
        f"    Without pool: {no_pool['avg_time_per_iteration']:.4f}s/iter, {no_pool['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    With pool:    {with_pool['avg_time_per_iteration']:.4f}s/iter, {with_pool['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    Improvements: {time_improvement:.1f}% faster, {memory_improvement:.1f}% less memory"
    )

    del attention_with_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_sequence_length_scaling(
    device: torch.device = torch.device("cpu"),
    enable_profiling: bool = False,
) -> dict:
    """
    Benchmark maximum sequence length capabilities with memory pool.

    Args:
        device: Device to test on
        enable_profiling: Enable memory profiling

    Returns:
        Scaling benchmark results
    """
    print(f"\nBenchmarking sequence length scaling on {device}")

    results = {
        "device": str(device),
        "sequence_lengths": [],
        "without_pool": [],
        "with_pool": [],
        "max_seq_without_pool": 0,
        "max_seq_with_pool": 0,
    }

    # Test sequence lengths (must be divisible by largest segment length)
    # With segment_lengths=[1024, 2048, 4096], largest is 4096
    base_seq_lengths = [4096, 8192, 16384, 32768, 65536]
    if device.type == "cuda":
        # More aggressive scaling for GPU
        base_seq_lengths.extend([262144, 524288])

    batch_size = 2
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    for seq_len in base_seq_lengths:
        print(f"  Testing seq_len={seq_len}")

        try:
            # Test without memory pool
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

            attention_no_pool = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                enable_memory_pool=False,
            ).to(device)

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

            del attention_no_pool, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

            results["max_seq_without_pool"] = seq_len

        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"    Without pool: OOM at seq_len={seq_len}")
            no_pool_time = float("inf")
            no_pool_memory = float("inf")

        try:
            # Test with memory pool
            if "query" not in locals():
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

            attention_with_pool = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                enable_memory_pool=True,
                enable_profiling=enable_profiling,
            ).to(device)

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

            del attention_with_pool, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

            results["max_seq_with_pool"] = seq_len

        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"    With pool: OOM at seq_len={seq_len}")
            with_pool_time = float("inf")
            with_pool_memory = float("inf")

        # Store results
        results["sequence_lengths"].append(seq_len)
        results["without_pool"].append(
            {
                "time": no_pool_time,
                "memory_mb": no_pool_memory,
                "success": no_pool_time != float("inf"),
            }
        )
        results["with_pool"].append(
            {
                "time": with_pool_time,
                "memory_mb": with_pool_memory,
                "success": with_pool_time != float("inf"),
            }
        )

        print(
            f"    Results: no_pool={no_pool_time:.3f}s/{no_pool_memory:.1f}MB, with_pool={with_pool_time:.3f}s/{with_pool_memory:.1f}MB"
        )

        # Clean up
        if "query" in locals():
            del query, key, value
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Stop if both failed
        if no_pool_time == float("inf") and with_pool_time == float("inf"):
            break

    print(
        f"  Max sequence length: without_pool={results['max_seq_without_pool']}, with_pool={results['max_seq_with_pool']}"
    )

    return results


def benchmark_multi_gpu_scaling(enable_profiling: bool = False) -> dict:
    """
    Benchmark multi-GPU scaling performance with memory pools.

    Args:
        enable_profiling: Enable memory profiling

    Returns:
        Multi-GPU benchmark results
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Multi-GPU benchmarking skipped - insufficient GPUs")
        return {"error": "Insufficient GPUs"}

    print(f"\nBenchmarking multi-GPU scaling ({torch.cuda.device_count()} GPUs)")

    results = {
        "num_gpus": torch.cuda.device_count(),
        "devices": [],
        "single_gpu": {},
        "multi_gpu": {},
    }

    # Configuration
    batch_size = 4  # Larger batch for multi-GPU
    seq_len = 16384
    num_heads = 16
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Test single GPU (GPU 0)
    print("  Testing single GPU...")
    device = torch.device("cuda:0")

    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )

    attention_single = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=True,
        enable_profiling=enable_profiling,
    ).to(device)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(5):  # Multiple iterations for stability
        output = attention_single(query, key, value)
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    single_gpu_time = (end_time - start_time) / 5
    single_gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    results["single_gpu"] = {
        "time_per_iteration": single_gpu_time,
        "memory_mb": single_gpu_memory,
        "device": str(device),
    }

    print(f"    Single GPU: {single_gpu_time:.4f}s/iter, {single_gpu_memory:.1f}MB")

    del attention_single, output, query, key, value
    torch.cuda.empty_cache()

    # Test multi-GPU (data parallel simulation)
    print("  Testing multi-GPU simulation...")
    gpu_results = []

    for gpu_id in range(min(2, torch.cuda.device_count())):  # Test up to 2 GPUs
        device = torch.device(f"cuda:{gpu_id}")
        results["devices"].append(str(device))

        # Smaller batch per GPU
        per_gpu_batch = batch_size // 2

        query = torch.randn(
            per_gpu_batch,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        key = torch.randn(
            per_gpu_batch,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        value = torch.randn(
            per_gpu_batch,
            seq_len,
            num_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )

        attention_gpu = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            enable_memory_pool=True,
            enable_profiling=enable_profiling,
        ).to(device)

        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        for _ in range(5):
            output = attention_gpu(query, key, value)
            torch.cuda.synchronize(device)

        end_time = time.perf_counter()
        gpu_time = (end_time - start_time) / 5
        gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        gpu_results.append(
            {
                "device": str(device),
                "time_per_iteration": gpu_time,
                "memory_mb": gpu_memory,
                "batch_size": per_gpu_batch,
            }
        )

        print(
            f"    {device}: {gpu_time:.4f}s/iter, {gpu_memory:.1f}MB (batch={per_gpu_batch})"
        )

        del attention_gpu, output, query, key, value
        torch.cuda.empty_cache()

    # Calculate multi-GPU efficiency
    max_gpu_time = max(result["time_per_iteration"] for result in gpu_results)
    total_memory = sum(result["memory_mb"] for result in gpu_results)

    results["multi_gpu"] = {
        "max_time_per_iteration": max_gpu_time,
        "total_memory_mb": total_memory,
        "gpu_results": gpu_results,
        "efficiency": single_gpu_time / max_gpu_time if max_gpu_time > 0 else 0,
        "memory_scaling": total_memory / single_gpu_memory
        if single_gpu_memory > 0
        else 0,
    }

    print(f"    Multi-GPU efficiency: {results['multi_gpu']['efficiency']:.2f}x")
    print(f"    Memory scaling: {results['multi_gpu']['memory_scaling']:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ImprovedDilatedAttention memory pool integration"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
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
        "--test-scaling", action="store_true", help="Test sequence length scaling"
    )
    parser.add_argument(
        "--test-multi-gpu", action="store_true", help="Test multi-GPU scaling"
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

    print("ImprovedDilatedAttention Memory Pool Integration Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CUDA GPUs available: {torch.cuda.device_count()}")

    all_results = {}

    # Basic memory pool benchmark
    all_results["basic"] = benchmark_attention_memory_pool(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        num_iterations=args.iterations,
        device=device,
        enable_profiling=args.enable_profiling,
    )

    # Sequence length scaling
    if args.test_scaling:
        all_results["scaling"] = benchmark_sequence_length_scaling(
            device=device,
            enable_profiling=args.enable_profiling,
        )

    # Multi-GPU scaling
    if args.test_multi_gpu:
        all_results["multi_gpu"] = benchmark_multi_gpu_scaling(
            enable_profiling=args.enable_profiling,
        )

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"improved-attention-memory-pool-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# ImprovedDilatedAttention Memory Pool Integration Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Num Heads: {args.num_heads}\n")
        f.write(f"- Head Dim: {args.head_dim}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        # Basic results
        if "basic" in all_results:
            basic = all_results["basic"]
            f.write("## Basic Performance Comparison\n\n")
            f.write("| Metric | Without Pool | With Pool | Improvement |\n")
            f.write("|--------|--------------|-----------|-------------|\n")
            f.write(
                f"| Time per iteration | {basic['without_memory_pool']['avg_time_per_iteration']:.4f}s | {basic['with_memory_pool']['avg_time_per_iteration']:.4f}s | {basic['improvements']['time_improvement_percent']:.1f}% |\n"
            )
            f.write(
                f"| Peak Memory | {basic['without_memory_pool']['peak_memory_mb']:.1f}MB | {basic['with_memory_pool']['peak_memory_mb']:.1f}MB | {basic['improvements']['memory_improvement_percent']:.1f}% |\n"
            )
            f.write(
                f"| Memory Reduction | - | - | {basic['improvements']['memory_reduction_mb']:.1f}MB |\n\n"
            )

        # Scaling results
        if "scaling" in all_results:
            scaling = all_results["scaling"]
            f.write("## Sequence Length Scaling\n\n")
            f.write(
                f"- Maximum sequence length without pool: {scaling['max_seq_without_pool']:,}\n"
            )
            f.write(
                f"- Maximum sequence length with pool: {scaling['max_seq_with_pool']:,}\n"
            )
            f.write(
                f"- Improvement: {((scaling['max_seq_with_pool'] - scaling['max_seq_without_pool']) / scaling['max_seq_without_pool'] * 100):.1f}%\n\n"
            )

        # Multi-GPU results
        if "multi_gpu" in all_results and "error" not in all_results["multi_gpu"]:
            mgpu = all_results["multi_gpu"]
            f.write("## Multi-GPU Scaling\n\n")
            f.write(f"- Number of GPUs: {mgpu['num_gpus']}\n")
            f.write(
                f"- Single GPU time: {mgpu['single_gpu']['time_per_iteration']:.4f}s\n"
            )
            f.write(
                f"- Multi-GPU max time: {mgpu['multi_gpu']['max_time_per_iteration']:.4f}s\n"
            )
            f.write(f"- Scaling efficiency: {mgpu['multi_gpu']['efficiency']:.2f}x\n\n")

        f.write("## Key Findings\n\n")

        if "basic" in all_results:
            time_imp = all_results["basic"]["improvements"]["time_improvement_percent"]
            memory_imp = all_results["basic"]["improvements"][
                "memory_improvement_percent"
            ]

            if time_imp > 0:
                f.write(
                    f"- ✅ **Performance improvement**: {time_imp:.1f}% faster processing\n"
                )
            elif time_imp > -5:
                f.write(
                    f"- ✅ **Negligible overhead**: Only {abs(time_imp):.1f}% slower\n"
                )
            else:
                f.write(f"- ⚠️ **Performance cost**: {abs(time_imp):.1f}% slower\n")

            if memory_imp > 0:
                f.write(
                    f"- ✅ **Memory efficiency**: {memory_imp:.1f}% less memory usage\n"
                )
            else:
                f.write(
                    f"- ⚠️ **Memory overhead**: {abs(memory_imp):.1f}% more memory usage\n"
                )

        f.write("\n### Memory Pool Features:\n")
        f.write("- Enhanced memory pool integration with ImprovedDilatedAttention\n")
        f.write("- Automatic strategy selection (auto, bucketed, fragment-aware)\n")
        f.write("- Temporary tensor pooling for scatter operations\n")
        f.write("- Optional memory profiling and monitoring\n")
        f.write("- Thread-safe operations for concurrent attention\n")

    print(f"\nBenchmark results saved to: {report_path}")

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 20)

    if "basic" in all_results:
        basic = all_results["basic"]
        print(
            f"Time improvement: {basic['improvements']['time_improvement_percent']:.1f}%"
        )
        print(
            f"Memory improvement: {basic['improvements']['memory_improvement_percent']:.1f}%"
        )
        print(f"Memory reduction: {basic['improvements']['memory_reduction_mb']:.1f}MB")

    if "scaling" in all_results:
        scaling = all_results["scaling"]
        improvement = (
            (scaling["max_seq_with_pool"] - scaling["max_seq_without_pool"])
            / scaling["max_seq_without_pool"]
            * 100
        )
        print(f"Max sequence improvement: {improvement:.1f}%")

    if "multi_gpu" in all_results and "error" not in all_results["multi_gpu"]:
        mgpu = all_results["multi_gpu"]
        print(f"Multi-GPU efficiency: {mgpu['multi_gpu']['efficiency']:.2f}x")


if __name__ == "__main__":
    main()
