#!/usr/bin/env python3
"""
Comprehensive benchmark of memory pool optimizations across all implementations.

This script tests DilatedAttention, ImprovedDilatedAttention, and RingDilatedAttentionV2
with the optimized memory pool configurations.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2


def benchmark_implementation(
    attention_class,
    class_name: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: list,
    dilation_rates: list,
    num_iterations: int,
    device: torch.device,
    ring_size: int = 4,
) -> dict:
    """Benchmark a single attention implementation."""
    print(f"\n{class_name}:")

    # Create test data
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    results = {}

    # Test without memory pool
    kwargs = {
        "segment_lengths": segment_lengths,
        "dilation_rates": dilation_rates,
        "enable_memory_pool": False,
    }
    if class_name == "RingDilatedAttentionV2":
        kwargs["ring_size"] = ring_size
        kwargs["device"] = device

    attention_no_pool = attention_class(**kwargs)
    if hasattr(attention_no_pool, "to"):
        attention_no_pool = attention_no_pool.to(device)

    # Warmup
    for _ in range(3):
        _ = attention_no_pool(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output = attention_no_pool(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    no_pool_time = (end_time - start_time) / num_iterations
    no_pool_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["without_pool"] = {
        "time_per_iter": no_pool_time,
        "peak_memory_mb": no_pool_memory,
    }

    # Cleanup
    if hasattr(attention_no_pool, "cleanup_buffers"):
        attention_no_pool.cleanup_buffers()
    del attention_no_pool, output
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test with memory pool (enabled explicitly)
    kwargs["enable_memory_pool"] = True
    kwargs["lightweight_pool"] = True

    attention_with_pool = attention_class(**kwargs)
    if hasattr(attention_with_pool, "to"):
        attention_with_pool = attention_with_pool.to(device)

    # Warmup
    for _ in range(3):
        _ = attention_with_pool(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = attention_with_pool(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    with_pool_time = (end_time - start_time) / num_iterations
    with_pool_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["with_pool"] = {
        "time_per_iter": with_pool_time,
        "peak_memory_mb": with_pool_memory,
    }

    # Calculate improvements
    time_diff = ((no_pool_time - with_pool_time) / no_pool_time) * 100
    memory_diff = (
        ((no_pool_memory - with_pool_memory) / no_pool_memory) * 100
        if no_pool_memory > 0
        else 0
    )

    results["improvements"] = {
        "time_improvement_percent": time_diff,
        "memory_improvement_percent": memory_diff,
    }

    # Print results
    print(f"  Without pool: {no_pool_time:.4f}s/iter, {no_pool_memory:.1f}MB")
    print(f"  With pool:    {with_pool_time:.4f}s/iter, {with_pool_memory:.1f}MB")
    print(f"  Improvement:  {time_diff:+.1f}% time, {memory_diff:+.1f}% memory")

    # Cleanup
    if hasattr(attention_with_pool, "cleanup_buffers"):
        attention_with_pool.cleanup_buffers()
    del attention_with_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark memory pool optimizations across all implementations"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=16384, help="Sequence length")
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
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Memory Pool Optimization Benchmark - All Implementations")
    print("=" * 60)
    print(f"Device: {device}")
    print(
        f"Configuration: batch={args.batch_size}, seq_len={args.seq_len}, heads={args.num_heads}, head_dim={args.head_dim}"
    )

    # Test configuration
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Benchmark all implementations
    all_results = {}

    implementations = [
        (DilatedAttention, "DilatedAttention"),
        (ImprovedDilatedAttention, "ImprovedDilatedAttention"),
        (RingDilatedAttentionV2, "RingDilatedAttentionV2"),
    ]

    for attention_class, class_name in implementations:
        try:
            results = benchmark_implementation(
                attention_class,
                class_name,
                args.batch_size,
                args.seq_len,
                args.num_heads,
                args.head_dim,
                segment_lengths,
                dilation_rates,
                args.iterations,
                device,
            )
            all_results[class_name] = results
        except Exception as e:
            print(f"  Error: {e}")
            all_results[class_name] = {"error": str(e)}

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"all-implementations-optimization-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Memory Pool Optimization Benchmark - All Implementations\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Num Heads: {args.num_heads}\n")
        f.write(f"- Head Dim: {args.head_dim}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- Segment Lengths: {segment_lengths}\n")
        f.write(f"- Dilation Rates: {dilation_rates}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        f.write("## Results Summary\n\n")
        f.write(
            "| Implementation | Without Pool | With Pool | Time Improvement | Memory Improvement |\n"
        )
        f.write(
            "|----------------|--------------|-----------|------------------|--------------------||\n"
        )

        for impl_name, results in all_results.items():
            if "error" in results:
                f.write(f"| {impl_name} | Error | Error | - | - |\n")
            else:
                no_pool = results["without_pool"]
                with_pool = results["with_pool"]
                improvements = results["improvements"]
                f.write(
                    f"| {impl_name} | {no_pool['time_per_iter']:.4f}s / {no_pool['peak_memory_mb']:.0f}MB | "
                    f"{with_pool['time_per_iter']:.4f}s / {with_pool['peak_memory_mb']:.0f}MB | "
                    f"{improvements['time_improvement_percent']:+.1f}% | "
                    f"{improvements['memory_improvement_percent']:+.1f}% |\n"
                )

        f.write("\n## Key Findings\n\n")
        f.write("### Optimizations Applied:\n")
        f.write("1. **1MB threshold**: Only use memory pool for tensors ≥ 1MB\n")
        f.write("2. **Disabled by default**: Memory pools are opt-in, not default\n")
        f.write(
            "3. **Smart allocation**: Avoid pool overhead for small temporary tensors\n"
        )
        f.write(
            "4. **Fixed SDPA warning**: Using `torch.is_grad_enabled()` instead of `.training`\n"
        )
        f.write("\n")

        f.write("### Performance Analysis:\n")
        for impl_name, results in all_results.items():
            if "error" not in results:
                time_imp = results["improvements"]["time_improvement_percent"]
                mem_imp = results["improvements"]["memory_improvement_percent"]

                f.write(f"\n**{impl_name}**:\n")
                if time_imp > 5:
                    f.write(f"- ✅ Time: {time_imp:.1f}% faster with pool\n")
                elif time_imp > -5:
                    f.write(f"- ✅ Time: Negligible overhead ({time_imp:.1f}%)\n")
                else:
                    f.write(f"- ⚠️ Time: {abs(time_imp):.1f}% slower with pool\n")

                if mem_imp > 5:
                    f.write(f"- ✅ Memory: {mem_imp:.1f}% reduction with pool\n")
                elif abs(mem_imp) < 5:
                    f.write(f"- ✅ Memory: Similar usage ({mem_imp:.1f}%)\n")
                else:
                    f.write(f"- ⚠️ Memory: {abs(mem_imp):.1f}% increase with pool\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
