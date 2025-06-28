#!/usr/bin/env python3
"""
Benchmark to specifically test the factory auto-enable performance.
"""

import gc
import time
import torch
from statistics import mean, stdev
from dilated_attention_pytorch.core import create_dilated_attention


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_configuration(
    name, attention, batch_size, seq_len, num_heads, head_dim, num_runs=10
):
    """Benchmark a specific configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create inputs
    shape = (batch_size, seq_len, num_heads, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float16)
    k = torch.randn(shape, device=device, dtype=torch.float16)
    v = torch.randn(shape, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(3):
        _ = attention(q, k, v)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = attention(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Get memory stats
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0

    return {
        "name": name,
        "avg_time_ms": mean(times),
        "std_time_ms": stdev(times) if len(times) > 1 else 0,
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "peak_memory_mb": peak_memory,
    }


def main():
    """Run benchmarks comparing auto-enabled vs manually configured memory pools."""
    print("=" * 80)
    print("Factory Auto-Enable Performance Benchmark")
    print("=" * 80)

    # Test configurations
    configs = [
        # (batch_size, seq_len, description)
        (1, 512, "Short sequence (no pool)"),
        (1, 2048, "Medium sequence (lightweight pool)"),
        (1, 4096, "Long sequence (lightweight pool)"),
        (1, 8192, "Very long sequence (full pool)"),
    ]

    num_heads = 8
    head_dim = 64

    for batch_size, seq_len, description in configs:
        print(f"\n{description}: B={batch_size}, L={seq_len}")
        print("-" * 60)

        clear_memory()

        # Test 1: Factory with auto-enable
        print("Creating attention with factory auto-enable...")
        attention_auto = create_dilated_attention(
            "improved",
            segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
            dilation_rates=[1, 2, 4],
        )

        # Check what was auto-configured
        enable_pool = getattr(attention_auto, "enable_memory_pool", None)
        lightweight = getattr(attention_auto, "lightweight_pool", None)
        print(
            f"  Auto-configured: enable_memory_pool={enable_pool}, lightweight_pool={lightweight}"
        )

        result_auto = benchmark_configuration(
            "Auto-enabled", attention_auto, batch_size, seq_len, num_heads, head_dim
        )

        del attention_auto
        clear_memory()

        # Test 2: Manual configuration (no pool)
        print("\nCreating attention with memory pool disabled...")
        attention_no_pool = create_dilated_attention(
            "improved",
            segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
            dilation_rates=[1, 2, 4],
            enable_memory_pool=False,
        )

        result_no_pool = benchmark_configuration(
            "No pool", attention_no_pool, batch_size, seq_len, num_heads, head_dim
        )

        del attention_no_pool
        clear_memory()

        # Test 3: Manual configuration (with pool) - only if auto would enable
        if seq_len >= 2048:
            print("\nCreating attention with memory pool manually enabled...")
            attention_with_pool = create_dilated_attention(
                "improved",
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=True,
                lightweight_pool=(seq_len < 8192),
            )

            result_with_pool = benchmark_configuration(
                "Manual pool",
                attention_with_pool,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            )

            del attention_with_pool
            clear_memory()
        else:
            result_with_pool = None

        # Print results
        print("\nResults:")
        print(
            f"  {'Configuration':<15} {'Avg Time (ms)':<15} {'Std Dev':<10} {'Peak Mem (MB)':<15}"
        )
        print("  " + "-" * 55)

        for result in [result_auto, result_no_pool, result_with_pool]:
            if result:
                print(
                    f"  {result['name']:<15} {result['avg_time_ms']:<15.2f} {result['std_time_ms']:<10.2f} {result['peak_memory_mb']:<15.1f}"
                )

        # Calculate speedup/overhead
        if result_with_pool:
            speedup_vs_no_pool = (
                result_no_pool["avg_time_ms"] / result_auto["avg_time_ms"]
            )
            print(f"\n  Auto-enable speedup vs no pool: {speedup_vs_no_pool:.2f}x")

            overhead_vs_manual = (
                (result_auto["avg_time_ms"] - result_with_pool["avg_time_ms"])
                / result_with_pool["avg_time_ms"]
                * 100
            )
            print(f"  Auto-enable overhead vs manual pool: {overhead_vs_manual:+.1f}%")


if __name__ == "__main__":
    main()
