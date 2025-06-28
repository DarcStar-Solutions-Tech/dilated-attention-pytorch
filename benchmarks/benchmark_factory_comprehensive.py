#!/usr/bin/env python3
"""
Comprehensive benchmark of factory pattern with auto-enable.
Tests different implementations and sequence lengths.
"""

import gc
import time
import torch
from statistics import mean, stdev
from tabulate import tabulate
from dilated_attention_pytorch.core import (
    create_dilated_attention,
    create_multihead_dilated_attention,
)


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_attention(attention, input_shape, num_runs=20, multihead=False):
    """Benchmark attention module."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if multihead:
        batch_size, seq_len, embed_dim = input_shape
        x = torch.randn(
            batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
        )

        # Warmup
        for _ in range(3):
            _ = attention(x, x, x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = attention(x, x, x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)
    else:
        batch_size, seq_len, num_heads, head_dim = input_shape
        q = torch.randn(input_shape, device=device, dtype=torch.float16)
        k = torch.randn(input_shape, device=device, dtype=torch.float16)
        v = torch.randn(input_shape, device=device, dtype=torch.float16)

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
            times.append((end - start) * 1000)

    # Get memory stats
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0

    return {
        "avg_ms": mean(times),
        "std_ms": stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_mb": peak_memory,
    }


def test_factory_auto_enable():
    """Test factory auto-enable across different scenarios."""
    print("=" * 80)
    print("Factory Pattern Auto-Enable Comprehensive Benchmark")
    print("=" * 80)

    # Test configurations
    test_configs = [
        # (impl_type, seq_len, batch_size, description)
        ("improved", 512, 2, "Short seq - should disable pool"),
        ("improved", 2048, 2, "Medium seq - should use lightweight"),
        ("improved", 4096, 1, "Long seq - should use lightweight"),
        ("improved", 8192, 1, "Very long seq - should use full pool"),
        ("standard", 1024, 2, "Standard impl - short seq"),
        ("standard", 4096, 1, "Standard impl - long seq"),
    ]

    # Special implementations that always enable pools
    if torch.cuda.is_available():
        test_configs.extend(
            [
                ("ring", 512, 1, "Ring - always enables pool"),
                ("block_sparse_ring", 1024, 1, "Block sparse - always enables"),
            ]
        )

    results = []

    for impl_type, seq_len, batch_size, description in test_configs:
        print(f"\n{description}")
        print("-" * 60)

        try:
            clear_memory()

            # Create with factory (auto-enable)
            print(f"Creating {impl_type} attention (seq_len={seq_len})...")
            attention = create_dilated_attention(
                impl_type,
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
            )

            # Check configuration
            enable_pool = getattr(attention, "enable_memory_pool", None)
            lightweight = getattr(attention, "lightweight_pool", None)
            has_pool = (
                hasattr(attention, "_memory_pool")
                and attention._memory_pool is not None
            )

            print(
                f"  Auto-configured: enable_memory_pool={enable_pool}, "
                f"lightweight_pool={lightweight}, has_pool={has_pool}"
            )

            # Benchmark
            shape = (batch_size, seq_len, 8, 64)
            auto_result = benchmark_attention(attention, shape)

            del attention
            clear_memory()

            # Create without pool for comparison
            attention_no_pool = create_dilated_attention(
                impl_type,
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                enable_memory_pool=False,
            )

            no_pool_result = benchmark_attention(attention_no_pool, shape)

            del attention_no_pool
            clear_memory()

            # Calculate metrics
            speedup = no_pool_result["avg_ms"] / auto_result["avg_ms"]
            mem_diff = auto_result["peak_mb"] - no_pool_result["peak_mb"]

            results.append(
                {
                    "Implementation": impl_type,
                    "Seq Len": seq_len,
                    "Pool Enabled": str(enable_pool),
                    "Lightweight": str(lightweight),
                    "Auto Time (ms)": f"{auto_result['avg_ms']:.2f}",
                    "No Pool Time (ms)": f"{no_pool_result['avg_ms']:.2f}",
                    "Speedup": f"{speedup:.2f}x",
                    "Auto Mem (MB)": f"{auto_result['peak_mb']:.1f}",
                    "No Pool Mem (MB)": f"{no_pool_result['peak_mb']:.1f}",
                    "Mem Diff (MB)": f"{mem_diff:+.1f}",
                }
            )

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    if results:
        print(tabulate(results, headers="keys", tablefmt="grid"))

    # Test multihead attention
    print("\n" + "=" * 80)
    print("MULTIHEAD ATTENTION AUTO-ENABLE TEST")
    print("=" * 80)

    multihead_configs = [
        (768, 12, 512, "Short sequence"),
        (768, 12, 4096, "Long sequence"),
    ]

    for embed_dim, num_heads, seq_len, description in multihead_configs:
        print(
            f"\n{description}: embed_dim={embed_dim}, heads={num_heads}, seq_len={seq_len}"
        )

        try:
            clear_memory()

            attention = create_multihead_dilated_attention(
                "improved",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
            )

            shape = (1, seq_len, embed_dim)
            result = benchmark_attention(attention, shape, multihead=True)

            print(
                f"  Time: {result['avg_ms']:.2f}ms, Memory: {result['peak_mb']:.1f}MB"
            )

            del attention
            clear_memory()

        except Exception as e:
            print(f"  Error: {e}")

    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    test_factory_auto_enable()
