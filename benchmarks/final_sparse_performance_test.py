#!/usr/bin/env python3
"""
Final test of sparse performance after all fixes.
"""

import torch
import time
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_all_sparse_configs():
    """Comprehensive benchmark of sparse configurations."""

    print("=== Final Sparse Performance Test ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Test matrix
    configs = [
        # (seq_len, dilation_rate, batch_size, description)
        (2048, 2, 1, "2K d=2 B=1"),
        (2048, 2, 2, "2K d=2 B=2"),
        (4096, 2, 1, "4K d=2 B=1"),
        (4096, 2, 2, "4K d=2 B=2"),
        (4096, 4, 1, "4K d=4 B=1"),
        (4096, 4, 2, "4K d=4 B=2"),
        (8192, 4, 1, "8K d=4 B=1"),
        (8192, 4, 2, "8K d=4 B=2"),
    ]

    implementations = [
        ("Unified", UnifiedHilbertAttention),
        ("Optimized", UnifiedHilbertAttentionOptimized),
        ("Enhanced", UnifiedHilbertAttentionOptimizedEnhanced),
    ]

    results = {}

    for seq_len, dilation_rate, batch_size, desc in configs:
        print(f"\n{desc}:")
        results[desc] = {}

        x = torch.randn(batch_size, seq_len, 512).cuda()

        for impl_name, impl_class in implementations:
            config = {
                "hidden_dim": 512,
                "num_heads": 8,
                "segment_size": 128,
                "dilation_rate": dilation_rate,
                "hilbert_threshold": 1024,
            }

            if impl_name == "Enhanced":
                config["enable_8k_optimization"] = True
                config["enable_multi_row"] = True

            try:
                module = impl_class(**config).cuda().eval()

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = module(x)
                torch.cuda.synchronize()

                # Benchmark
                torch.cuda.synchronize()
                start = time.perf_counter()

                for _ in range(10):
                    with torch.no_grad():
                        _ = module(x)
                    torch.cuda.synchronize()

                end = time.perf_counter()
                avg_time = (end - start) / 10 * 1000  # ms

                results[desc][impl_name] = avg_time
                print(f"  {impl_name}: {avg_time:.2f}ms")

            except Exception as e:
                print(f"  {impl_name}: Failed - {e}")
                results[desc][impl_name] = None

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Performance by configuration
    ax1.set_title("Sparse Attention Performance Comparison")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Time (ms)")

    labels = list(results.keys())
    x_pos = range(len(labels))

    width = 0.25
    for i, impl_name in enumerate(["Unified", "Optimized", "Enhanced"]):
        times = [results[cfg].get(impl_name, 0) for cfg in labels]
        ax1.bar([p + width * (i - 1) for p in x_pos], times, width, label=impl_name)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Speedup vs Unified
    ax2.set_title("Speedup vs Unified Implementation")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Speedup Factor")

    for impl_name in ["Optimized", "Enhanced"]:
        speedups = []
        for cfg in labels:
            if results[cfg].get("Unified") and results[cfg].get(impl_name):
                speedup = results[cfg]["Unified"] / results[cfg][impl_name]
                speedups.append(speedup)
            else:
                speedups.append(1.0)

        ax2.plot(x_pos, speedups, "o-", label=impl_name, linewidth=2, markersize=8)

    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Baseline")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("final_sparse_performance.png", dpi=150)
    print("\n✓ Plot saved to final_sparse_performance.png")

    # Summary statistics
    print("\n\n=== Summary Statistics ===")

    # Calculate average performance
    for impl_name in ["Unified", "Optimized", "Enhanced"]:
        times = []
        for cfg in results:
            if results[cfg].get(impl_name) is not None:
                times.append(results[cfg][impl_name])

        if times:
            avg_time = sum(times) / len(times)
            print(f"\n{impl_name}:")
            print(f"  Average time: {avg_time:.2f}ms")

            if impl_name != "Unified":
                # Calculate average speedup
                speedups = []
                for cfg in results:
                    if results[cfg].get("Unified") and results[cfg].get(impl_name):
                        speedup = results[cfg]["Unified"] / results[cfg][impl_name]
                        speedups.append(speedup)

                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    print(f"  Average speedup vs Unified: {avg_speedup:.2f}x")

    # Find best implementation for each config
    print("\n\nBest Implementation by Configuration:")
    for cfg in results:
        times = {k: v for k, v in results[cfg].items() if v is not None}
        if times:
            best_impl = min(times, key=times.get)
            best_time = times[best_impl]
            print(f"  {cfg}: {best_impl} ({best_time:.2f}ms)")


if __name__ == "__main__":
    benchmark_all_sparse_configs()
