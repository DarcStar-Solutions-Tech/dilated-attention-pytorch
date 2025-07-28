#!/usr/bin/env python3
"""
Benchmark strided access optimization for dilated attention.
"""

import torch
import torch.nn as nn
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

# Import kernels
from dilated_attention_pytorch.kernels import HilbertAttentionCore
from dilated_attention_pytorch.kernels.hilbert_attention_strided_simple import (
    HilbertAttentionStridedSimple,
)


def measure_performance(
    module: nn.Module, x: torch.Tensor, num_warmup: int = 5, num_runs: int = 20
) -> Dict[str, float]:
    """Measure memory and time performance."""
    device = x.device

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # Measure memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    else:
        peak_memory = 0

    # Measure time
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "memory_mb": peak_memory,
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    }


def test_dilation_rates():
    """Test performance improvements with different dilation rates."""
    print("=== Strided Access Performance Test ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")
    print("Testing impact of strided access for dilated attention\n")

    # Configuration
    batch_size = 2
    seq_len = 2048
    hidden_dim = 768
    num_heads = 12
    segment_size = 256

    dilation_rates = [1, 2, 4, 8]

    results_original = []
    results_strided = []

    print(
        "Dilation | Original (ms) | Strided (ms) | Speedup | Mem Original | Mem Strided | Mem Reduction"
    )
    print("-" * 95)

    for dilation_rate in dilation_rates:
        # Create modules
        original = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        strided = HilbertAttentionStridedSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Benchmark
        try:
            res_orig = measure_performance(original, x)
            results_original.append(res_orig)
        except Exception as e:
            print(f"Original failed for dilation={dilation_rate}: {e}")
            res_orig = {"time_mean": float("inf"), "memory_mb": 0}
            results_original.append(res_orig)

        try:
            res_strided = measure_performance(strided, x)
            results_strided.append(res_strided)
        except Exception as e:
            print(f"Strided failed for dilation={dilation_rate}: {e}")
            res_strided = {"time_mean": float("inf"), "memory_mb": 0}
            results_strided.append(res_strided)

        # Calculate improvements
        if res_orig["time_mean"] != float("inf"):
            speedup = res_orig["time_mean"] / res_strided["time_mean"]
            mem_reduction = (
                (res_orig["memory_mb"] - res_strided["memory_mb"])
                / res_orig["memory_mb"]
                * 100
            )
        else:
            speedup = 0
            mem_reduction = 0

        print(
            f"{dilation_rate:8d} | {res_orig['time_mean']:13.2f} | {res_strided['time_mean']:12.2f} | "
            f"{speedup:7.2f}x | {res_orig['memory_mb']:12.1f} | {res_strided['memory_mb']:11.1f} | {mem_reduction:13.1f}%"
        )

        # Cleanup
        del x, original, strided
        torch.cuda.empty_cache()
        gc.collect()

    # Expected bandwidth reduction
    print("\n" + "=" * 95)
    print("Expected Bandwidth Reduction:")
    print("-" * 40)
    for d in dilation_rates:
        reduction = (1 - 1 / d) * 100
        print(f"Dilation {d}: {reduction:.1f}% bandwidth reduction")

    return dilation_rates, results_original, results_strided


def test_sequence_lengths():
    """Test strided access across different sequence lengths."""
    print("\n\n=== Sequence Length Scaling Test ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    # Configuration
    batch_size = 2
    hidden_dim = 768
    num_heads = 12
    segment_size = 256
    dilation_rate = 4  # Test with dilation=4

    seq_lengths = [512, 1024, 2048, 4096]

    print(f"Testing with dilation_rate={dilation_rate}")
    print("\nSeq Len | Original (ms) | Strided (ms) | Speedup | Memory Savings")
    print("-" * 70)

    speedups = []

    for seq_len in seq_lengths:
        # Skip if too large
        estimated_mem = (batch_size * seq_len * hidden_dim * 4 * 6) / 1e9
        if estimated_mem > 6.0:  # Conservative limit
            print(f"{seq_len:7d} | SKIPPED - Too large for GPU")
            continue

        # Create modules
        original = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        strided = HilbertAttentionStridedSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Benchmark
        res_orig = measure_performance(original, x)
        res_strided = measure_performance(strided, x)

        speedup = res_orig["time_mean"] / res_strided["time_mean"]
        mem_saving = (
            (res_orig["memory_mb"] - res_strided["memory_mb"])
            / res_orig["memory_mb"]
            * 100
        )

        speedups.append(speedup)

        print(
            f"{seq_len:7d} | {res_orig['time_mean']:13.2f} | {res_strided['time_mean']:12.2f} | "
            f"{speedup:7.2f}x | {mem_saving:14.1f}%"
        )

        # Cleanup
        del x, original, strided
        torch.cuda.empty_cache()
        gc.collect()

    return seq_lengths[: len(speedups)], speedups


def plot_results(dilation_rates, results_orig, results_strided):
    """Plot performance comparisons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data
    times_orig = [r["time_mean"] for r in results_orig]
    times_strided = [r["time_mean"] for r in results_strided]
    mem_orig = [r["memory_mb"] for r in results_orig]
    mem_strided = [r["memory_mb"] for r in results_strided]

    # Time comparison
    x = np.arange(len(dilation_rates))
    width = 0.35

    _ = ax1.bar(x - width / 2, times_orig, width, label="Original", color="#E74C3C")
    _ = ax1.bar(x + width / 2, times_strided, width, label="Strided", color="#27AE60")

    ax1.set_xlabel("Dilation Rate")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Performance: Original vs Strided Access")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dilation_rates)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add speedup labels
    for i, (t1, t2) in enumerate(zip(times_orig, times_strided)):
        if t1 != float("inf") and t2 != float("inf"):
            speedup = t1 / t2
            ax1.text(
                i,
                max(t1, t2) * 1.05,
                f"{speedup:.1f}x",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Memory comparison
    _ = ax2.bar(x - width / 2, mem_orig, width, label="Original", color="#E74C3C")
    _ = ax2.bar(x + width / 2, mem_strided, width, label="Strided", color="#27AE60")

    ax2.set_xlabel("Dilation Rate")
    ax2.set_ylabel("Memory (MB)")
    ax2.set_title("Memory Usage: Original vs Strided Access")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dilation_rates)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add reduction labels
    for i, (m1, m2) in enumerate(zip(mem_orig, mem_strided)):
        if m1 > 0:
            reduction = (m1 - m2) / m1 * 100
            ax2.text(
                i,
                max(m1, m2) * 1.05,
                f"-{reduction:.0f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig("strided_access_performance.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=" * 80)
    print("Strided Access Optimization Benchmark")
    print("=" * 80)

    # Test different dilation rates
    dilation_rates, results_orig, results_strided = test_dilation_rates()

    # Test sequence length scaling
    seq_lengths, speedups = test_sequence_lengths()

    # Plot results
    if len(results_orig) > 0:
        plot_results(dilation_rates, results_orig, results_strided)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("Results saved to: strided_access_performance.png")
    print("=" * 80)
