#!/usr/bin/env python3
"""
Benchmark hardware-specific tuning for Hilbert Attention kernels.

Tests the performance improvements from adaptive block sizes and implementation selection.
"""

import torch
import torch.nn as nn
import time
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

# Import the kernels
from dilated_attention_pytorch.kernels import UnifiedHilbertAttention
from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def get_gpu_info():
    """Get GPU information for the benchmark."""
    if not torch.cuda.is_available():
        return "CPU", "N/A", 0

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    return name, f"{capability[0]}.{capability[1]}", memory_gb


def benchmark_implementation(
    module: nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_warmup: int = 5,
    num_runs: int = 20,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """Benchmark a single implementation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move module to device and set dtype
    module = module.to(device)
    if dtype == torch.float16:
        module = module.half()

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    forward_times = []
    for _ in range(num_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x)

        if device == "cuda":
            torch.cuda.synchronize()

        forward_times.append((time.perf_counter() - start) * 1000)  # ms

    # Measure memory
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = module(x)

        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    else:
        peak_memory = 0

    return {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "memory_mb": peak_memory,
    }


def test_sequence_scaling():
    """Test performance across different sequence lengths."""
    print("\n=== Testing Sequence Length Scaling ===\n")

    gpu_name, compute_cap, memory_gb = get_gpu_info()
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: {compute_cap}")
    print(f"Memory: {memory_gb:.1f} GB\n")

    # Configuration
    batch_size = 2
    hidden_dim = 768
    num_heads = 12
    segment_size = 128
    dilation_rate = 2

    sequence_lengths = [128, 256, 512, 768, 1024, 1536, 2048]

    # Create modules
    core_module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        use_custom_backward=False,  # For fair comparison
    )

    simple_module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    )

    # Results storage
    core_results = []
    simple_results = []

    print(
        "Sequence | Core (ms) | Simple (ms) | Speedup | Core Mem | Simple Mem | Mem Reduction"
    )
    print("-" * 85)

    for seq_len in sequence_lengths:
        # Skip if too large for GPU memory
        estimated_memory = (
            batch_size * seq_len * hidden_dim * 4 * 6
        ) / 1e9  # Rough estimate
        if estimated_memory > memory_gb * 0.8:
            print(
                f"{seq_len:8d} | SKIPPED - Estimated {estimated_memory:.1f} GB > {memory_gb * 0.8:.1f} GB available"
            )
            continue

        # Benchmark Core
        try:
            core_stats = benchmark_implementation(
                core_module, batch_size, seq_len, hidden_dim
            )
            core_results.append(core_stats)
        except Exception as e:
            print(f"{seq_len:8d} | Core failed: {str(e)}")
            core_results.append({"forward_mean": float("inf"), "memory_mb": 0})
            continue

        # Benchmark Simple
        try:
            simple_stats = benchmark_implementation(
                simple_module, batch_size, seq_len, hidden_dim
            )
            simple_results.append(simple_stats)
        except Exception as e:
            print(f"{seq_len:8d} | Simple failed: {str(e)}")
            simple_results.append({"forward_mean": float("inf"), "memory_mb": 0})
            continue

        # Calculate metrics
        speedup = simple_stats["forward_mean"] / core_stats["forward_mean"]
        mem_reduction = (
            (simple_stats["memory_mb"] - core_stats["memory_mb"])
            / simple_stats["memory_mb"]
            * 100
        )

        # Check if Triton was used
        if hasattr(core_module, "should_use_triton"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            uses_triton = core_module.should_use_triton(seq_len, device)
            triton_marker = "*" if uses_triton else " "
        else:
            triton_marker = ""

        print(
            f"{seq_len:8d}{triton_marker}| {core_stats['forward_mean']:9.2f} | {simple_stats['forward_mean']:11.2f} | "
            f"{speedup:7.2f}x | {core_stats['memory_mb']:8.1f} | {simple_stats['memory_mb']:10.1f} | {mem_reduction:12.1f}%"
        )

    print("\n* = Using Triton kernel (adaptive selection)")

    # Plot results
    valid_indices = [
        i for i, r in enumerate(core_results) if r["forward_mean"] != float("inf")
    ]
    if valid_indices:
        plot_results(
            [sequence_lengths[i] for i in valid_indices],
            [core_results[i] for i in valid_indices],
            [simple_results[i] for i in valid_indices],
            gpu_name,
        )


def plot_results(
    seq_lengths: List[int],
    core_results: List[Dict],
    simple_results: List[Dict],
    gpu_name: str,
):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Extract data
    core_times = [r["forward_mean"] for r in core_results]
    simple_times = [r["forward_mean"] for r in simple_results]
    speedups = [s / c for c, s in zip(core_times, simple_times)]

    # Performance plot
    ax1.plot(seq_lengths, core_times, "b-o", label="UnifiedHilbertAttention", linewidth=2)
    ax1.plot(
        seq_lengths, simple_times, "r-s", label="UnifiedHilbertAttention", linewidth=2
    )
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Forward Pass Time (ms)")
    ax1.set_title(f"Performance Comparison on {gpu_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Speedup plot
    ax2.plot(seq_lengths, speedups, "g-^", linewidth=2)
    ax2.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Speedup Factor")
    ax2.set_title("Triton Kernel Speedup vs PyTorch")
    ax2.grid(True, alpha=0.3)

    # Add annotations for key transitions
    for i, (seq, speedup) in enumerate(zip(seq_lengths, speedups)):
        if i > 0 and (speedup < 1 and speedups[i - 1] >= 1):
            ax2.annotate(
                f"Crossover\n@ {seq}",
                xy=(seq, speedup),
                xytext=(seq, speedup + 0.5),
                arrowprops=dict(arrowstyle="->", color="red"),
                ha="center",
            )

    plt.tight_layout()
    plt.savefig("hardware_tuning_results.png", dpi=150, bbox_inches="tight")
    plt.show()


def test_block_size_impact():
    """Test the impact of different block sizes."""
    print("\n=== Testing Block Size Impact ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping block size test")
        return

    # Test configuration
    _ = 2
    seq_len = 1024
    hidden_dim = 768
    num_heads = 12

    # Create a module to get its block size selection
    module = UnifiedHilbertAttention(
        hidden_dim=hidden_dim, num_heads=num_heads, segment_size=128, dilation_rate=2
    )

    device = torch.device("cuda")
    BLOCK_M, BLOCK_N, BLOCK_D = module.get_optimal_block_sizes(seq_len, device)

    print(f"Optimal block sizes for seq_len={seq_len}:")
    print(f"  BLOCK_M: {BLOCK_M}")
    print(f"  BLOCK_N: {BLOCK_N}")
    print(f"  BLOCK_D: {BLOCK_D}")

    # Show how block sizes change with sequence length
    print("\nBlock size selection by sequence length:")
    print("Seq Length | BLOCK_M | BLOCK_N | BLOCK_D")
    print("-" * 40)

    for test_seq in [64, 128, 256, 512, 768, 1024, 2048, 4096]:
        M, N, D = module.get_optimal_block_sizes(test_seq, device)
        print(f"{test_seq:10d} | {M:7d} | {N:7d} | {D:7d}")


if __name__ == "__main__":
    print("=" * 80)
    print("Hardware-Specific Tuning Benchmark")
    print("=" * 80)

    # Run tests
    test_sequence_scaling()
    test_block_size_impact()

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("Results saved to: hardware_tuning_results.png")
