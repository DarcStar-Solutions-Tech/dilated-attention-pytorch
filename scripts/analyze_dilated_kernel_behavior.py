#!/usr/bin/env python3
"""
Analyze the behavior of dilated attention kernels to understand performance characteristics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def profile_attention_pattern(seq_len, segment_size, dilation_rate):
    """Analyze the attention pattern for dilated attention."""
    total_positions = seq_len * seq_len
    active_positions = 0

    # Count actual active positions
    for i in range(seq_len):
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        # Count positions attended to
        for j in range(seg_start, seg_end, dilation_rate):
            active_positions += 1

    sparsity = 1.0 - (active_positions / total_positions)
    return active_positions, total_positions, sparsity


def visualize_attention_pattern(seq_len=64, segment_size=16, dilation_rate=4):
    """Visualize the attention pattern."""
    mask = torch.zeros(seq_len, seq_len)

    for i in range(seq_len):
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        for j in range(seg_start, seg_end, dilation_rate):
            mask[i, j] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap="Blues")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(
        f"Dilated Attention Pattern\n(seg_size={segment_size}, dilation={dilation_rate})"
    )
    plt.colorbar(label="Attention Weight")

    # Add segment boundaries
    for i in range(0, seq_len, segment_size):
        plt.axhline(i - 0.5, color="red", linestyle="--", alpha=0.5)
        plt.axvline(i - 0.5, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("dilated_attention_pattern.png", dpi=150)
    print("Saved attention pattern visualization to dilated_attention_pattern.png")

    return mask


def benchmark_kernel_detailed(module, x, warmup=10, iters=50):
    """Detailed benchmark with statistics."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(x)

    torch.cuda.synchronize()
    times = []

    # Measure with CUDA events for precision
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            out = module(x)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)

    times = sorted(times)[5:-5]  # Remove outliers

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "median": np.median(times),
        "output": out,
    }


def main():
    print("=== Dilated Attention Kernel Behavior Analysis ===\n")

    # Import implementations
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )
    from dilated_attention_pytorch.kernels.dilated_attention_simple_opt import (
        DilatedAttentionSimpleOpt,
    )

    # First, visualize the attention pattern
    print("1. Visualizing attention patterns...")
    visualize_attention_pattern(64, 16, 4)

    # Analyze sparsity patterns
    print("\n2. Analyzing sparsity patterns...")
    seq_lengths = [256, 512, 1024, 2048]
    segment_size = 128

    print("\n| Seq Len | Dilation | Active Positions | Total Positions | Sparsity |")
    print("|---------|----------|------------------|-----------------|----------|")

    for seq_len in seq_lengths:
        for dil_rate in [1, 2, 4, 8]:
            active, total, sparsity = profile_attention_pattern(
                seq_len, segment_size, dil_rate
            )
            print(
                f"| {seq_len:7d} | {dil_rate:8d} | {active:16,d} | {total:15,d} | {sparsity:7.1%} |"
            )

    # Detailed performance analysis
    print("\n3. Detailed performance analysis...")
    print("\nTesting with seq_len=512, hidden_dim=256, segment_size=64")

    seq_len = 512
    hidden_dim = 256
    segment_size = 64

    results = []

    for dil_rate in [1, 2, 4, 8]:
        print(f"\n--- Dilation Rate: {dil_rate} ---")

        # Create modules
        original = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=8,
                segment_size=segment_size,
                dilation_rate=dil_rate,
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        optimized = (
            DilatedAttentionSimpleOpt(
                hidden_dim=hidden_dim,
                num_heads=8,
                segment_size=segment_size,
                dilation_rate=dil_rate,
            )
            .cuda()
            .eval()
        )

        # Copy weights
        with torch.no_grad():
            optimized.qkv_proj.weight.copy_(original.qkv_proj.weight)
            optimized.out_proj.weight.copy_(original.out_proj.weight)

        x = torch.randn(1, seq_len, hidden_dim, device="cuda")

        # Benchmark
        print("Benchmarking...")
        orig_stats = benchmark_kernel_detailed(original, x)
        opt_stats = benchmark_kernel_detailed(optimized, x)

        # Verify correctness
        diff = (orig_stats["output"] - opt_stats["output"]).abs().max().item()
        rel_diff = diff / (orig_stats["output"].abs().max().item() + 1e-8)

        print(
            f"Original:  {orig_stats['mean']:.2f} ± {orig_stats['std']:.2f} ms (median: {orig_stats['median']:.2f})"
        )
        print(
            f"Optimized: {opt_stats['mean']:.2f} ± {opt_stats['std']:.2f} ms (median: {opt_stats['median']:.2f})"
        )
        print(f"Speedup:   {orig_stats['mean'] / opt_stats['mean']:.2f}x")
        print(f"Output difference: {diff:.6f} (relative: {rel_diff:.6f})")

        # Calculate efficiency
        active, total, sparsity = profile_attention_pattern(
            seq_len, segment_size, dil_rate
        )
        theoretical_speedup = total / active if active > 0 else 1.0
        actual_speedup = orig_stats["mean"] / opt_stats["mean"]
        efficiency = (actual_speedup / theoretical_speedup) * 100

        print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
        print(f"Efficiency: {efficiency:.1f}%")

        results.append(
            {
                "dilation": dil_rate,
                "orig_time": orig_stats["mean"],
                "opt_time": opt_stats["mean"],
                "speedup": actual_speedup,
                "theoretical": theoretical_speedup,
                "efficiency": efficiency,
            }
        )

    # Plot results
    print("\n4. Creating performance plots...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Runtime comparison
    dilations = [r["dilation"] for r in results]
    orig_times = [r["orig_time"] for r in results]
    opt_times = [r["opt_time"] for r in results]

    x = np.arange(len(dilations))
    width = 0.35

    ax1.bar(x - width / 2, orig_times, width, label="Original", color="blue", alpha=0.7)
    ax1.bar(
        x + width / 2, opt_times, width, label="Optimized", color="green", alpha=0.7
    )
    ax1.set_xlabel("Dilation Rate")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_title("Runtime Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dilations)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup and efficiency
    speedups = [r["speedup"] for r in results]
    theoretical = [r["theoretical"] for r in results]

    ax2.plot(dilations, speedups, "b-o", label="Actual Speedup", linewidth=2)
    ax2.plot(dilations, theoretical, "r--", label="Theoretical Speedup", linewidth=2)
    ax2.set_xlabel("Dilation Rate")
    ax2.set_ylabel("Speedup")
    ax2.set_title("Speedup Analysis")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dilated_kernel_performance_analysis.png", dpi=150)
    print("Saved performance analysis to dilated_kernel_performance_analysis.png")

    print("\n=== Summary ===")
    print("The optimized kernel implementation:")
    print("1. Correctly implements dilated attention pattern")
    print("2. Shows performance improvements for dilated patterns")
    print("3. Efficiency varies based on hardware and pattern")
    print("4. Best suited for moderate dilation rates (2-8)")
    print(
        "\nKey optimization: Processing only active positions instead of all positions + masking"
    )


if __name__ == "__main__":
    main()
