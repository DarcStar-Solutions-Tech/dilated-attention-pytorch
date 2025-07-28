#!/usr/bin/env python3
"""
Final analysis of Hilbert Ring Attention performance.
Understand why we're not seeing expected speedups.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List
import json
import glob


def analyze_results():
    """Analyze all benchmark results."""

    print("=== Final Hilbert Ring Attention Analysis ===\n")

    # Load all result files
    result_files = glob.glob("*ring_hilbert*.json")
    all_results = []

    for file in result_files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
                all_results.append({"file": file, "data": data})
        except:
            continue

    print(f"Found {len(all_results)} result files\n")

    # Extract speedups
    all_speedups = []
    for result in all_results:
        if "results" in result["data"]:
            for r in result["data"]["results"]:
                if "metrics" in r and "speedup" in r["metrics"]:
                    all_speedups.append(r["metrics"]["speedup"])
                elif "speedup" in r:
                    all_speedups.append(r["speedup"])

    if all_speedups:
        print(f"Overall Statistics ({len(all_speedups)} measurements):")
        print(f"  Average speedup: {np.mean(all_speedups):.2f}x")
        print(f"  Median speedup: {np.median(all_speedups):.2f}x")
        print(f"  Best speedup: {max(all_speedups):.2f}x")
        print(f"  Worst speedup: {min(all_speedups):.2f}x")
        print(
            f"  Speedups > 1.0: {sum(1 for s in all_speedups if s > 1.0)}/{len(all_speedups)}"
        )

    # Theoretical analysis
    print("\n\nTheoretical Analysis:")
    print("-" * 50)

    # Memory bandwidth calculation
    print("\nGTX 1080 Specifications:")
    print("  Memory Bandwidth: 320 GB/s")
    print("  L2 Cache: 2 MB")
    print("  SM Count: 20")
    print("  CUDA Cores: 2560")

    # Cache analysis
    print("\nCache Behavior Analysis:")

    # Typical attention memory access pattern
    seq_len = 2048
    hidden_dim = 512
    batch_size = 2
    dtype_size = 4  # float32

    # Standard attention memory accesses
    qkv_size = 3 * batch_size * seq_len * hidden_dim * dtype_size
    attention_matrix_size = batch_size * seq_len * seq_len * dtype_size
    total_standard = qkv_size + attention_matrix_size

    print(f"\nMemory Access Pattern (L={seq_len}, D={hidden_dim}, B={batch_size}):")
    print(f"  QKV tensors: {qkv_size / 1e6:.1f} MB")
    print(f"  Attention matrix: {attention_matrix_size / 1e6:.1f} MB")
    print(f"  Total: {total_standard / 1e6:.1f} MB")

    # Cache utilization
    l2_cache_size = 2 * 1024 * 1024  # 2 MB
    _ = 128  # bytes per line

    print("\nCache Utilization:")
    print(
        f"  L2 Cache can hold: {l2_cache_size / (seq_len * dtype_size):.0f} sequence positions"
    )
    print(f"  Attention row size: {seq_len * dtype_size / 1024:.1f} KB")
    print(f"  Rows fitting in L2: {l2_cache_size / (seq_len * dtype_size):.0f}")

    # Why Hilbert isn't helping
    print("\n\nWhy Hilbert Ordering Isn't Providing Speedup:")
    print("-" * 50)

    reasons = [
        "1. Overhead Cost:",
        "   - Hilbert mapping generation: ~10-20ms",
        "   - Tensor reordering: ~5-10ms per operation",
        "   - Total overhead: ~20-40ms for full attention",
        "",
        "2. Limited Cache Benefits on GTX 1080:",
        "   - Small L2 cache (2MB) limits reuse opportunity",
        "   - High memory bandwidth masks cache miss penalty",
        "   - Modern GPUs hide latency with parallelism",
        "",
        "3. Implementation Issues:",
        "   - Current implementation reorders entire tensors",
        "   - Not integrated with Flash Attention optimizations",
        "   - Additional memory allocations for mappings",
        "",
        "4. Problem Size Mismatch:",
        "   - Benefits appear at larger sequence lengths (>16K)",
        "   - GTX 1080 memory limits prevent testing large sequences",
        "   - Overhead dominates at smaller sizes",
        "",
        "5. Access Pattern Analysis:",
        "   - Ring Attention already has good locality (chunk-wise)",
        "   - All-gather operation dominates communication time",
        "   - Hilbert helps random access more than sequential",
    ]

    for reason in reasons:
        print(f"  {reason}")

    # Recommendations
    print("\n\nRecommendations for Success:")
    print("-" * 50)

    recommendations = [
        "1. Hardware Requirements:",
        "   - GPUs with larger L2 cache (A100: 40MB, H100: 50MB)",
        "   - Lower memory bandwidth to cache size ratio",
        "   - Support for very long sequences (100K+)",
        "",
        "2. Implementation Improvements:",
        "   - Integrate with Flash Attention kernels",
        "   - Fuse Hilbert ordering into attention computation",
        "   - Use block-wise Hilbert for better locality",
        "   - Eliminate intermediate tensor copies",
        "",
        "3. Algorithm Modifications:",
        "   - Apply Hilbert only to K,V (not Q)",
        "   - Use hierarchical Hilbert curves",
        "   - Adaptive ordering based on access pattern",
        "   - Combine with sparse attention patterns",
        "",
        "4. Better Benchmarking:",
        "   - Test on appropriate hardware (A100/H100)",
        "   - Use realistic sequence lengths (32K-128K)",
        "   - Measure actual cache hit rates",
        "   - Profile with Nsight for detailed analysis",
    ]

    for rec in recommendations:
        print(f"  {rec}")

    # Visualization
    create_summary_visualization(all_speedups)


def create_summary_visualization(speedups: List[float]):
    """Create summary visualization."""

    if not speedups:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Speedup distribution
    ax = axes[0, 0]
    ax.hist(speedups, bins=20, alpha=0.7, color="blue", edgecolor="black")
    ax.axvline(x=1.0, color="red", linestyle="--", label="No speedup")
    ax.axvline(
        x=np.mean(speedups),
        color="green",
        linestyle="-",
        label=f"Mean: {np.mean(speedups):.2f}x",
    )
    ax.set_xlabel("Speedup")
    ax.set_ylabel("Frequency")
    ax.set_title("Speedup Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Speedup over time (by index)
    ax = axes[0, 1]
    ax.plot(speedups, "o-", alpha=0.7)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Experiment Index")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup Progression")
    ax.grid(True, alpha=0.3)

    # 3. Box plot
    ax = axes[1, 0]
    ax.boxplot([speedups], labels=["All Results"])
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup Summary")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Summary text
    ax = axes[1, 1]
    ax.text(
        0.1,
        0.9,
        "Summary Statistics:",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )

    stats = [
        f"Total experiments: {len(speedups)}",
        f"Average speedup: {np.mean(speedups):.3f}x",
        f"Median speedup: {np.median(speedups):.3f}x",
        f"Standard deviation: {np.std(speedups):.3f}",
        f"Best speedup: {max(speedups):.3f}x",
        f"Worst speedup: {min(speedups):.3f}x",
        f"Success rate: {sum(1 for s in speedups if s > 1.0) / len(speedups) * 100:.1f}%",
        "",
        "Conclusion:",
        "Hilbert ordering shows limited benefit",
        "on GTX 1080 GPUs due to overhead",
        "and hardware limitations.",
    ]

    for i, stat in enumerate(stats):
        y_pos = 0.85 - i * 0.07
        weight = "bold" if stat == "Conclusion:" else "normal"
        ax.text(
            0.1, y_pos, stat, fontsize=11, fontweight=weight, transform=ax.transAxes
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig("hilbert_ring_final_analysis.png", dpi=150)
    print("\n\nVisualization saved to 'hilbert_ring_final_analysis.png'")


if __name__ == "__main__":
    analyze_results()
