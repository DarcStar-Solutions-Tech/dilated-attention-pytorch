#!/usr/bin/env python3
"""
Plot kernel performance comparison.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_performance_plots():
    """Create performance comparison plots."""

    # Data from benchmarks
    configs = ["Small\n(128)", "Medium\n(512)", "Large\n(1024)"]

    # Forward pass times (ms)
    core_forward = [0.31, 1.18, 21.65]
    simple_forward = [0.98, 3.34, 10.01]

    # Backward pass times (ms)
    core_backward = [0.85, 1.39, 43.40]
    simple_backward = [3.13, 14.91, 87.84]

    # Memory usage (MB)
    core_memory = [19.8, 43.0, 101.3]
    simple_memory = [23.1, 55.3, 123.8]

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Kernel Performance Comparison: HilbertAttentionCore vs HilbertAttentionSimple",
        fontsize=16,
    )

    # Forward pass comparison
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2, core_forward, width, label="Core (Triton)", color="#2E86AB"
    )
    bars2 = ax1.bar(
        x + width / 2, simple_forward, width, label="Simple (PyTorch)", color="#F24236"
    )

    ax1.set_ylabel("Time (ms)", fontsize=12)
    ax1.set_title("Forward Pass Performance", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Backward pass comparison
    bars3 = ax2.bar(
        x - width / 2, core_backward, width, label="Core (Triton)", color="#2E86AB"
    )
    bars4 = ax2.bar(
        x + width / 2, simple_backward, width, label="Simple (PyTorch)", color="#F24236"
    )

    ax2.set_ylabel("Time (ms)", fontsize=12)
    ax2.set_title("Backward Pass Performance", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Memory usage comparison
    bars5 = ax3.bar(
        x - width / 2, core_memory, width, label="Core (Triton)", color="#2E86AB"
    )
    bars6 = ax3.bar(
        x + width / 2, simple_memory, width, label="Simple (PyTorch)", color="#F24236"
    )

    ax3.set_ylabel("Memory (MB)", fontsize=12)
    ax3.set_title("Memory Usage", fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Speedup chart
    forward_speedup = [s / c for s, c in zip(simple_forward, core_forward)]
    backward_speedup = [s / c for s, c in zip(simple_backward, core_backward)]
    memory_reduction = [(s - c) / s * 100 for s, c in zip(simple_memory, core_memory)]

    x2 = np.arange(len(configs))
    bars7 = ax4.bar(
        x2 - width / 2, forward_speedup, width, label="Forward", color="#A23B72"
    )
    bars8 = ax4.bar(
        x2 + width / 2, backward_speedup, width, label="Backward", color="#C18FCF"
    )

    ax4.set_ylabel("Speedup Factor", fontsize=12)
    ax4.set_title("Triton Kernel Speedup", fontsize=14)
    ax4.set_xticks(x2)
    ax4.set_xticklabels(configs)
    ax4.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(
                f"{height:.1f}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig("kernel_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Create memory reduction chart
    fig2, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(configs, memory_reduction, color="#2E86AB")
    ax.set_ylabel("Memory Reduction (%)", fontsize=12)
    ax.set_title("Memory Efficiency: Triton vs PyTorch", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("kernel_memory_efficiency.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_performance_plots()
    print("Performance plots saved as:")
    print("  - kernel_performance_comparison.png")
    print("  - kernel_memory_efficiency.png")
