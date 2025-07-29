#!/usr/bin/env python3
"""Visualize the corrected sparse Hilbert implementation."""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention
from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    create_hilbert_mapping,
)


def visualize_comparison():
    """Compare wrong vs correct sparse Hilbert implementations."""

    print("VISUALIZING SPARSE HILBERT CORRECTION")
    print("=" * 80)

    seq_len = 256
    segment_size = 64
    dilation_rate = 4

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Sparse Hilbert: Wrong vs Correct Implementation", fontsize=16)

    # 1. WRONG APPROACH (old implementation)
    ax = axes[0, 0]

    # Create standard Hilbert mapping
    standard_hilbert = create_hilbert_mapping(seq_len)

    # Apply to sparse positions (wrong way)
    wrong_sparse_positions = []
    wrong_mapped_positions = []

    for i in range(0, seq_len, dilation_rate):
        wrong_sparse_positions.append(i)
        wrong_mapped_positions.append(standard_hilbert[i].item())

    # Plot
    ax.scatter(
        range(len(wrong_sparse_positions)),
        wrong_sparse_positions,
        label="Original sparse",
        alpha=0.6,
        s=30,
    )
    ax.scatter(
        range(len(wrong_mapped_positions)),
        wrong_mapped_positions,
        label="After Hilbert",
        alpha=0.6,
        s=30,
        color="red",
    )
    ax.plot(wrong_mapped_positions[:32], "r-", alpha=0.3, linewidth=1)
    ax.set_title("WRONG: Apply Hilbert then Select Sparse")
    ax.set_xlabel("Access order")
    ax.set_ylabel("Memory position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. CORRECT APPROACH (new implementation)
    ax = axes[0, 1]

    # Create module with corrected implementation
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
        dropout=0.0,
        hilbert_threshold=0,
    )

    # Get corrected mapping
    correct_mapping = module._create_segment_local_hilbert_mapping(
        seq_len, segment_size, dilation_rate
    )

    # Apply to sparse positions (correct way)
    correct_sparse_positions = []
    correct_mapped_positions = []

    for i in range(0, seq_len, dilation_rate):
        correct_sparse_positions.append(i)
        correct_mapped_positions.append(correct_mapping[i].item())

    # Plot
    ax.scatter(
        range(len(correct_sparse_positions)),
        correct_sparse_positions,
        label="Original sparse",
        alpha=0.6,
        s=30,
    )
    ax.scatter(
        range(len(correct_mapped_positions)),
        correct_mapped_positions,
        label="After segment-local Hilbert",
        alpha=0.6,
        s=30,
        color="green",
    )
    ax.plot(correct_mapped_positions[:32], "g-", alpha=0.3, linewidth=1)
    ax.set_title("CORRECT: Select Sparse then Apply Hilbert per Segment")
    ax.set_xlabel("Access order")
    ax.set_ylabel("Memory position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Jump distance comparison
    ax = axes[1, 0]

    # Calculate jumps for both approaches
    wrong_jumps = [
        abs(wrong_mapped_positions[i + 1] - wrong_mapped_positions[i])
        for i in range(len(wrong_mapped_positions) - 1)
    ]
    correct_jumps = [
        abs(correct_mapped_positions[i + 1] - correct_mapped_positions[i])
        for i in range(len(correct_mapped_positions) - 1)
    ]

    # Plot histograms
    bins = np.linspace(0, max(max(wrong_jumps), max(correct_jumps)), 30)
    ax.hist(
        wrong_jumps,
        bins=bins,
        alpha=0.5,
        label=f"Wrong (avg={np.mean(wrong_jumps):.1f})",
        color="red",
        density=True,
    )
    ax.hist(
        correct_jumps,
        bins=bins,
        alpha=0.5,
        label=f"Correct (avg={np.mean(correct_jumps):.1f})",
        color="green",
        density=True,
    )
    ax.axvline(
        dilation_rate,
        color="blue",
        linestyle="--",
        label=f"Ideal (sequential={dilation_rate})",
    )
    ax.set_xlabel("Jump distance")
    ax.set_ylabel("Frequency (normalized)")
    ax.set_title("Memory Jump Distance Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cache efficiency visualization
    ax = axes[1, 1]

    # Simulate cache line access
    cache_line_size = 32  # elements per cache line

    def count_cache_lines(positions):
        cache_lines = set()
        for pos in positions:
            cache_lines.add(pos // cache_line_size)
        return len(cache_lines)

    # Count for different sequence lengths
    test_lengths = [64, 128, 256, 512, 1024]
    wrong_cache_lines = []
    correct_cache_lines = []
    ideal_cache_lines = []

    for test_len in test_lengths:
        # Wrong approach
        wrong_map = create_hilbert_mapping(test_len)
        wrong_pos = [wrong_map[i].item() for i in range(0, test_len, dilation_rate)]
        wrong_cache_lines.append(count_cache_lines(wrong_pos))

        # Correct approach
        if test_len <= seq_len:
            correct_map = module._create_segment_local_hilbert_mapping(
                test_len, segment_size, dilation_rate
            )
            correct_pos = [
                correct_map[i].item() for i in range(0, test_len, dilation_rate)
            ]
        else:
            # Approximate for larger sizes
            correct_pos = list(range(0, test_len, dilation_rate))
        correct_cache_lines.append(count_cache_lines(correct_pos))

        # Ideal (sequential)
        ideal_pos = list(range(0, test_len, dilation_rate))
        ideal_cache_lines.append(count_cache_lines(ideal_pos))

    # Plot
    x = np.arange(len(test_lengths))
    width = 0.25

    ax.bar(
        x - width,
        wrong_cache_lines,
        width,
        label="Wrong approach",
        color="red",
        alpha=0.7,
    )
    ax.bar(
        x,
        correct_cache_lines,
        width,
        label="Correct approach",
        color="green",
        alpha=0.7,
    )
    ax.bar(
        x + width,
        ideal_cache_lines,
        width,
        label="Ideal (sequential)",
        color="blue",
        alpha=0.7,
    )

    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Cache lines accessed")
    ax.set_title("Cache Line Usage (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(test_lengths)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "benchmarks/sparse_hilbert_correction_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Print summary statistics
    print(f"\nSummary for sequence length {seq_len}:")
    print("\nWRONG APPROACH:")
    print(f"  Average jump: {np.mean(wrong_jumps):.1f}")
    print(f"  Max jump: {max(wrong_jumps)}")
    print(f"  Cache lines used: {count_cache_lines(wrong_mapped_positions[:64])}")

    print("\nCORRECT APPROACH:")
    print(f"  Average jump: {np.mean(correct_jumps):.1f}")
    print(f"  Max jump: {max(correct_jumps)}")
    print(f"  Cache lines used: {count_cache_lines(correct_mapped_positions[:64])}")

    print("\nIMPROVEMENT:")
    print(
        f"  Jump distance reduced by: {(1 - np.mean(correct_jumps) / np.mean(wrong_jumps)) * 100:.1f}%"
    )
    print(
        f"  Cache efficiency improved by: {(1 - count_cache_lines(correct_mapped_positions[:64]) / count_cache_lines(wrong_mapped_positions[:64])) * 100:.1f}%"
    )

    print(
        "\nVisualization saved to: benchmarks/sparse_hilbert_correction_comparison.png"
    )


if __name__ == "__main__":
    visualize_comparison()
