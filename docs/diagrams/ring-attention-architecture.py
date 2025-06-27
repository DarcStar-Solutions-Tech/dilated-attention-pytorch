#!/usr/bin/env python3
"""
Create visual diagrams explaining correct Ring Attention architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_ring_attention_diagram():
    """Create a visual explanation of Ring Attention."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Current (Wrong) Implementation
    ax1.set_title("Current Implementation (WRONG)", fontsize=16, color="red")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    # GPU boxes
    gpu0_box = patches.Rectangle(
        (0.5, 6), 4, 3, linewidth=2, edgecolor="black", facecolor="lightcoral"
    )
    gpu1_box = patches.Rectangle(
        (5.5, 6), 4, 3, linewidth=2, edgecolor="black", facecolor="lightcoral"
    )
    ax1.add_patch(gpu0_box)
    ax1.add_patch(gpu1_box)

    ax1.text(2.5, 7.5, "GPU 0", ha="center", va="center", fontsize=12, weight="bold")
    ax1.text(7.5, 7.5, "GPU 1", ha="center", va="center", fontsize=12, weight="bold")

    # Show divided Q, K, V
    ax1.text(2.5, 6.8, "Q[0:N/2]", ha="center", fontsize=10)
    ax1.text(2.5, 6.4, "K[0:N/2]", ha="center", fontsize=10)
    ax1.text(2.5, 6.0, "V[0:N/2]", ha="center", fontsize=10)

    ax1.text(7.5, 6.8, "Q[N/2:N]", ha="center", fontsize=10)
    ax1.text(7.5, 6.4, "K[N/2:N]", ha="center", fontsize=10)
    ax1.text(7.5, 6.0, "V[N/2:N]", ha="center", fontsize=10)

    # Show problem
    ax1.text(
        5,
        5,
        "❌ Each GPU only sees half the queries!",
        ha="center",
        fontsize=12,
        color="red",
    )
    ax1.text(5, 4.5, "❌ No rotation happening", ha="center", fontsize=12, color="red")
    ax1.text(
        5,
        4,
        "❌ Memory: O(N) per GPU (no benefit!)",
        ha="center",
        fontsize=12,
        color="red",
    )

    # Right: Correct Ring Attention
    ax2.set_title("Correct Ring Attention", fontsize=16, color="green")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # GPU boxes
    gpu0_box = patches.Rectangle(
        (0.5, 6), 4, 3, linewidth=2, edgecolor="black", facecolor="lightgreen"
    )
    gpu1_box = patches.Rectangle(
        (5.5, 6), 4, 3, linewidth=2, edgecolor="black", facecolor="lightgreen"
    )
    ax2.add_patch(gpu0_box)
    ax2.add_patch(gpu1_box)

    ax2.text(2.5, 7.5, "GPU 0", ha="center", va="center", fontsize=12, weight="bold")
    ax2.text(7.5, 7.5, "GPU 1", ha="center", va="center", fontsize=12, weight="bold")

    # Show full Q, chunked K, V
    ax2.text(
        2.5,
        6.8,
        "Q[0:N] (full)",
        ha="center",
        fontsize=10,
        weight="bold",
        color="darkgreen",
    )
    ax2.text(2.5, 6.4, "K[0:N/2]", ha="center", fontsize=10)
    ax2.text(2.5, 6.0, "V[0:N/2]", ha="center", fontsize=10)

    ax2.text(
        7.5,
        6.8,
        "Q[0:N] (full)",
        ha="center",
        fontsize=10,
        weight="bold",
        color="darkgreen",
    )
    ax2.text(7.5, 6.4, "K[N/2:N]", ha="center", fontsize=10)
    ax2.text(7.5, 6.0, "V[N/2:N]", ha="center", fontsize=10)

    # Show rotation arrow
    arrow = patches.FancyArrowPatch(
        (4.5, 6.2),
        (5.5, 6.2),
        connectionstyle="arc3,rad=-.3",
        arrowstyle="<->",
        mutation_scale=20,
        linewidth=2,
        color="blue",
    )
    ax2.add_patch(arrow)
    ax2.text(5, 5.5, "K,V rotate", ha="center", fontsize=10, color="blue")

    # Show benefits
    ax2.text(
        5, 4.5, "✓ Each GPU has ALL queries", ha="center", fontsize=12, color="green"
    )
    ax2.text(5, 4, "✓ K,V rotate through ring", ha="center", fontsize=12, color="green")
    ax2.text(
        5,
        3.5,
        "✓ Memory: O(N/ring_size) for K,V!",
        ha="center",
        fontsize=12,
        color="green",
    )

    # Add main title
    fig.suptitle(
        "Ring Attention: Wrong vs Correct Implementation", fontsize=18, weight="bold"
    )

    plt.tight_layout()
    plt.savefig(
        "docs/diagrams/ring-attention-architecture.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create memory scaling diagram
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data for memory scaling
    seq_lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536])
    ring_sizes = [1, 2, 4, 8, 16]

    for ring_size in ring_sizes:
        # Theoretical memory usage: O(N) for Q + O(N/ring_size) for K,V
        memory = seq_lengths * (1 + 2.0 / ring_size) * 8 * 64 * 2 / (1024**3)  # GB
        ax.plot(seq_lengths, memory, "o-", label=f"Ring size {ring_size}", markersize=8)

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Memory Usage (GB)", fontsize=12)
    ax.set_title(
        "Theoretical Memory Scaling with Ring Attention", fontsize=14, weight="bold"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add annotations
    ax.annotate(
        "Standard Attention\nO(N²) memory",
        xy=(16384, 8),
        xytext=(8192, 20),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    ax.annotate(
        "Ring Attention\nO(N/ring_size)",
        xy=(32768, 0.5),
        xytext=(16384, 0.2),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10,
        color="green",
    )

    plt.tight_layout()
    plt.savefig(
        "docs/diagrams/ring-attention-memory-scaling.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("Diagrams created:")
    print("- docs/diagrams/ring-attention-architecture.png")
    print("- docs/diagrams/ring-attention-memory-scaling.png")


def create_ring_rotation_animation_frames():
    """Create frames showing how K,V rotate through the ring."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    steps = [
        "Step 1: Initial State",
        "Step 2: After First Rotation",
        "Step 3: After Second Rotation",
        "Step 4: After Third Rotation (Complete)",
    ]

    kv_positions = [
        [(0, 1), (1, 2), (2, 3), (3, 0)],  # Step 1
        [(3, 0), (0, 1), (1, 2), (2, 3)],  # Step 2
        [(2, 3), (3, 0), (0, 1), (1, 2)],  # Step 3
        [(1, 2), (2, 3), (3, 0), (0, 1)],  # Step 4
    ]

    for step_idx, (ax, title, positions) in enumerate(zip(axes, steps, kv_positions)):
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.axis("off")

        # Draw GPUs in a ring
        gpu_positions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Right, Top, Left, Bottom
        gpu_labels = ["GPU 0", "GPU 1", "GPU 2", "GPU 3"]

        for i, (x, y) in enumerate(gpu_positions):
            # GPU box
            rect = patches.Rectangle(
                (x - 0.4, y - 0.3),
                0.8,
                0.6,
                linewidth=2,
                edgecolor="black",
                facecolor="lightblue",
            )
            ax.add_patch(rect)
            ax.text(x, y + 0.15, gpu_labels[i], ha="center", fontsize=10, weight="bold")

            # Show Q (always full)
            ax.text(x, y, "Q[full]", ha="center", fontsize=8, color="darkgreen")

            # Show K,V chunks
            chunk_from, chunk_to = positions[i]
            ax.text(x, y - 0.15, f"K,V[{chunk_from}]", ha="center", fontsize=8)

        # Draw rotation arrows for steps 2-4
        if step_idx > 0:
            # Circular arrows
            for i in range(4):
                next_i = (i + 1) % 4
                x1, y1 = gpu_positions[i]
                x2, y2 = gpu_positions[next_i]

                # Calculate arrow position
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx / length * 0.5, dy / length * 0.5

                arrow = patches.FancyArrowPatch(
                    (x1 + dx, y1 + dy),
                    (x2 - dx, y2 - dy),
                    arrowstyle="->",
                    mutation_scale=15,
                    linewidth=2,
                    color="red",
                )
                ax.add_patch(arrow)

        # Add center text
        ax.text(
            0,
            0,
            f"Iteration {step_idx + 1}",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"),
        )

    fig.suptitle(
        "Ring Attention: K,V Rotation Through 4 GPUs", fontsize=16, weight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        "docs/diagrams/ring-attention-rotation.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print("- docs/diagrams/ring-attention-rotation.png")


if __name__ == "__main__":
    # Create diagrams
    create_ring_attention_diagram()
    create_ring_rotation_animation_frames()

    print("\nDiagrams created successfully!")
    print("\nThese diagrams show:")
    print("1. The fundamental difference between wrong and correct implementation")
    print("2. Theoretical memory scaling with different ring sizes")
    print("3. How K,V chunks rotate through the ring while Q stays fixed")
