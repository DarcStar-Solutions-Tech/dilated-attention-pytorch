#!/usr/bin/env python3
"""
Create a visual diagram showing the difference between V2 and Hybrid implementations.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_dilated_attention_diagram():
    """Create a visual comparison of V2 vs Hybrid dilated attention."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Common parameters
    seq_len = 16
    positions = list(range(seq_len))

    # V2 Collective (Correct)
    ax1.set_title(
        "V2 Collective: Segment → Dilate (CORRECT)", fontsize=16, fontweight="bold"
    )
    ax1.set_xlim(-0.5, seq_len - 0.5)
    ax1.set_ylim(-0.5, 4.5)

    # Draw original sequence
    for i in positions:
        rect = patches.Rectangle(
            (i - 0.4, 3.6),
            0.8,
            0.8,
            linewidth=1,
            edgecolor="black",
            facecolor="lightblue",
        )
        ax1.add_patch(rect)
        ax1.text(i, 4, str(i), ha="center", va="center", fontsize=10)

    ax1.text(
        -1, 4, "Original:", ha="right", va="center", fontsize=12, fontweight="bold"
    )

    # Draw segments
    segment_colors = ["lightgreen", "lightcoral"]
    for seg_idx in range(2):
        start = seg_idx * 8
        end = start + 8

        # Segment boundary
        rect = patches.Rectangle(
            (start - 0.45, 2.35),
            8 - 0.1,
            1.1,
            linewidth=2,
            edgecolor="darkgreen",
            facecolor=segment_colors[seg_idx],
            alpha=0.3,
        )
        ax1.add_patch(rect)

        # Segment positions
        for i in range(start, end):
            rect = patches.Rectangle(
                (i - 0.4, 2.6),
                0.8,
                0.8,
                linewidth=1,
                edgecolor="black",
                facecolor=segment_colors[seg_idx],
            )
            ax1.add_patch(rect)
            ax1.text(i, 3, str(i), ha="center", va="center", fontsize=10)

    ax1.text(
        -1, 3, "Segments:", ha="right", va="center", fontsize=12, fontweight="bold"
    )

    # Draw dilated segments (dilation=2)
    for seg_idx in range(2):
        start = seg_idx * 8
        dilated_positions = list(range(start, start + 8, 2))

        # Show which positions are selected
        for i, pos in enumerate(dilated_positions):
            x = start + i * 2
            rect = patches.Rectangle(
                (x - 0.4, 0.6),
                0.8,
                0.8,
                linewidth=1,
                edgecolor="black",
                facecolor=segment_colors[seg_idx],
            )
            ax1.add_patch(rect)
            ax1.text(x, 1, str(pos), ha="center", va="center", fontsize=10)

            # Draw arrow
            ax1.arrow(
                pos,
                2.5,
                0,
                -1.3,
                head_width=0.2,
                head_length=0.1,
                fc="gray",
                ec="gray",
                alpha=0.5,
            )

    ax1.text(
        -1,
        1,
        "Dilated\n(rate=2):",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Add segment labels
    ax1.text(
        3.5,
        1.5,
        "Segment 0",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5),
    )
    ax1.text(
        11.5,
        1.5,
        "Segment 1",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5),
    )

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # Hybrid (Incorrect)
    ax2.set_title(
        "Hybrid: Dilate → Segment (INCORRECT)",
        fontsize=16,
        fontweight="bold",
        color="red",
    )
    ax2.set_xlim(-0.5, seq_len - 0.5)
    ax2.set_ylim(-0.5, 4.5)

    # Draw original sequence
    for i in positions:
        rect = patches.Rectangle(
            (i - 0.4, 3.6),
            0.8,
            0.8,
            linewidth=1,
            edgecolor="black",
            facecolor="lightblue",
        )
        ax2.add_patch(rect)
        ax2.text(i, 4, str(i), ha="center", va="center", fontsize=10)

    ax2.text(
        -1, 4, "Original:", ha="right", va="center", fontsize=12, fontweight="bold"
    )

    # Draw globally dilated sequence
    dilated_global = list(range(0, seq_len, 2))
    for i, pos in enumerate(dilated_global):
        rect = patches.Rectangle(
            (i * 2 - 0.4, 2.6),
            0.8,
            0.8,
            linewidth=1,
            edgecolor="black",
            facecolor="yellow",
        )
        ax2.add_patch(rect)
        ax2.text(i * 2, 3, str(pos), ha="center", va="center", fontsize=10)

        # Draw arrow
        ax2.arrow(
            pos,
            3.5,
            (i * 2 - pos) * 0.8,
            -0.4,
            head_width=0.2,
            head_length=0.1,
            fc="gray",
            ec="gray",
            alpha=0.5,
        )

    ax2.text(
        -1,
        3,
        "Global\nDilation:",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Draw ring chunks (arbitrary segmentation)
    chunk_colors = ["orange", "purple"]
    for chunk_idx in range(2):
        start = chunk_idx * 4
        end = start + 4

        # Chunk boundary
        rect = patches.Rectangle(
            (start * 2 - 0.45, 0.35),
            8 - 0.1,
            1.1,
            linewidth=2,
            edgecolor="darkred",
            facecolor=chunk_colors[chunk_idx],
            alpha=0.3,
        )
        ax2.add_patch(rect)

        # Chunk positions
        for i in range(start, end):
            if i < len(dilated_global):
                rect = patches.Rectangle(
                    (i * 2 - 0.4, 0.6),
                    0.8,
                    0.8,
                    linewidth=1,
                    edgecolor="black",
                    facecolor=chunk_colors[chunk_idx],
                )
                ax2.add_patch(rect)
                ax2.text(
                    i * 2,
                    1,
                    str(dilated_global[i]),
                    ha="center",
                    va="center",
                    fontsize=10,
                )

    ax2.text(
        -1, 1, "Ring\nChunks:", ha="right", va="center", fontsize=12, fontweight="bold"
    )

    # Add warning labels
    ax2.text(
        3.5,
        1.5,
        "Chunk 0: [0,2,4,6]",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5),
    )
    ax2.text(
        11.5,
        1.5,
        "Chunk 1: [8,10,12,14]",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="purple", alpha=0.5),
    )

    # Add warning text
    ax2.text(
        8,
        -0.2,
        "⚠️ Positions from different segments mixed in chunks!",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
    )

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    plt.tight_layout()
    plt.savefig("dilated_attention_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig("dilated_attention_comparison.pdf", bbox_inches="tight")
    print("Saved diagrams: dilated_attention_comparison.png and .pdf")


if __name__ == "__main__":
    create_dilated_attention_diagram()
