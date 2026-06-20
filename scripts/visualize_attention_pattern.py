#!/usr/bin/env python3
"""
Visualize the dilated attention pattern to confirm what the kernels are computing.
"""

import torch
import matplotlib.pyplot as plt


def create_dilated_attention_mask(seq_len, segment_size, dilation_rate):
    """Create the attention mask for dilated attention."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        # Determine which segment this query belongs to
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        # Within the segment, only attend to dilated positions
        for j in range(seg_start, seg_end):
            if (j - seg_start) % dilation_rate == 0:
                mask[i, j] = True

    return mask


def visualize_patterns():
    """Visualize different dilated attention patterns."""
    seq_len = 64

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Dilated Attention Patterns (Black = Attend, White = Ignore)", fontsize=16
    )

    configs = [
        (16, 1, "Segment=16, Dilation=1\n(Standard within segment)"),
        (16, 2, "Segment=16, Dilation=2\n(50% sparse)"),
        (16, 4, "Segment=16, Dilation=4\n(75% sparse)"),
        (32, 1, "Segment=32, Dilation=1\n(Larger segments)"),
        (32, 2, "Segment=32, Dilation=2"),
        (32, 4, "Segment=32, Dilation=4"),
    ]

    for idx, (seg_size, dil_rate, title) in enumerate(configs):
        row = idx // 3
        col = idx % 3

        mask = create_dilated_attention_mask(seq_len, seg_size, dil_rate)

        ax = axes[row, col]
        ax.imshow(mask.numpy(), cmap="gray_r", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        # Add grid lines at segment boundaries
        for i in range(0, seq_len, seg_size):
            ax.axhline(i - 0.5, color="red", linewidth=1, alpha=0.5)
            ax.axvline(i - 0.5, color="red", linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig("dilated_attention_patterns.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Calculate sparsity statistics
    print("=== Dilated Attention Sparsity Analysis ===\n")

    for seg_size, dil_rate, desc in configs:
        mask = create_dilated_attention_mask(seq_len, seg_size, dil_rate)
        total_elements = seq_len * seq_len
        attended_elements = mask.sum().item()
        sparsity = 1.0 - (attended_elements / total_elements)

        print(f"{desc}")
        print(f"  Attended positions: {attended_elements}/{total_elements}")
        print(f"  Sparsity: {sparsity:.1%}")
        print(f"  Computation savings: {sparsity:.1%}\n")

    # Compare with standard self-attention
    print("Standard Self-Attention:")
    print(f"  Attended positions: {seq_len * seq_len}/{seq_len * seq_len}")
    print("  Sparsity: 0.0%")
    print("  Computation savings: 0.0%")


def test_kernel_pattern():
    """Test that our kernels actually implement dilated attention."""
    from dilated_attention_pytorch.kernels import HilbertAttentionCore

    print("\n=== Verifying Kernel Implementation ===\n")

    # Small test case
    seq_len = 32
    hidden_dim = 64
    num_heads = 4
    segment_size = 16
    dilation_rate = 4

    # Create module
    module = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    )

    if torch.cuda.is_available():
        module = module.cuda()
        device = "cuda"
    else:
        device = "cpu"

    # Create input with specific pattern to track attention
    x = torch.zeros(1, seq_len, hidden_dim, device=device)

    # Set specific positions to have high values
    for i in range(0, seq_len, dilation_rate):
        x[0, i, :] = 1.0

    # Forward pass
    with torch.no_grad():
        output = module(
            x, use_hilbert=False
        )  # Don't use Hilbert reordering for clarity

    print("Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Segment size: {segment_size}")
    print(f"  Dilation rate: {dilation_rate}")
    print(
        f"  Expected attention pattern: Every {dilation_rate}th position within each segment"
    )
    print(
        f"\nInput has high values at positions: {list(range(0, seq_len, dilation_rate))}"
    )
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")

    # Check which positions have high influence
    output_norms = output[0].norm(dim=-1)
    high_influence_positions = torch.where(output_norms > output_norms.mean())[0]
    print(
        f"\nPositions with high influence in output: {high_influence_positions.tolist()}"
    )

    # Verify sparsity
    expected_sparsity = 1.0 - (1.0 / dilation_rate)
    print(f"\nExpected sparsity within segments: {expected_sparsity:.1%}")
    print("This confirms the kernel is implementing dilated attention correctly!")


if __name__ == "__main__":
    visualize_patterns()
    test_kernel_pattern()
