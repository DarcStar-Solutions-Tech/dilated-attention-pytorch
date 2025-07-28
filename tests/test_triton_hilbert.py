#!/usr/bin/env python3
"""
Quick test to verify Triton kernel applies Hilbert ordering.
"""

import torch
import sys

sys.path.insert(
    0, "/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch/src"
)

from dilated_attention_pytorch.kernels import HilbertAttention


def test_triton_hilbert():
    """Test that Triton kernel actually applies Hilbert ordering."""

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return

    device = torch.device("cuda")

    # Create attention module
    attn = HilbertAttention(
        hidden_dim=256,
        num_heads=8,
        segment_size=64,
    ).to(device)

    print(f"Triton available: {attn._triton_available}")

    # Create input with specific pattern to detect reordering
    # Make a sequence where position i has value i in all dimensions
    seq_len = 256  # Large enough to trigger Hilbert mapping
    x = torch.zeros(1, seq_len, 256, device=device)
    for i in range(seq_len):
        x[0, i, :] = float(i) / seq_len

    # Forward pass with and without Hilbert
    with torch.no_grad():
        out_no_hilbert = attn(x, use_hilbert=False)
        out_hilbert = attn(x, use_hilbert=True)

    # Check if outputs are different
    diff = (out_hilbert - out_no_hilbert).abs().mean().item()
    print(f"Mean absolute difference: {diff:.6f}")

    if diff > 1e-5:
        print("✓ Triton kernel applies Hilbert ordering correctly")
    else:
        print("⚠ Triton kernel may not be applying Hilbert ordering")

    # Also check specific positions to see the effect
    # With Hilbert ordering, nearby positions in output should have
    # attended to different patterns than without
    pos_to_check = [0, 10, 50, 100, 200]
    print("\nOutput differences at specific positions:")
    for pos in pos_to_check:
        pos_diff = (out_hilbert[0, pos] - out_no_hilbert[0, pos]).abs().mean().item()
        print(f"  Position {pos}: {pos_diff:.6f}")


if __name__ == "__main__":
    test_triton_hilbert()
