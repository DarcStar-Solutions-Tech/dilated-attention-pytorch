#!/usr/bin/env python3
"""
Simple test for per-segment Hilbert ordering.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def test_simple():
    """Simple test with small sequences."""
    print("Testing per-segment Hilbert ordering with small sequences...")

    # Small test parameters
    batch_size = 1
    seq_len = 512
    num_heads = 4
    head_dim = 32
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module
    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[128, 256, 512],
        dilation_rates=[1, 1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print(f"Device: {device}")
    print(f"Input shape: {q.shape}")

    # Test forward pass
    try:
        output = attention(q, k, v)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Output mean: {output.mean().item():.6f}")
        print(f"✓ Output std: {output.std().item():.6f}")

        # Test Hilbert cache
        print("\nHilbert cache sizes:")
        print(f"  Forward cache: {list(attention._hilbert_cache.keys())}")
        print(f"  Inverse cache: {list(attention._inverse_hilbert_cache.keys())}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_simple()
