#!/usr/bin/env python3
"""
Test optimized attention fallback chain in hybrid implementation.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)
from dilated_attention_pytorch.ring_attention_lse_optimized import (
    get_attention_backend_info,
)


def test_backend_detection():
    """Test which attention backends are available."""
    print("Checking available attention backends...")

    info = get_attention_backend_info()
    print("\nBackends available:")
    print(f"  Flash Attention: {info['flash_attn']}")
    print(f"  PyTorch SDPA: {info['sdpa']}")
    print(f"  xFormers: {info['xformers']}")
    print(f"  Total backends: {info['backends_available']}")

    # Also check what hybrid sees
    print("\nChecking hybrid implementation setup:")

    # Import the flag
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        HAS_OPTIMIZED_LSE,
    )

    print(f"  HAS_OPTIMIZED_LSE: {HAS_OPTIMIZED_LSE}")

    # Test with small example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RingDilatedAttentionHybrid(
        segment_lengths=[32],
        dilation_rates=[1],
        dropout=0.0,
        device=device,
        dtype=torch.float32,
    )

    # Small input
    batch, seq_len, heads, dim = 1, 64, 4, 32
    q = torch.randn(batch, seq_len, heads, dim, device=device)
    k = torch.randn(batch, seq_len, heads, dim, device=device)
    v = torch.randn(batch, seq_len, heads, dim, device=device)

    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)
        print(f"  Success! Output shape: {output.shape}")

        # Try to see which backend was used by checking if optimized version exists
        if HAS_OPTIMIZED_LSE:
            print("  Using optimized attention with backend fallbacks")
        else:
            print("  Using standard einsum attention")

    except Exception as e:
        print(f"  Error: {str(e)}")


if __name__ == "__main__":
    test_backend_detection()
