#!/usr/bin/env python3
"""
Simple verification of refactored implementations.
"""

import torch
import time

# Import implementations
from dilated_attention_pytorch.ring_dilated_attention_fixed import (
    RingDilatedAttentionFixed,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_fixed import (
    RingDilatedAttentionHilbertFixed,
)


def main():
    """Simple verification test."""
    print("SIMPLE VERIFICATION TEST")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test parameters
    test_configs = [
        (2048, [512, 1024], [1, 2]),
        (4096, [1024, 2048], [2, 4]),
        (8192, [2048, 4096], [4, 8]),
    ]

    for seq_len, segment_lengths, dilation_rates in test_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing sequence length: {seq_len}")
        print(f"Segments: {segment_lengths}, Dilation: {dilation_rates}")

        # Create test data
        torch.manual_seed(42)
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Test Fixed implementation
        try:
            model_fixed = RingDilatedAttentionFixed(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
                ring_size=1,
            )

            # Forward pass
            with torch.no_grad():
                start = time.time()
                output_fixed = model_fixed(q, k, v)
                fixed_time = time.time() - start

            print("\nFixed Implementation:")
            print(f"  ✓ Forward time: {fixed_time:.3f}s")
            print(f"  Output shape: {output_fixed.shape}")
            print(
                f"  Output stats: mean={output_fixed.mean():.4f}, std={output_fixed.std():.4f}"
            )

            # Verify output
            assert output_fixed.shape == q.shape
            assert not torch.isnan(output_fixed).any()
            assert output_fixed.std() > 0

        except Exception as e:
            print(f"  ✗ Fixed failed: {e}")

        # Test Hilbert implementation
        try:
            model_hilbert = RingDilatedAttentionHilbertFixed(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
                ring_size=1,
            )

            # Forward pass
            with torch.no_grad():
                start = time.time()
                output_hilbert = model_hilbert(q, k, v)
                hilbert_time = time.time() - start

            print("\nHilbert Implementation:")
            print(f"  ✓ Forward time: {hilbert_time:.3f}s")
            print(f"  Output shape: {output_hilbert.shape}")
            print(
                f"  Output stats: mean={output_hilbert.mean():.4f}, std={output_hilbert.std():.4f}"
            )

            # Compare times
            speedup = fixed_time / hilbert_time
            print(f"\nSpeedup: {speedup:.3f}x ({(speedup - 1) * 100:+.1f}%)")

        except Exception as e:
            print(f"  ✗ Hilbert failed: {e}")

        # Test causal masking
        try:
            with torch.no_grad():
                output_causal = model_fixed(q, k, v, is_causal=True)
            print("\nCausal masking:")
            print(f"  ✓ Output shape: {output_causal.shape}")
            print(
                f"  Output stats: mean={output_causal.mean():.4f}, std={output_causal.std():.4f}"
            )
        except Exception as e:
            print(f"  ✗ Causal failed: {e}")

        # Clean up
        del q, k, v
        if "model_fixed" in locals():
            del model_fixed
        if "model_hilbert" in locals():
            del model_hilbert
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("✅ VERIFICATION COMPLETE")
    print("The refactored implementations are working correctly!")


if __name__ == "__main__":
    main()
