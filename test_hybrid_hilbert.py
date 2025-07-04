#!/usr/bin/env python3
"""
Quick test of Hybrid Hilbert Ring Attention.
"""

import torch
import time


# Test basic functionality
def test_basic():
    """Test that Hilbert hybrid works."""
    try:
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
            RingDilatedAttentionHybridOptimizedV2,
        )
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        print("Imports successful!")

        # Test parameters
        batch_size = 1
        seq_len = 8192
        num_heads = 8
        hidden_dim = 512
        head_dim = hidden_dim // num_heads

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"\nTesting on {device} with dtype {dtype}")
        print(f"Sequence length: {seq_len}")

        # Create models
        model_standard = RingDilatedAttentionHybridOptimizedV2(
            segment_lengths=[2048],
            dilation_rates=[1],
            dropout=0.0,
            ring_size=1,  # Single GPU test
            device=device,
            dtype=dtype,
        )

        model_hilbert = RingDilatedAttentionHybridHilbert(
            segment_lengths=[2048],
            dilation_rates=[1],
            dropout=0.0,
            ring_size=1,  # Single GPU test
            device=device,
            dtype=dtype,
            use_hilbert=True,
        )

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        print("\nRunning standard model...")
        with torch.no_grad():
            start = time.time()
            out_standard = model_standard(q, k, v)
            standard_time = (time.time() - start) * 1000

        print(f"Standard time: {standard_time:.1f} ms")
        print(f"Output shape: {out_standard.shape}")

        print("\nRunning Hilbert model...")
        with torch.no_grad():
            start = time.time()
            out_hilbert = model_hilbert(q, k, v)
            hilbert_time = (time.time() - start) * 1000

        print(f"Hilbert time: {hilbert_time:.1f} ms")
        print(f"Output shape: {out_hilbert.shape}")

        print(f"\nSpeedup: {standard_time / hilbert_time:.2f}x")

        # Check outputs are reasonable
        print("\nOutput stats:")
        print(
            f"Standard - Mean: {out_standard.mean().item():.4f}, Std: {out_standard.std().item():.4f}"
        )
        print(
            f"Hilbert  - Mean: {out_hilbert.mean().item():.4f}, Std: {out_hilbert.std().item():.4f}"
        )

        print("\n✓ Test passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_basic()
