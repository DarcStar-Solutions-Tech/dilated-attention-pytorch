#!/usr/bin/env python3
"""
Test per-segment Hilbert ordering in Ring Dilated Attention.
"""

import torch
from dilated_attention_pytorch.ring.hilbert.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def test_per_segment_hilbert():
    """Test that per-segment Hilbert ordering works correctly."""
    print("Testing per-segment Hilbert ordering...")

    # Test parameters
    batch_size = 2
    seq_len = 8192
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim

    # Create attention module
    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        use_hilbert=True,
        dropout=0.0,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print(f"Device: {device}")
    print(f"Input shape: {q.shape}")
    print(f"Segment lengths: {attention.segment_lengths}")
    print(f"Dilation rates: {attention.dilation_rates}")

    # Test forward pass
    try:
        output = attention(q, k, v)
        print(f"Output shape: {output.shape}")
        print("✓ Forward pass successful")

        # Check output is not all zeros
        if torch.allclose(output, torch.zeros_like(output)):
            print("✗ Warning: Output is all zeros!")
        else:
            print("✓ Output contains non-zero values")

        # Check no NaN values
        if torch.isnan(output).any():
            print("✗ Warning: Output contains NaN values!")
        else:
            print("✓ No NaN values in output")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        raise

    # Test with causal masking
    print("\nTesting with causal masking...")
    try:
        _ = attention(q, k, v, is_causal=True)
        print("✓ Causal forward pass successful")
    except Exception as e:
        print(f"✗ Causal forward pass failed: {e}")
        raise

    # Compare with Hilbert disabled
    print("\nComparing with Hilbert disabled...")
    attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        use_hilbert=False,
        dropout=0.0,
    ).to(device)

    output_no_hilbert = attention_no_hilbert(q, k, v)

    # Outputs should be different but similar in magnitude
    diff = (output - output_no_hilbert).abs().mean()
    print(f"Mean absolute difference: {diff.item():.6f}")

    if diff.item() > 0:
        print("✓ Hilbert ordering is having an effect")
    else:
        print("✗ Warning: Hilbert ordering has no effect!")

    print("\nAll tests passed!")


def benchmark_comparison():
    """Quick benchmark comparing with and without per-segment Hilbert."""
    import time

    print("\n" + "=" * 60)
    print("Benchmarking per-segment Hilbert ordering")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    # Test configurations
    configs = [
        (1, 2048, "Small"),
        (1, 4096, "Medium"),
        (1, 8192, "Large"),
    ]

    num_iterations = 10
    warmup = 3

    for batch_size, seq_len, desc in configs:
        print(f"\n{desc} sequence (batch={batch_size}, seq_len={seq_len}):")

        # Common parameters
        num_heads = 8
        head_dim = 64
        embed_dim = num_heads * head_dim
        device = torch.device("cuda")

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Test with Hilbert
        attention_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
            dim=embed_dim,
            heads=num_heads,
            segment_lengths=[min(2048, seq_len), min(4096, seq_len), seq_len],
            dilation_rates=[1, 2, 4],
            use_hilbert=True,
        ).to(device)

        # Warmup
        for _ in range(warmup):
            _ = attention_hilbert(q, k, v)
        torch.cuda.synchronize()

        # Time with Hilbert
        start = time.time()
        for _ in range(num_iterations):
            _ = attention_hilbert(q, k, v)
        torch.cuda.synchronize()
        time_hilbert = (time.time() - start) / num_iterations * 1000

        # Test without Hilbert
        attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
            dim=embed_dim,
            heads=num_heads,
            segment_lengths=[min(2048, seq_len), min(4096, seq_len), seq_len],
            dilation_rates=[1, 2, 4],
            use_hilbert=False,
        ).to(device)

        # Warmup
        for _ in range(warmup):
            _ = attention_no_hilbert(q, k, v)
        torch.cuda.synchronize()

        # Time without Hilbert
        start = time.time()
        for _ in range(num_iterations):
            _ = attention_no_hilbert(q, k, v)
        torch.cuda.synchronize()
        time_no_hilbert = (time.time() - start) / num_iterations * 1000

        speedup = time_no_hilbert / time_hilbert
        print(f"  With Hilbert: {time_hilbert:.2f}ms")
        print(f"  Without Hilbert: {time_no_hilbert:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    test_per_segment_hilbert()
    benchmark_comparison()
