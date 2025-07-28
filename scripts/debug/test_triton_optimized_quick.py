#!/usr/bin/env python3
"""Quick test of the Triton optimized implementation."""

import torch
import time


def test_triton_optimized():
    """Quick test of Triton optimized implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("=" * 60)
    print("TESTING TRITON OPTIMIZED IMPLEMENTATION")
    print("=" * 60)

    # Import implementations
    try:
        from dilated_attention_pytorch.ring_dilated_attention_triton_optimized import (
            RingDilatedAttentionTritonOptimized,
        )

        print("✓ Successfully imported RingDilatedAttentionTritonOptimized")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return

    # Test parameters
    seq_len = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print("\nTest configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")

    # Create model
    model = RingDilatedAttentionTritonOptimized(
        segment_lengths=[2048],
        dilation_rates=[4],
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=dtype,
        use_triton_hilbert=True,
        apply_hilbert_to_dilated=True,
    ).eval()

    print("\n✓ Model created successfully")

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print("✓ Inputs created")

    # Test forward pass
    try:
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)
        print(f"✓ Forward pass successful, output shape: {output.shape}")

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = seq_len / elapsed
        print("\nPerformance:")
        print(f"  Time: {elapsed * 1000:.2f} ms")
        print(f"  Throughput: {throughput:,.0f} tokens/sec")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()

    # Test different configurations
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT CONFIGURATIONS")
    print("=" * 60)

    configs = [
        ([2048], [1], "No dilation"),
        ([2048], [4], "Dilation=4"),
        ([1024, 2048], [2, 4], "Multi-segment"),
    ]

    for segments, dilations, desc in configs:
        print(f"\n{desc} - segments={segments}, dilation={dilations}:")

        try:
            model = RingDilatedAttentionTritonOptimized(
                segment_lengths=segments,
                dilation_rates=dilations,
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=dtype,
                use_triton_hilbert=True,
                apply_hilbert_to_dilated=True,
            ).eval()

            with torch.no_grad():
                output = model(q, k, v, is_causal=False)

            print(f"  ✓ Success - output shape: {output.shape}")

        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}...")


if __name__ == "__main__":
    test_triton_optimized()
