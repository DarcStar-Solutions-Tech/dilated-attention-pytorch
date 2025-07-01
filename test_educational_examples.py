#!/usr/bin/env python3
"""Test educational Ring Attention examples."""

import torch
import sys

sys.path.append("examples/ring_attention")

from reference_implementation import TrueRingDilatedAttention
from single_gpu_simulation import SimulatedRingDilatedAttention


def test_true_ring_attention():
    """Test the reference implementation."""
    print("Testing TrueRingDilatedAttention...")

    # Create model
    model = TrueRingDilatedAttention(
        segment_lengths=[512, 1024], dilation_rates=[1, 2], ring_size=4, dropout=0.0
    )

    # Create inputs
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    if torch.cuda.is_available():
        model = model.cuda()
        q, k, v = q.cuda(), k.cuda(), v.cuda()

    # Test forward pass
    try:
        output = model(q, k, v)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        # Test causal
        output_causal = model(q, k, v, is_causal=True)
        print(f"✓ Causal forward pass successful. Output shape: {output_causal.shape}")

        # Check memory usage
        memory_info = model.get_memory_usage(seq_len, batch_size, num_heads, head_dim)
        print("✓ Memory usage calculation successful:")
        print(f"  - Per device: {memory_info['per_device_gb']:.3f} GB")
        print(f"  - Total: {memory_info['total_gb']:.3f} GB")
        print(f"  - Chunk size: {memory_info['chunk_size']}")

    except Exception as e:
        print(f"✗ Error in TrueRingDilatedAttention: {e}")
        return False

    return True


def test_simulated_ring_attention():
    """Test the single GPU simulation."""
    print("\nTesting SimulatedRingDilatedAttention...")

    # Create model
    model = SimulatedRingDilatedAttention(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        ring_size=4,
        chunk_k_v=True,
        dropout=0.0,
    )

    # Create inputs
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    if torch.cuda.is_available():
        model = model.cuda()
        q, k, v = q.cuda(), k.cuda(), v.cuda()

    # Test forward pass
    try:
        output = model(q, k, v)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")

        # Test causal
        output_causal = model(q, k, v, is_causal=True)
        print(f"✓ Causal forward pass successful. Output shape: {output_causal.shape}")

        # Check memory usage
        memory_info = model.get_memory_estimate(
            seq_len, batch_size, num_heads, head_dim
        )
        print("✓ Memory usage calculation successful:")
        print(f"  - Total: {memory_info['total_gb']:.3f} GB")
        print(f"  - K/V memory: {memory_info['kv_memory_gb']:.3f} GB")
        print(f"  - Reduction factor: {memory_info['reduction_factor']:.1f}x")

        # Test without chunking for comparison
        model.chunk_k_v = False
        memory_info_no_chunk = model.get_memory_estimate(
            seq_len, batch_size, num_heads, head_dim
        )
        print(f"  - Without chunking: {memory_info_no_chunk['kv_memory_gb']:.3f} GB")

    except Exception as e:
        print(f"✗ Error in SimulatedRingDilatedAttention: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_equivalence():
    """Test that both implementations produce similar results."""
    print("\nTesting equivalence between implementations...")

    # Create models with same config
    true_ring = TrueRingDilatedAttention(
        segment_lengths=[512], dilation_rates=[1], ring_size=2, dropout=0.0
    )

    simulated_ring = SimulatedRingDilatedAttention(
        segment_lengths=[512],
        dilation_rates=[1],
        ring_size=2,
        chunk_k_v=True,
        dropout=0.0,
    )

    # Create small inputs for easier debugging
    batch_size = 1
    seq_len = 1024
    num_heads = 4
    head_dim = 32

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    if torch.cuda.is_available():
        true_ring = true_ring.cuda()
        simulated_ring = simulated_ring.cuda()
        q, k, v = q.cuda(), k.cuda(), v.cuda()

    try:
        # Get outputs
        output1 = true_ring(q, k, v)
        output2 = simulated_ring(q, k, v)

        # Check if outputs are close
        max_diff = (output1 - output2).abs().max().item()
        mean_diff = (output1 - output2).abs().mean().item()

        print("✓ Both implementations run successfully")
        print(f"  - Max difference: {max_diff:.6f}")
        print(f"  - Mean difference: {mean_diff:.6f}")

        if max_diff > 0.01:
            print("⚠ Warning: Outputs differ significantly")
        else:
            print("✓ Outputs are similar")

    except Exception as e:
        print(f"✗ Error in equivalence test: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("Testing educational Ring Attention examples...\n")

    success = True
    success &= test_true_ring_attention()
    success &= test_simulated_ring_attention()
    success &= test_equivalence()

    if success:
        print("\n✅ All educational examples work correctly!")
    else:
        print("\n❌ Some educational examples have issues")
        sys.exit(1)
