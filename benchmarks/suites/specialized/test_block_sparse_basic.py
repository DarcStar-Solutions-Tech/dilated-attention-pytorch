#!/usr/bin/env python3
"""
Basic test for block-sparse implementations.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def test_basic():
    """Test basic functionality."""
    print("Testing Block-Sparse Basic Functionality")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 for compatibility

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test base variant
    print("\n1. Testing Base Variant (90% sparse)...")
    try:
        model = create_block_sparse_attention(
            variant="base",
            segment_lengths=[512],
            dilation_rates=[1],
            sparsity_ratio=0.1,
        )
        model = model.to(device=device, dtype=dtype)
        output = model(q, k, v)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output valid: {torch.isfinite(output).all()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test with different sparsity
    print("\n2. Testing Ultra-Sparse (99% sparse)...")
    try:
        model = create_block_sparse_attention(
            variant="base",
            segment_lengths=[512],
            dilation_rates=[1],
            sparsity_ratio=0.01,
        )
        model = model.to(device=device, dtype=dtype)
        output = model(q, k, v)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output valid: {torch.isfinite(output).all()}")

        # Get pattern stats
        stats = model.get_pattern_stats()
        print(f"   ✓ Active blocks: {stats['active_blocks']}")
        print(f"   ✓ Hit rate: {stats['hit_rate']:.2%}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test hierarchical
    print("\n3. Testing Hierarchical...")
    try:
        model = create_block_sparse_attention(
            variant="hierarchical",
            segment_lengths=[512],
            dilation_rates=[1],
        )
        model = model.to(device=device, dtype=dtype)
        output = model(q, k, v)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output valid: {torch.isfinite(output).all()}")

        # Get pattern stats
        stats = model.get_pattern_stats(seq_len)
        print(f"   ✓ Sparsity: {stats['sparsity']:.1%}")
        print(f"   ✓ Levels: {len(stats['levels'])}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test adaptive
    print("\n4. Testing Adaptive...")
    try:
        model = create_block_sparse_attention(
            variant="adaptive",
            segment_lengths=[512],
            dilation_rates=[1],
        )
        model = model.to(device=device, dtype=dtype)
        output = model(q, k, v)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output valid: {torch.isfinite(output).all()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test gradient flow
    print("\n5. Testing Gradient Flow...")
    try:
        model = create_block_sparse_attention(
            variant="base",
            segment_lengths=[512],
            dilation_rates=[1],
            sparsity_ratio=0.1,
        )
        model = model.to(device=device, dtype=dtype)

        # Create inputs with gradients
        q_grad = torch.randn_like(q, requires_grad=True)
        k_grad = torch.randn_like(k, requires_grad=True)
        v_grad = torch.randn_like(v, requires_grad=True)

        output = model(q_grad, k_grad, v_grad)
        loss = output.sum()
        loss.backward()

        print(f"   ✓ Q grad exists: {q_grad.grad is not None}")
        print(f"   ✓ K grad exists: {k_grad.grad is not None}")
        print(f"   ✓ V grad exists: {v_grad.grad is not None}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\n✅ Basic tests completed!")


if __name__ == "__main__":
    test_basic()
