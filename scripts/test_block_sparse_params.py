#!/usr/bin/env python3
"""
Test script to verify BlockSparseRingDilatedAttention parameter handling.
"""

import torch
from dilated_attention_pytorch import BlockSparseRingDilatedAttention
from dilated_attention_pytorch.core import create_dilated_attention


def test_block_sparse_parameters():
    """Test different ways of creating BlockSparseRingDilatedAttention."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 1024

    print("Testing BlockSparseRingDilatedAttention parameter handling\n")

    # Test 1: Direct instantiation with sparsity_ratio
    print("1. Direct instantiation with sparsity_ratio:")
    try:
        attention1 = BlockSparseRingDilatedAttention(
            segment_lengths=[256, 512, 1024],
            dilation_rates=[1, 2, 4],
            sparsity_ratio=0.95,
            enable_memory_pool=True,
        )
        print(
            f"   ✓ Success! Sparsity ratio: {attention1.sparse_config.sparsity_ratio}"
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 2: Using SparsePatternConfig
    print("\n2. Using SparsePatternConfig:")
    try:
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
            SparsePatternConfig,
        )

        sparse_config = SparsePatternConfig(
            pattern_type="dilated_sparse", sparsity_ratio=0.9, block_size=128
        )

        attention2 = BlockSparseRingDilatedAttention(
            segment_lengths=[256, 512, 1024],
            dilation_rates=[1, 2, 4],
            sparse_config=sparse_config,
            enable_memory_pool=True,
        )
        print(
            f"   ✓ Success! Sparsity ratio: {attention2.sparse_config.sparsity_ratio}"
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 3: Through factory
    print("\n3. Through factory pattern:")
    try:
        attention3 = create_dilated_attention(
            "block_sparse_ring",
            segment_lengths=[256, 512, 1024],
            dilation_rates=[1, 2, 4],
            sparsity_ratio=0.8,
        )
        print("   ✓ Success! Created via factory")
        if hasattr(attention3, "sparse_config"):
            print(f"   Sparsity ratio: {attention3.sparse_config.sparsity_ratio}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test 4: Test forward pass
    print("\n4. Testing forward pass:")
    try:
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[256, 512, 1024],
            dilation_rates=[1, 2, 4],
            sparsity_ratio=0.95,
        )

        # Create inputs
        batch_size = 1
        num_heads = 8
        head_dim = 64
        shape = (batch_size, seq_len, num_heads, head_dim)

        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        # Forward pass
        output = attention(q, k, v)
        print(f"   ✓ Forward pass successful! Output shape: {output.shape}")
        print(
            f"   Sparsity: {attention.sparse_config.sparsity_ratio * 100:.0f}% sparse"
        )
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    test_block_sparse_parameters()
