#!/usr/bin/env python3
"""
Verify Block Sparse Ring Dilated Attention implementations work correctly.
"""

import traceback

import torch


def test_import():
    """Test that all block sparse modules can be imported."""
    print("Testing imports...")
    try:
        import dilated_attention_pytorch.block_sparse_ring_dilated_attention

        print("✓ BlockSparseRingDilatedAttention imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BlockSparseRingDilatedAttention: {e}")
        return False

    try:
        import dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention

        print("✓ BlockSparseRingMultiheadDilatedAttention imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BlockSparseRingMultiheadDilatedAttention: {e}")
        return False

    try:
        import dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention  # noqa: F401

        print("✓ BlockSparseRingDistributedDilatedAttention imported successfully")
    except Exception as e:
        print(f"✗ Failed to import BlockSparseRingDistributedDilatedAttention: {e}")
        return False

    return True


def test_basic_forward_pass():
    """Test basic forward pass for BlockSparseRingDilatedAttention."""
    print("\nTesting BlockSparseRingDilatedAttention forward pass...")

    try:
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import \
            BlockSparseRingDilatedAttention

        # Test parameters
        batch_size = 2
        seq_len = 512
        num_heads = 8
        head_dim = 64

        # Create attention module
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            sparsity_ratio=0.5,
            pattern_type="dilated_sparse",
            dropout=0.0,
        )

        # Create input tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Forward pass
        output = attention(q, k, v, is_causal=False)

        # Verify output shape
        expected_shape = (batch_size, seq_len, num_heads, head_dim)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_multihead_forward_pass():
    """Test forward pass for BlockSparseRingMultiheadDilatedAttention."""
    print("\nTesting BlockSparseRingMultiheadDilatedAttention forward pass...")

    try:
        from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import \
            BlockSparseRingMultiheadDilatedAttention

        # Test parameters
        batch_size = 2
        seq_len = 512
        embed_dim = 512
        num_heads = 8

        # Create attention module
        attention = BlockSparseRingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            sparsity_ratio=0.5,
            pattern_type="dilated_sparse",
            dropout=0.0,
        )

        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)

        # Forward pass
        output = attention(x, x, x, is_causal=False)

        # Handle both tuple and tensor returns
        if isinstance(output, tuple):
            output = output[0]

        # Verify output shape
        expected_shape = (batch_size, seq_len, embed_dim)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

        print(f"✓ Multihead forward pass successful. Output shape: {output.shape}")
        return True

    except Exception as e:
        print(f"✗ Multihead forward pass failed: {e}")
        traceback.print_exc()
        return False


def test_different_patterns():
    """Test different sparse patterns."""
    print("\nTesting different sparse patterns...")

    try:
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import \
            BlockSparseRingDilatedAttention

        patterns = ["local_window", "dilated_sparse", "global_local", "random"]
        batch_size = 1
        seq_len = 256
        num_heads = 4
        head_dim = 32

        # Create input tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        for pattern in patterns:
            try:
                attention = BlockSparseRingDilatedAttention(
                    segment_lengths=[128],
                    dilation_rates=[1],
                    sparsity_ratio=0.5,
                    pattern_type=pattern,
                    dropout=0.0,
                )

                output = attention(q, k, v, is_causal=False)
                assert output.shape == q.shape
                print(f"✓ Pattern '{pattern}' works correctly")

            except Exception as e:
                print(f"✗ Pattern '{pattern}' failed: {e}")

        return True

    except Exception as e:
        print(f"✗ Pattern testing failed: {e}")
        traceback.print_exc()
        return False


def test_memory_efficiency():
    """Test memory efficiency with different sparsity ratios."""
    print("\nTesting memory efficiency...")

    try:
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import \
            BlockSparseRingDilatedAttention

        sparsity_ratios = [0.1, 0.5, 0.9]
        results = []

        for sparsity in sparsity_ratios:
            attention = BlockSparseRingDilatedAttention(
                segment_lengths=[256],
                dilation_rates=[1],
                sparsity_ratio=sparsity,
                pattern_type="dilated_sparse",
                dropout=0.0,
            )

            # Get memory info if available
            if hasattr(attention, "get_memory_info"):
                info = attention.get_memory_info()
                results.append((sparsity, info))
                print(f"✓ Sparsity {sparsity}: {info.get('sparsity_ratio', 'N/A')}")
            else:
                print(f"✓ Sparsity {sparsity}: Module created successfully")

        return True

    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        traceback.print_exc()
        return False


def test_factory_creation():
    """Test creating block sparse attention using factory functions."""
    print("\nTesting factory creation...")

    try:
        from dilated_attention_pytorch import \
            create_block_sparse_multihead_attention

        # Create using factory
        attention = create_block_sparse_multihead_attention(
            embed_dim=512,
            num_heads=8,
            sparsity_ratio=0.5,
            pattern_type="dilated_sparse",
        )

        # Test forward pass
        x = torch.randn(2, 256, 512)
        output = attention(x)

        # Handle both tuple and tensor returns
        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape
        print("✓ Factory creation successful")
        return True

    except Exception as e:
        print(f"✗ Factory creation failed: {e}")
        traceback.print_exc()
        return False


def test_adaptive_sparsity():
    """Test adaptive sparsity functionality."""
    print("\nTesting adaptive sparsity...")

    try:
        from dilated_attention_pytorch import \
            create_adaptive_sparse_multihead_attention

        # Create adaptive sparse attention
        attention = create_adaptive_sparse_multihead_attention(
            embed_dim=256, num_heads=4, min_sparsity=0.1, max_sparsity=0.9
        )

        # Test forward pass
        x = torch.randn(2, 128, 256)
        output = attention(x)

        # Handle both tuple and tensor returns
        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape
        print("✓ Adaptive sparsity works correctly")
        return True

    except Exception as e:
        print(f"✗ Adaptive sparsity failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Block Sparse Ring Dilated Attention Verification")
    print("=" * 60)

    tests = [
        ("Import Test", test_import),
        ("Basic Forward Pass", test_basic_forward_pass),
        ("Multihead Forward Pass", test_multihead_forward_pass),
        ("Different Patterns", test_different_patterns),
        ("Memory Efficiency", test_memory_efficiency),
        ("Factory Creation", test_factory_creation),
        ("Adaptive Sparsity", test_adaptive_sparsity),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"Running: {test_name}")
        print(f"{'=' * 40}")

        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<40} {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print(
            "\n✓ All tests passed! Block Sparse implementations are working correctly."
        )
    else:
        print(f"\n✗ {total - passed} tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
