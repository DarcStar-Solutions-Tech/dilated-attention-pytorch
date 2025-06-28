#!/usr/bin/env python3
"""
Test memory pool integration with BlockSparseRingDilatedAttention.
"""

import pytest
import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


@pytest.mark.parametrize("enable_memory_pool", [False, True])
@pytest.mark.parametrize("seq_len", [1024, 4096])
@pytest.mark.parametrize(
    "pattern_type", ["local_window", "dilated_sparse", "global_local"]
)
def test_block_sparse_memory_pool(enable_memory_pool, seq_len, pattern_type):
    """Test BlockSparseRingDilatedAttention with and without memory pools."""
    batch_size = 2
    num_heads = 8
    head_dim = 64

    # Create sparse config
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=0.1,  # 90% sparse
        block_size=128,
        local_window_size=512,
        global_tokens=64,
    )

    # Create attention module
    attention = BlockSparseRingDilatedAttention(
        segment_lengths=[512, 1024, 2048],
        dilation_rates=[1, 2, 4],
        sparse_config=sparse_config,
        enable_memory_pool=enable_memory_pool,
        lightweight_pool=True,
    )

    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test forward pass
    output = attention(q, k, v)
    assert output.shape == q.shape

    # Test with causal mask
    output_causal = attention(q, k, v, is_causal=True)
    assert output_causal.shape == q.shape

    # Test with attention weights return (only for smaller sequences)
    if seq_len <= 2048:
        output_with_weights, weights = attention(q, k, v, return_attention_weights=True)
        assert output_with_weights.shape == q.shape
        assert isinstance(weights, dict)
        assert "block_indices" in weights
        assert "block_values" in weights

    # Cleanup
    attention.cleanup_buffers()


@pytest.mark.parametrize("sparsity_ratio", [0.05, 0.1, 0.2])
def test_memory_pool_with_different_sparsity(sparsity_ratio):
    """Test memory pool performance with different sparsity levels."""
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=sparsity_ratio,
        block_size=128,
    )

    # Test both with and without memory pool
    for enable_pool in [False, True]:
        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[1024, 2048, 4096],
            dilation_rates=[1, 2, 4],
            sparse_config=sparse_config,
            enable_memory_pool=enable_pool,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = attention(q, k, v)
        assert output.shape == q.shape

        # Verify memory pool is being used
        if enable_pool:
            assert hasattr(attention, "_memory_pool")
            assert attention._memory_pool is not None

        attention.cleanup_buffers()


def test_memory_pool_cleanup():
    """Test that memory pools and caches are properly cleaned up."""
    attention = BlockSparseRingDilatedAttention(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        sparse_config=SparsePatternConfig(sparsity_ratio=0.1),
        enable_memory_pool=True,
    )

    # Create some cached masks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(1, 1024, 8, 64, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # This should create causal masks in cache
    _ = attention(q, k, v, is_causal=True)

    # Verify cache exists
    assert hasattr(attention, "_causal_mask_cache")
    assert len(attention._causal_mask_cache) > 0

    # Cleanup
    attention.cleanup_buffers()

    # Verify cache is cleared
    assert len(attention._causal_mask_cache) == 0


def test_memory_pool_with_large_sequences():
    """Test memory pool benefits with large sequences."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for large sequence test")

    batch_size = 1
    seq_len = 16384  # Large sequence
    num_heads = 8
    head_dim = 64

    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=0.05,  # 95% sparse for memory efficiency
        block_size=256,
        local_window_size=1024,
    )

    attention = BlockSparseRingDilatedAttention(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        sparse_config=sparse_config,
        enable_memory_pool=True,
        lightweight_pool=True,
    )

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Should not OOM with memory pool
    output = attention(q, k, v)
    assert output.shape == q.shape

    attention.cleanup_buffers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
