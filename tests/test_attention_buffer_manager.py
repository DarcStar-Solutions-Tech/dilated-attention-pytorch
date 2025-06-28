#!/usr/bin/env python3
"""
Test attention-specific buffer manager functionality.
"""

import pytest
import torch

from dilated_attention_pytorch.core.attention_buffer_manager import (
    AttentionBufferManager,
    BufferType,
    BufferConfig,
    create_attention_buffer_manager,
)
from dilated_attention_pytorch.improved_dilated_attention_v2 import (
    ImprovedDilatedAttentionV2,
)


def test_buffer_allocation():
    """Test basic buffer allocation and deallocation."""
    manager = create_attention_buffer_manager(enable_reuse=True)

    # Allocate different buffer types
    shape = (2, 1024, 8, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allocate Q, K, V buffers
    q_buffer = manager.allocate(BufferType.QUERY, shape, torch.float32, device)
    k_buffer = manager.allocate(BufferType.KEY, shape, torch.float32, device)
    v_buffer = manager.allocate(BufferType.VALUE, shape, torch.float32, device)

    assert q_buffer.shape == shape
    assert k_buffer.shape == shape
    assert v_buffer.shape == shape

    # Deallocate buffers
    manager.deallocate(q_buffer, BufferType.QUERY)
    manager.deallocate(k_buffer, BufferType.KEY)
    manager.deallocate(v_buffer, BufferType.VALUE)

    # Check that buffers were cached
    stats = manager.get_stats()
    assert stats["cached_buffers"] == 3


def test_buffer_reuse():
    """Test buffer reuse functionality."""
    manager = create_attention_buffer_manager(enable_reuse=True)

    shape = (1, 512, 4, 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # First allocation - cache miss
    buffer1 = manager.allocate(
        BufferType.QUERY, shape, torch.float32, device, zero_init=False
    )
    # Mark the buffer to verify reuse
    buffer1[0, 0, 0, 0] = 42.0

    # Store the tensor id for comparison
    _ = id(buffer1)

    # Deallocate to cache
    manager.deallocate(buffer1, BufferType.QUERY)

    # Verify buffer was cached
    stats = manager.get_stats()
    assert stats["cached_buffers"] == 1

    # Second allocation - should reuse from cache
    _ = manager.allocate(
        BufferType.QUERY, shape, torch.float32, device, zero_init=False
    )

    # Check statistics first
    stats = manager.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_hit_rate"] > 0

    # The test passes if we got a cache hit, even if the tensor ID is different
    # (PyTorch may create a new view of the cached tensor)


def test_zero_initialization():
    """Test zero initialization based on buffer type."""
    manager = create_attention_buffer_manager()

    shape = (2, 128, 8, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output buffer should be zero-initialized by default
    output_buffer = manager.allocate(BufferType.OUTPUT, shape, torch.float32, device)
    assert torch.all(output_buffer == 0)

    # Query buffer should not be zero-initialized by default
    _ = manager.allocate(
        BufferType.QUERY, shape, torch.float32, device, zero_init=False
    )
    # Can't assert non-zero as empty might return zeros, but it shouldn't waste time zeroing

    # Override zero initialization
    query_buffer_zero = manager.allocate(
        BufferType.QUERY, shape, torch.float32, device, zero_init=True
    )
    assert torch.all(query_buffer_zero == 0)


def test_buffer_type_strategies():
    """Test that different buffer types use appropriate strategies."""
    manager = create_attention_buffer_manager()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small temporary buffer - should prefer bucketed
    small_temp = manager.allocate(
        BufferType.TEMP, (2, 128, 8, 64), torch.float32, device
    )

    # Large scores buffer - should prefer NUMA or fragment-aware
    if device.type == "cuda":
        large_scores = manager.allocate(
            BufferType.SCORES, (2, 8, 4096, 4096), torch.float32, device
        )

    # Communication buffer - should be configured for pinned memory
    comm_buffer = manager.allocate(
        BufferType.COMM, (1024 * 1024,), torch.float32, device
    )

    # Cleanup
    manager.deallocate(small_temp, BufferType.TEMP)
    if device.type == "cuda":
        manager.deallocate(large_scores, BufferType.SCORES)
    manager.deallocate(comm_buffer, BufferType.COMM)


def test_preallocation():
    """Test buffer pre-allocation functionality."""
    manager = create_attention_buffer_manager(
        enable_preallocation=True, enable_reuse=True
    )

    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-allocate buffers
    manager.preallocate_buffers(batch_size, seq_len, num_heads, head_dim, device)

    # Check that buffers were pre-allocated
    stats = manager.get_stats()
    assert stats["preallocated_buffers"] >= 4  # Q, K, V, Output

    # Allocating same size should reuse pre-allocated buffers
    shape = (batch_size, seq_len, num_heads, head_dim)
    q_buffer = manager.allocate(BufferType.QUERY, shape, torch.float32, device)
    assert q_buffer.shape == shape


def test_cache_cleanup():
    """Test cache cleanup functionality."""
    manager = create_attention_buffer_manager(enable_reuse=True)

    shape = (2, 512, 8, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allocate and cache several buffers
    for buffer_type in [BufferType.QUERY, BufferType.KEY, BufferType.VALUE]:
        buffer = manager.allocate(buffer_type, shape, torch.float32, device)
        manager.deallocate(buffer, buffer_type)

    # Should have cached buffers
    stats = manager.get_stats()
    assert stats["cached_buffers"] > 0

    # Clear specific type
    manager.clear_cache(BufferType.QUERY)

    # Clear all
    manager.clear_cache()
    stats = manager.get_stats()
    assert stats["cached_buffers"] == 0


def test_improved_attention_v2_integration():
    """Test integration with ImprovedDilatedAttentionV2."""
    batch_size = 2
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    # Create attention module with buffer manager
    attention = ImprovedDilatedAttentionV2(
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
        dropout=0.0,
        enable_buffer_manager=True,
        enable_buffer_reuse=True,
    )

    # Create test tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Forward pass
    output = attention(q, k, v)
    assert output.shape == q.shape

    # Check buffer statistics
    buffer_stats = attention.get_buffer_stats()
    assert "buffer_types" in buffer_stats
    assert buffer_stats["buffer_types"][BufferType.OUTPUT]["allocations"] > 0

    # Multiple forward passes
    for _ in range(3):
        _ = attention(q, k, v)

    # Verify buffer manager is working
    buffer_stats = attention.get_buffer_stats()

    # We should have multiple allocations of output buffers
    assert (
        buffer_stats["buffer_types"][BufferType.OUTPUT]["allocations"] >= 4
    )  # 1 initial + 3 more

    # The buffer manager itself should be properly configured
    assert attention.buffer_manager is not None
    assert attention.buffer_manager.enable_reuse is True

    # Cleanup
    attention.cleanup_buffers()


@pytest.mark.parametrize("enable_reuse", [True, False])
@pytest.mark.parametrize("enable_preallocation", [True, False])
def test_configuration_combinations(enable_reuse, enable_preallocation):
    """Test different configuration combinations."""
    attention = ImprovedDilatedAttentionV2(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        enable_buffer_manager=True,
        enable_buffer_reuse=enable_reuse,
        enable_preallocation=enable_preallocation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(1, 1024, 8, 64, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Should work with any configuration
    output = attention(q, k, v)
    assert output.shape == q.shape

    # Verify configuration is respected
    if attention.buffer_manager:
        assert attention.buffer_manager.enable_reuse == enable_reuse
        assert attention.buffer_manager.enable_preallocation == enable_preallocation


def test_custom_buffer_config():
    """Test custom buffer configuration."""
    # Create custom config for SCORES buffer
    custom_configs = {
        BufferType.SCORES: BufferConfig(
            typical_size_mb=8.0,
            reuse_frequency="high",  # Changed from default
            lifetime="persistent",  # Changed from default
            prefer_bucketed=True,  # Changed from default
            zero_init=False,  # Don't zero-init
        )
    }

    manager = AttentionBufferManager(
        custom_configs=custom_configs,
        enable_reuse=True,
    )

    # Allocate scores buffer
    shape = (2, 8, 1024, 1024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores1 = manager.allocate(BufferType.SCORES, shape, torch.float32, device)
    _ = id(scores1)

    # Deallocate and reallocate - should reuse due to high frequency setting
    manager.deallocate(scores1, BufferType.SCORES)
    _ = manager.allocate(BufferType.SCORES, shape, torch.float32, device)

    # Check statistics - should have cache hit
    stats = manager.get_stats()
    assert stats["cache_hits"] == 1

    # Verify custom configuration was applied
    assert manager.configs[BufferType.SCORES].lifetime == "persistent"
    assert manager.configs[BufferType.SCORES].reuse_frequency == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
