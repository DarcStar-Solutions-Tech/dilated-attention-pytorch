"""
Test suite for production-ready Ring Dilated Attention.

This module tests the production features including:
- Gradient checkpointing
- Memory pool management
- Error recovery
- Mixed precision support
- Performance optimizations
"""

import pytest
import torch

from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
    MemoryPool,
    create_production_ring_attention,
)


class TestMemoryPool:
    """Test memory pool functionality."""

    def test_buffer_allocation_and_reuse(self):
        """Test that buffers are properly allocated and reused."""
        pool = MemoryPool(max_size=5)

        # Allocate a buffer
        shape = torch.Size([2, 4, 8])
        dtype = torch.float32
        device = torch.device("cpu")

        buffer1 = pool.get_buffer(shape, dtype, device)
        assert buffer1.shape == shape
        assert buffer1.dtype == dtype

        # Return buffer to pool
        pool.return_buffer(buffer1)

        # Get another buffer - should reuse
        buffer2 = pool.get_buffer(shape, dtype, device)
        assert buffer2 is buffer1  # Same object reused

        # Check statistics
        stats = pool.get_stats()
        assert stats["allocations"] == 1
        assert stats["reuses"] == 1
        assert stats["reuse_rate"] == 0.5

    def test_pool_size_limit(self):
        """Test that pool respects size limit."""
        pool = MemoryPool(max_size=2)
        shape = torch.Size([10, 10])

        # Create and return multiple buffers
        buffers = []
        for i in range(5):
            buf = pool.get_buffer(shape, torch.float32, torch.device("cpu"))
            buffers.append(buf)

        for buf in buffers:
            pool.return_buffer(buf)

        # Only max_size buffers should be pooled
        stats = pool.get_stats()
        assert stats["pooled_buffers"] == 2


class TestRingAttentionConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = RingAttentionConfig(
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            dropout=0.1,
            use_gradient_checkpointing=True,
        )
        assert config.segment_lengths == [2048, 4096]
        assert config.dilation_rates == [1, 2]

    def test_invalid_config(self):
        """Test invalid configurations are rejected."""
        # Mismatched lengths
        with pytest.raises(AssertionError):
            RingAttentionConfig(
                segment_lengths=[2048, 4096],
                dilation_rates=[1],  # Wrong length
            )

        # Invalid segment length
        with pytest.raises(AssertionError):
            RingAttentionConfig(segment_lengths=[0, 4096], dilation_rates=[1, 2])


class TestRingDilatedAttentionProduction:
    """Test production Ring Attention implementation."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def basic_config(self):
        return RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            dropout=0.0,
            use_gradient_checkpointing=False,
            use_memory_pool=True,
        )

    @pytest.fixture
    def test_tensors(self, device):
        """Create test tensors."""
        batch_size, seq_len, num_heads, head_dim = 2, 2048, 8, 64

        torch.manual_seed(42)
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        return query, key, value

    def test_basic_forward(self, basic_config, test_tensors):
        """Test basic forward pass."""
        model = RingDilatedAttentionProduction(basic_config)
        query, key, value = test_tensors

        output = model(query, key, value, is_causal=False)

        assert output.shape == query.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_checkpointing(self, device, test_tensors):
        """Test gradient checkpointing reduces memory usage."""
        query, key, value = test_tensors
        query.requires_grad_(True)

        # Without gradient checkpointing
        config_no_cp = RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            use_gradient_checkpointing=False,
        )
        model_no_cp = RingDilatedAttentionProduction(config_no_cp)

        # With gradient checkpointing
        config_with_cp = RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            use_gradient_checkpointing=True,
        )
        model_with_cp = RingDilatedAttentionProduction(config_with_cp)

        # Compare outputs (should be identical in eval mode)
        model_no_cp.eval()
        model_with_cp.eval()

        with torch.no_grad():
            output_no_cp = model_no_cp(query, key, value)
            output_with_cp = model_with_cp(query, key, value)

        torch.testing.assert_close(output_no_cp, output_with_cp, rtol=1e-5, atol=1e-5)

        # In training mode, gradient checkpointing should work
        model_with_cp.train()
        output_train = model_with_cp(query, key, value)
        loss = output_train.sum()
        loss.backward()

        assert query.grad is not None

    def test_memory_pool_usage(self, device, test_tensors):
        """Test memory pool is used effectively."""
        config = RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            use_memory_pool=True,
            memory_pool_size=5,
        )
        model = RingDilatedAttentionProduction(config)
        query, key, value = test_tensors

        # Run multiple forward passes
        for _ in range(3):
            _ = model(query, key, value)

        # Check pool statistics
        stats = model.get_memory_stats()
        pool_stats = stats.get("memory_pool", {})

        # Should have some reuses after multiple passes
        assert pool_stats.get("reuses", 0) > 0
        assert pool_stats.get("reuse_rate", 0) > 0

    def test_mixed_precision(self, device, test_tensors):
        """Test mixed precision support."""
        if device.type != "cuda":
            pytest.skip("Mixed precision requires CUDA")

        config = RingAttentionConfig(
            segment_lengths=[512, 1024], dilation_rates=[1, 2], mixed_precision=True
        )
        model = RingDilatedAttentionProduction(config)
        query, key, value = test_tensors

        # Model should use float16 internally
        assert model.dtype == torch.float16

        # Should handle float32 inputs
        output = model(query, key, value)
        assert output.dtype == query.dtype  # Output matches input dtype

    def test_causal_masking(self, basic_config, test_tensors):
        """Test causal masking works correctly."""
        model = RingDilatedAttentionProduction(basic_config)
        query, key, value = test_tensors

        output_causal = model(query, key, value, is_causal=True)
        output_non_causal = model(query, key, value, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal)

    def test_different_sequence_lengths(self, basic_config, device):
        """Test handling of various sequence lengths."""
        model = RingDilatedAttentionProduction(basic_config)
        batch_size, num_heads, head_dim = 1, 8, 64

        # Test different sequence lengths
        for seq_len in [256, 512, 1024, 2048, 3000]:
            query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            output = model(query, key, value)
            assert output.shape == query.shape
            assert not torch.isnan(output).any()

    def test_cache_effectiveness(self, basic_config, device):
        """Test that caches are used effectively."""
        model = RingDilatedAttentionProduction(basic_config)

        batch_size, seq_len, num_heads, head_dim = 1, 2048, 8, 64
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # First forward pass - caches should be populated
        _ = model(query, key, value, is_causal=True)

        stats1 = model.get_memory_stats()
        cached_indices1 = stats1["cached_indices"]
        cached_masks1 = stats1["cached_masks"]

        assert cached_indices1 > 0
        assert cached_masks1 > 0

        # Second forward pass - caches should be reused
        _ = model(query, key, value, is_causal=True)

        stats2 = model.get_memory_stats()
        cached_indices2 = stats2["cached_indices"]
        cached_masks2 = stats2["cached_masks"]

        # Cache sizes should be the same (no new allocations)
        assert cached_indices2 == cached_indices1
        assert cached_masks2 == cached_masks1

    def test_clear_caches(self, basic_config, device):
        """Test cache clearing functionality."""
        model = RingDilatedAttentionProduction(basic_config)

        # Populate caches
        batch_size, seq_len, num_heads, head_dim = 1, 2048, 8, 64
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        _ = model(query, key, value)

        # Clear caches
        model.clear_caches()

        stats = model.get_memory_stats()
        assert stats["cached_indices"] == 0
        assert stats["cached_masks"] == 0

        if model.memory_pool:
            assert stats["memory_pool"]["pooled_buffers"] == 0

    def test_factory_function(self, device):
        """Test factory function creates proper instance."""
        model = create_production_ring_attention(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            dropout=0.1,
            use_gradient_checkpointing=True,
        )

        assert isinstance(model, RingDilatedAttentionProduction)
        assert model.config.segment_lengths == [512, 1024]
        assert model.config.dropout == 0.1
        assert model.config.use_gradient_checkpointing is True


class TestRingAttentionModes:
    """Test different operation modes."""

    @pytest.fixture
    def small_tensors(self):
        """Small tensors for testing."""
        batch_size, seq_len, num_heads, head_dim = 1, 1024, 4, 32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        return query, key, value

    def test_single_mode(self, small_tensors):
        """Test single GPU mode."""
        config = RingAttentionConfig(
            segment_lengths=[256, 512], dilation_rates=[1, 2], ring_size=1
        )
        model = RingDilatedAttentionProduction(config)
        assert model.mode == "single"

        query, key, value = small_tensors
        output = model(query, key, value)
        assert output.shape == query.shape

    def test_simulated_mode(self, small_tensors):
        """Test simulated ring mode on single GPU."""
        config = RingAttentionConfig(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            ring_size=4,  # Simulate 4-way ring on single GPU
        )
        model = RingDilatedAttentionProduction(config)
        assert model.mode == "simulated"

        query, key, value = small_tensors
        output = model(query, key, value)
        assert output.shape == query.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
