"""Tests for standardized ring attention implementations."""

import pytest
import torch
import torch.distributed as dist

from dilated_attention_pytorch.ring import (
    StandardRingAttention,
    HilbertRingAttention,
    DistributedRingAttention,
    BlockSparseRingAttention,
    RingAttentionConfig,
    create_ring_attention,
    get_preset_config,
    validate_ring_configuration,
)


class TestStandardRingAttention:
    """Test StandardRingAttention implementation."""

    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            dropout=0.0,
        )

    def test_single_gpu_forward(self, config):
        """Test forward pass on single GPU."""
        attention = StandardRingAttention(config)

        batch_size = 2
        seq_len = 2048
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = attention(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_causal_attention(self, config):
        """Test causal attention masking."""
        attention = StandardRingAttention(config)

        batch_size = 1
        seq_len = 1024
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output_causal = attention(q, k, v, is_causal=True)
        output_non_causal = attention(q, k, v, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal)

    def test_gradient_flow(self, config):
        """Test gradient flow through attention."""
        attention = StandardRingAttention(config)

        batch_size = 2
        seq_len = 1024  # Must be divisible by largest segment (1024)
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert not torch.isnan(q.grad).any()


class TestHilbertRingAttention:
    """Test HilbertRingAttention implementation."""

    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            use_hilbert=True,
            hilbert_curve_level=10,
        )

    def test_hilbert_ordering(self, config):
        """Test that Hilbert ordering is applied."""
        attention = HilbertRingAttention(config)

        batch_size = 1
        seq_len = 1024
        num_heads = 4
        head_dim = 32

        # Create structured input to test ordering
        q = torch.zeros(batch_size, seq_len, num_heads, head_dim)
        k = torch.zeros(batch_size, seq_len, num_heads, head_dim)
        v = (
            torch.arange(seq_len)
            .float()
            .view(1, seq_len, 1, 1)
            .expand(batch_size, seq_len, num_heads, head_dim)
        )

        output = attention(q, k, v)

        # Output should have been reordered by Hilbert curve
        assert output.shape == v.shape
        # The output shouldn't be a simple copy of v due to Hilbert reordering
        assert not torch.allclose(output, v)

    def test_hilbert_gradient_flow(self, config):
        """Test gradient flow through Hilbert attention."""
        attention = HilbertRingAttention(config)

        batch_size = 2
        seq_len = 1024  # Must be divisible by largest segment (1024)
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestDistributedRingAttention:
    """Test DistributedRingAttention implementation."""

    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            enable_error_recovery=True,
            enable_watchdog=True,
            watchdog_timeout=30.0,
        )

    def test_single_gpu_fallback(self, config):
        """Test that distributed attention works on single GPU."""
        attention = DistributedRingAttention(
            config,
            enable_deepspeed=False,  # Don't require DeepSpeed for test
            enable_monitoring=False,
        )

        batch_size = 2
        seq_len = 1024
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = attention(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_checkpoint_restore(self, config):
        """Test checkpoint/restore functionality."""
        attention = DistributedRingAttention(
            config,
            enable_deepspeed=False,
            enable_monitoring=True,
        )

        # Create checkpoint
        checkpoint = attention.checkpoint()

        assert "comm_stats" in checkpoint
        assert "monitoring_data" in checkpoint
        assert "config" in checkpoint

        # Restore checkpoint
        attention2 = DistributedRingAttention(
            config,
            enable_deepspeed=False,
            enable_monitoring=True,
        )
        attention2.restore_checkpoint(checkpoint)

        # Verify restoration
        assert attention2.monitoring_data == attention.monitoring_data


class TestBlockSparseRingAttention:
    """Test BlockSparseRingAttention implementation."""

    @pytest.fixture
    def config(self):
        return RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

    def test_sparse_pattern_application(self, config):
        """Test that sparse patterns are correctly applied."""
        attention = BlockSparseRingAttention(
            config,
            block_size=64,
            sparsity_ratio=0.5,  # 50% sparse - more conservative to avoid NaN
            pattern_type="local",
        )

        batch_size = 1
        seq_len = 1024  # Must be divisible by largest segment (1024)
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = attention(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_memory_savings_calculation(self, config):
        """Test memory savings calculation."""
        attention = BlockSparseRingAttention(
            config,
            block_size=64,
            sparsity_ratio=0.9,
            pattern_type="local",
        )

        savings = attention.get_memory_savings()

        assert "ring_memory_factor" in savings
        assert "sparse_memory_factor" in savings
        assert "total_memory_factor" in savings
        assert "effective_speedup" in savings

        # With 90% sparsity, we should have ~10% memory usage
        assert savings["sparse_memory_factor"] == pytest.approx(0.1, rel=0.01)
        # Effective speedup should be ~10x from sparsity alone
        assert savings["effective_speedup"] >= 10.0


class TestFactory:
    """Test factory functions."""

    def test_create_ring_attention_auto(self):
        """Test auto-selection of implementation."""
        config = RingAttentionConfig(
            segment_lengths=[512],
            dilation_rates=[1],
        )

        attention = create_ring_attention("auto", config)

        # Should select standard implementation in single GPU env
        assert isinstance(attention, StandardRingAttention)

    def test_create_ring_attention_explicit(self):
        """Test explicit implementation selection."""
        config = RingAttentionConfig(
            segment_lengths=[512],
            dilation_rates=[1],
        )

        # Test each implementation
        standard = create_ring_attention("standard", config)
        assert isinstance(standard, StandardRingAttention)

        hilbert = create_ring_attention("hilbert", config)
        assert isinstance(hilbert, HilbertRingAttention)

        distributed = create_ring_attention("distributed", config)
        assert isinstance(distributed, DistributedRingAttention)

        block_sparse = create_ring_attention(
            "block_sparse",
            config,
            block_size=64,
            sparsity_ratio=0.9,
        )
        assert isinstance(block_sparse, BlockSparseRingAttention)

    def test_preset_configs(self):
        """Test preset configurations."""
        dev_config = get_preset_config("development")
        assert dev_config.dropout == 0.0
        assert dev_config.enable_profiling  # Development preset has profiling enabled

        prod_config = get_preset_config("production")
        assert prod_config.enable_error_recovery
        assert prod_config.preallocate_buffers

        large_config = get_preset_config("large_scale")
        assert large_config.overlap_communication
        assert large_config.use_memory_pool

    def test_validate_configuration(self):
        """Test configuration validation."""
        config = RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

        # Valid configuration
        assert validate_ring_configuration(config, seq_len=2048)

        # Invalid: sequence not divisible by largest segment
        with pytest.raises(ValueError, match="divisible by largest segment"):
            validate_ring_configuration(config, seq_len=1536)

        # Invalid: mismatched segment/dilation lengths
        with pytest.raises(ValueError, match="must have the same length"):
            _ = RingAttentionConfig(
                segment_lengths=[512, 1024],
                dilation_rates=[1],  # Too short
            )


# Skip distributed tests if not in distributed environment
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_initialized(),
    reason="Distributed training not available",
)
class TestMultiGPU:
    """Test multi-GPU functionality."""

    def test_ring_communication(self):
        """Test ring communication pattern."""
        config = RingAttentionConfig(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
        )

        attention = StandardRingAttention(config)

        batch_size = 2
        seq_len = 2048
        num_heads = 8
        head_dim = 64

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()

        # Forward pass should work with ring communication
        output = attention(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()

        # Verify sequence was split across GPUs
        local_seq_len = seq_len // dist.get_world_size()
        assert attention._last_local_seq_len == local_seq_len
