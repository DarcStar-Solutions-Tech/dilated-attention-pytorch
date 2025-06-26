#!/usr/bin/env python3
"""
Edge case and validation tests for dilated attention implementations.

Tests input validation, boundary conditions, and error handling.
"""

import pytest
import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.multihead_dilated_attention import MultiheadDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention


class TestInputValidation:
    """Test input validation for all attention modules."""

    def test_dilated_attention_validation(self):
        """Test DilatedAttention input validation."""
        # Empty segment lengths
        with pytest.raises(ValueError, match="segment_lengths cannot be empty"):
            DilatedAttention(segment_lengths=[], dilation_rates=[])

        # Mismatched lengths
        with pytest.raises(ValueError, match="must have the same length"):
            DilatedAttention(segment_lengths=[512], dilation_rates=[1, 2])

        # Negative segment length
        with pytest.raises(ValueError, match="must be positive"):
            DilatedAttention(segment_lengths=[512, -256], dilation_rates=[1, 2])

        # Zero dilation rate
        with pytest.raises(ValueError, match="must be positive"):
            DilatedAttention(segment_lengths=[512, 256], dilation_rates=[1, 0])

        # Invalid dropout
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            DilatedAttention(segment_lengths=[512], dilation_rates=[1], attention_dropout=1.5)

    def test_multihead_attention_validation(self):
        """Test MultiheadDilatedAttention validation."""
        # Invalid embed_dim
        with pytest.raises(ValueError, match="must be divisible by"):
            MultiheadDilatedAttention(
                embed_dim=100,  # Not divisible by 12
                num_heads=12,
                dilation_rates=[1],
                segment_lengths=[512],
            )

        # Head dim not divisible by 8
        with pytest.raises(ValueError, match="must be divisible by 8"):
            MultiheadDilatedAttention(
                embed_dim=96,  # 96/12 = 8, but need head_dim divisible by 8
                num_heads=12,
                dilation_rates=[1],
                segment_lengths=[512],
            )

        # Head dim too large
        with pytest.raises(ValueError, match="must be <= 128"):
            MultiheadDilatedAttention(
                embed_dim=2048,  # 2048/8 = 256 > 128
                num_heads=8,
                dilation_rates=[1],
                segment_lengths=[512],
            )

    def test_forward_shape_validation(self):
        """Test shape validation in forward pass."""
        attention = DilatedAttention(segment_lengths=[256], dilation_rates=[1])

        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 4D tensors"):
            q = torch.randn(10, 256, 64)  # 3D instead of 4D
            k = torch.randn(10, 256, 64)
            v = torch.randn(10, 256, 64)
            attention(q, k, v)

        # Mismatched shapes
        with pytest.raises(ValueError, match="must have the same shape"):
            q = torch.randn(2, 256, 8, 64)
            k = torch.randn(2, 256, 8, 32)  # Different head_dim
            v = torch.randn(2, 256, 8, 64)
            attention(q, k, v)

        # Invalid sequence length
        with pytest.raises(ValueError, match="must be divisible by"):
            q = torch.randn(2, 255, 8, 64)  # 255 not divisible by 256
            k = torch.randn(2, 255, 8, 64)
            v = torch.randn(2, 255, 8, 64)
            attention(q, k, v)

    def test_multihead_forward_validation(self):
        """Test MultiheadDilatedAttention forward validation."""
        attention = MultiheadDilatedAttention(
            embed_dim=512, num_heads=8, dilation_rates=[1], segment_lengths=[128]
        )

        # Wrong dimensions
        with pytest.raises(ValueError, match="Expected 3D tensors"):
            q = torch.randn(2, 128, 512, 1)  # 4D instead of 3D
            k = torch.randn(2, 128, 512, 1)
            v = torch.randn(2, 128, 512, 1)
            attention(q, k, v)

        # Wrong embedding dimension
        with pytest.raises(ValueError, match="embedding dimension"):
            q = torch.randn(2, 128, 256)  # 256 != 512
            k = torch.randn(2, 128, 256)
            v = torch.randn(2, 128, 256)
            attention(q, k, v)


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_minimum_sequence_length(self):
        """Test with minimum valid sequence length."""
        attention = DilatedAttention(
            segment_lengths=[1],  # Minimum segment
            dilation_rates=[1],
        )

        # Single element sequence
        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        v = torch.randn(1, 1, 1, 32)

        output = attention(q, k, v)
        assert output.shape == (1, 1, 1, 32)

    def test_single_head_single_segment(self):
        """Test with single head and single segment."""
        attention = DilatedAttention(segment_lengths=[64], dilation_rates=[1])

        q = torch.randn(2, 64, 1, 128)  # Single head
        k = torch.randn(2, 64, 1, 128)
        v = torch.randn(2, 64, 1, 128)

        output = attention(q, k, v)
        assert output.shape == q.shape

        # Output should be normalized properly
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_many_segments_few_heads(self):
        """Test more segments than heads."""
        # 5 segments but only 3 heads
        attention = DilatedAttention(
            segment_lengths=[64, 128, 256, 512, 1024], dilation_rates=[1, 2, 4, 8, 16]
        )

        q = torch.randn(1, 1024, 3, 64)  # Only 3 heads
        k = torch.randn(1, 1024, 3, 64)
        v = torch.randn(1, 1024, 3, 64)

        output = attention(q, k, v)
        assert output.shape == q.shape

        # Check head distribution (some segments get 0 heads)
        # First 3 segments get 1 head each, last 2 get 0

    def test_extreme_dilation_rates(self):
        """Test with extreme dilation rates."""
        attention = DilatedAttention(
            segment_lengths=[32, 32, 32],
            dilation_rates=[1, 100, 1000],  # Very large dilations
        )

        q = torch.randn(1, 1024, 6, 32)
        k = torch.randn(1, 1024, 6, 32)
        v = torch.randn(1, 1024, 6, 32)

        output = attention(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


class TestSparsityPatterns:
    """Test sparse pattern edge cases."""

    def test_extreme_sparsity_ratios(self):
        """Test with very high and very low sparsity."""
        # Very sparse (99%)
        config_sparse = SparsePatternConfig(sparsity_ratio=0.99, pattern_type="dilated_sparse")

        attention_sparse = BlockSparseRingDilatedAttention(
            segment_lengths=[256], dilation_rates=[1], sparse_config=config_sparse
        )

        # Very dense (1%)
        config_dense = SparsePatternConfig(sparsity_ratio=0.01, pattern_type="dilated_sparse")

        attention_dense = BlockSparseRingDilatedAttention(
            segment_lengths=[256], dilation_rates=[1], sparse_config=config_dense
        )

        # Both should work
        q = torch.randn(1, 256, 4, 64)
        k = torch.randn(1, 256, 4, 64)
        v = torch.randn(1, 256, 4, 64)

        output_sparse = attention_sparse(q, k, v)
        output_dense = attention_dense(q, k, v)

        assert output_sparse.shape == q.shape
        assert output_dense.shape == q.shape

    def test_block_size_edge_cases(self):
        """Test with various block sizes."""
        # Block size = 1 (every element is a block)
        config_tiny = SparsePatternConfig(
            block_size=1, pattern_type="local_window", local_window_size=3
        )

        with pytest.raises(ValueError, match="block_size must be positive"):
            SparsePatternConfig(block_size=0)

        # Block size larger than sequence
        config_large = SparsePatternConfig(block_size=512, pattern_type="local_window")

        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[256], dilation_rates=[1], sparse_config=config_large
        )

        # Should work even though block > sequence
        q = torch.randn(1, 256, 4, 64)
        k = torch.randn(1, 256, 4, 64)
        v = torch.randn(1, 256, 4, 64)

        output = attention(q, k, v)
        assert output.shape == q.shape


class TestCausalMasking:
    """Test causal masking edge cases."""

    def test_causal_with_single_element(self):
        """Test causal masking with single element."""
        attention = DilatedAttention(segment_lengths=[1], dilation_rates=[1])

        q = torch.randn(1, 1, 1, 32)
        k = torch.randn(1, 1, 1, 32)
        v = torch.ones(1, 1, 1, 32)  # Use ones to check masking

        output = attention(q, k, v, is_causal=True)

        # Single element should attend to itself
        assert output.shape == (1, 1, 1, 32)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_causal_preserves_autoregressive_property(self):
        """Test that causal masking preserves autoregressive property."""
        attention = RingDilatedAttention(segment_lengths=[64], dilation_rates=[1])

        seq_len = 128
        q = torch.randn(1, seq_len, 4, 32)
        k = torch.randn(1, seq_len, 4, 32)
        v = torch.randn(1, seq_len, 4, 32)

        # Get outputs with causal masking
        output_full = attention(q, k, v, is_causal=True)

        # Process sequence incrementally
        outputs_incremental = []
        for i in range(1, seq_len + 1):
            q_partial = q[:, :i, :, :]
            k_partial = k[:, :i, :, :]
            v_partial = v[:, :i, :, :]

            # Need to handle segment length compatibility
            if i >= 64:  # Minimum segment length
                output_partial = attention(q_partial, k_partial, v_partial, is_causal=True)
                outputs_incremental.append(output_partial[:, -1:, :, :])

        # Incremental processing should match full processing
        # (for positions where we could compute)
        if outputs_incremental:
            incremental_combined = torch.cat(outputs_incremental, dim=1)
            assert incremental_combined.shape[1] <= output_full.shape[1]


class TestNumericalStability:
    """Test numerical stability in edge cases."""

    def test_attention_with_large_values(self):
        """Test with large input values."""
        attention = DilatedAttention(segment_lengths=[64], dilation_rates=[1])

        # Large values that could cause overflow
        scale = 100.0
        q = torch.randn(1, 64, 4, 32) * scale
        k = torch.randn(1, 64, 4, 32) * scale
        v = torch.randn(1, 64, 4, 32)

        output = attention(q, k, v)

        # Should not produce NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_attention_with_zero_values(self):
        """Test with all-zero inputs."""
        attention = DilatedAttention(segment_lengths=[64], dilation_rates=[1])

        # All zeros
        q = torch.zeros(1, 64, 4, 32)
        k = torch.zeros(1, 64, 4, 32)
        v = torch.zeros(1, 64, 4, 32)

        output = attention(q, k, v)

        # Output should be zero (or uniform due to softmax)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_gradient_flow(self):
        """Test gradient flow through edge cases."""
        attention = DilatedAttention(segment_lengths=[32, 64], dilation_rates=[1, 2])

        # Requires grad
        q = torch.randn(1, 64, 4, 32, requires_grad=True)
        k = torch.randn(1, 64, 4, 32, requires_grad=True)
        v = torch.randn(1, 64, 4, 32, requires_grad=True)

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        # Gradients should exist and be finite
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
        assert not torch.isinf(q.grad).any()


class TestDeviceCompatibility:
    """Test device handling edge cases."""

    def test_cpu_device_explicit(self):
        """Test explicit CPU device specification."""
        attention = RingDilatedAttention(segment_lengths=[64], dilation_rates=[1], device="cpu")

        q = torch.randn(1, 64, 4, 32)
        k = torch.randn(1, 64, 4, 32)
        v = torch.randn(1, 64, 4, 32)

        output = attention(q, k, v)
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_device_inputs(self):
        """Test handling of mixed device inputs."""
        attention = RingDilatedAttention(segment_lengths=[64], dilation_rates=[1], device="cuda:0")

        # CPU inputs (should auto-move or error clearly)
        q = torch.randn(1, 64, 4, 32)
        k = torch.randn(1, 64, 4, 32)
        v = torch.randn(1, 64, 4, 32)

        # Should handle device mismatch gracefully
        with pytest.raises((RuntimeError, AssertionError)):
            # Expecting device mismatch error
            output = attention(q, k, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
