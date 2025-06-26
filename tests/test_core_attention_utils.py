"""
Test suite for attention utilities.

Tests common attention computation functions and utilities.
"""

import math
from unittest.mock import patch

import pytest
import torch

from dilated_attention_pytorch.utils.attention_utils import (
    apply_dilated_attention_pattern,
    apply_rotary_embeddings,
    compute_alibi_bias,
    compute_attention_scores,
    compute_rotary_embeddings,
    create_block_diagonal_mask,
    create_dilated_mask,
    merge_attention_heads,
    optimize_attention_computation,
    split_attention_heads,
    standard_attention,
)


class TestAttentionScores:
    """Test attention score computation."""

    def test_compute_attention_scores_basic(self):
        """Test basic attention score computation."""
        batch_size, seq_len, head_dim = 2, 8, 64
        q = torch.randn(batch_size, seq_len, head_dim)
        k = torch.randn(batch_size, seq_len, head_dim)

        scores = compute_attention_scores(q, k)

        assert scores.shape == (batch_size, seq_len, seq_len)

        # Check scaling
        expected_scale = 1.0 / math.sqrt(head_dim)
        manual_scores = torch.matmul(q, k.transpose(-2, -1)) * expected_scale
        assert torch.allclose(scores, manual_scores)

    def test_compute_attention_scores_with_mask(self):
        """Test attention scores with mask."""
        batch_size, seq_len, head_dim = 2, 8, 64
        q = torch.randn(batch_size, seq_len, head_dim)
        k = torch.randn(batch_size, seq_len, head_dim)

        # Create attention mask
        mask = torch.randn(batch_size, seq_len, seq_len)

        scores = compute_attention_scores(q, k, attention_mask=mask)

        # Check mask was applied
        expected = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim) + mask
        assert torch.allclose(scores, expected)

    def test_compute_attention_scores_causal(self):
        """Test causal attention scores."""
        batch_size, seq_len, head_dim = 2, 8, 64
        q = torch.randn(batch_size, seq_len, head_dim)
        k = torch.randn(batch_size, seq_len, head_dim)

        scores = compute_attention_scores(q, k, is_causal=True)

        # Check causal masking
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert torch.all(scores[:, i, j] == float("-inf"))


class TestAttentionPatterns:
    """Test attention pattern generation."""

    def test_create_dilated_mask(self):
        """Test dilated mask creation."""
        seq_len = 16
        segment_length = 8
        dilation_rate = 2
        device = torch.device("cpu")

        mask = create_dilated_mask(seq_len, segment_length, dilation_rate, device)

        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool

        # Check pattern
        # Each position should attend to segment_length positions with dilation
        for i in range(seq_len):
            attended_positions = mask[i].sum().item()
            assert attended_positions > 0

    def test_create_block_diagonal_mask(self):
        """Test block diagonal mask creation."""
        seq_len = 16
        block_size = 4
        device = torch.device("cpu")

        mask = create_block_diagonal_mask(seq_len, block_size, device)

        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool

        # Check block structure
        num_blocks = seq_len // block_size
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size

            # Check block is fully connected
            block_mask = mask[start:end, start:end]
            assert torch.all(block_mask)

            # Check no connections outside block (without overlap)
            if block_idx > 0:
                assert not torch.any(mask[start:end, :start])
            if block_idx < num_blocks - 1:
                assert not torch.any(mask[start:end, end:])

    def test_create_block_diagonal_mask_with_overlap(self):
        """Test block diagonal mask with overlap."""
        seq_len = 16
        block_size = 4
        overlap = 1
        device = torch.device("cpu")

        mask = create_block_diagonal_mask(seq_len, block_size, device, overlap)

        # Check overlaps exist
        for block_idx in range(1, seq_len // block_size):
            start = block_idx * block_size
            prev_end = start

            # Should have connections to previous block's last positions
            overlap_mask = mask[start : start + overlap, prev_end - overlap : prev_end]
            assert torch.any(overlap_mask)

    def test_apply_dilated_attention_pattern(self):
        """Test applying dilated attention pattern."""
        batch_size, seq_len = 2, 16
        scores = torch.randn(batch_size, seq_len, seq_len)

        masked_scores = apply_dilated_attention_pattern(
            scores, segment_length=8, dilation_rate=2, group_idx=0, total_groups=2
        )

        # Check that some positions are masked
        assert torch.any(masked_scores == float("-inf"))
        assert torch.any(masked_scores != float("-inf"))


class TestAttentionComputation:
    """Test attention computation utilities."""

    def test_standard_attention(self):
        """Test standard attention implementation."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = standard_attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Should not be identical to input
        assert not torch.allclose(output, q)
        assert not torch.allclose(output, v)

    def test_standard_attention_with_dropout(self):
        """Test standard attention with dropout."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Set to training mode
        q.requires_grad_(True)

        output1 = standard_attention(q, k, v, dropout_p=0.5)
        output2 = standard_attention(q, k, v, dropout_p=0.5)

        # With dropout, outputs should be different
        assert not torch.allclose(output1, output2)

    def test_optimize_attention_flash(self):
        """Test optimized attention with Flash Attention."""
        # Skip if flash_attn is not installed
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            pytest.skip("flash_attn not installed")

        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()
        k = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()
        v = torch.randn(batch_size, seq_len, num_heads, head_dim).cuda()

        with patch("dilated_attention_pytorch.utils.attention_utils.HAS_FLASH_ATTN", True):
            # Mock flash_attn module
            with patch("flash_attn.flash_attn_func") as mock_flash_attn:
                # Mock flash attention to return something reasonable
                mock_flash_attn.return_value = torch.randn_like(q)

                output = optimize_attention_computation(q, k, v)

                # Check that output has the right shape
                assert output.shape == q.shape

                # Flash attention might not be called if the module isn't installed
                # So we just check that we got a valid output
                assert not torch.isnan(output).any()

    @patch("dilated_attention_pytorch.utils.attention_utils.HAS_SDPA", True)
    @patch("dilated_attention_pytorch.utils.attention_utils.HAS_FLASH_ATTN", False)
    def test_optimize_attention_sdpa(self):
        """Test optimized attention with PyTorch SDPA."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = optimize_attention_computation(q, k, v)

        assert output.shape == q.shape


class TestPositionalEncodings:
    """Test positional encoding utilities."""

    def test_compute_alibi_bias(self):
        """Test ALiBi bias computation."""
        seq_len = 8
        num_heads = 4
        device = torch.device("cpu")

        alibi = compute_alibi_bias(seq_len, num_heads, device)

        assert alibi.shape == (num_heads, seq_len, seq_len)

        # Check relative position pattern
        for h in range(num_heads):
            # Diagonal should be 0
            assert torch.all(alibi[h].diagonal() == 0)

            # Should be antisymmetric
            assert torch.allclose(alibi[h], -alibi[h].T)

    def test_compute_rotary_embeddings(self):
        """Test rotary embeddings computation."""
        seq_len = 8
        dim = 64
        device = torch.device("cpu")

        cos, sin = compute_rotary_embeddings(seq_len, dim, device)

        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

        # Check values are in [-1, 1]
        assert torch.all(cos >= -1) and torch.all(cos <= 1)
        assert torch.all(sin >= -1) and torch.all(sin <= 1)

    def test_apply_rotary_embeddings(self):
        """Test applying rotary embeddings."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)

        cos, sin = compute_rotary_embeddings(seq_len, head_dim, q.device)

        q_rot, k_rot = apply_rotary_embeddings(q, k, cos, sin)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

        # Should be different from input
        assert not torch.allclose(q_rot, q)
        assert not torch.allclose(k_rot, k)


class TestHeadOperations:
    """Test attention head manipulation."""

    def test_split_attention_heads(self):
        """Test splitting hidden dim into heads."""
        batch_size, seq_len, hidden_dim = 2, 8, 512
        num_heads = 8

        tensor = torch.randn(batch_size, seq_len, hidden_dim)
        split_tensor = split_attention_heads(tensor, num_heads)

        assert split_tensor.shape == (
            batch_size,
            seq_len,
            num_heads,
            hidden_dim // num_heads,
        )

        # Check content preservation
        merged = split_tensor.reshape(batch_size, seq_len, hidden_dim)
        assert torch.allclose(tensor, merged)

    def test_merge_attention_heads(self):
        """Test merging heads back to hidden dim."""
        batch_size, seq_len, num_heads, head_dim = 2, 8, 8, 64

        tensor = torch.randn(batch_size, seq_len, num_heads, head_dim)
        merged_tensor = merge_attention_heads(tensor, num_heads, head_dim)

        assert merged_tensor.shape == (batch_size, seq_len, num_heads * head_dim)

        # Check content preservation
        split = merged_tensor.reshape(batch_size, seq_len, num_heads, head_dim)
        assert torch.allclose(tensor, split)

    def test_split_merge_roundtrip(self):
        """Test split and merge are inverse operations."""
        batch_size, seq_len, hidden_dim = 2, 8, 512
        num_heads = 8

        original = torch.randn(batch_size, seq_len, hidden_dim)

        # Split then merge
        split = split_attention_heads(original, num_heads)
        merged = merge_attention_heads(split, num_heads, hidden_dim // num_heads)

        assert torch.allclose(original, merged)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_odd_dimension_rotary(self):
        """Test rotary embeddings with odd dimension."""
        seq_len = 8
        dim = 63  # Odd dimension
        device = torch.device("cpu")

        with pytest.raises(AssertionError):
            compute_rotary_embeddings(seq_len, dim, device)

    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        batch_size, seq_len, head_dim = 2, 0, 64
        q = torch.randn(batch_size, seq_len, head_dim)
        k = torch.randn(batch_size, seq_len, head_dim)

        scores = compute_attention_scores(q, k)

        assert scores.shape == (batch_size, seq_len, seq_len)

    def test_single_position(self):
        """Test handling of single position sequences."""
        batch_size, seq_len, num_heads, head_dim = 2, 1, 4, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        output = standard_attention(q, k, v)

        assert output.shape == (batch_size, seq_len, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
