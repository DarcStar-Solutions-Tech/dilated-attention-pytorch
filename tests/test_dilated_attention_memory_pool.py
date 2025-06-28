#!/usr/bin/env python3
"""
Test memory pool integration with DilatedAttention.

This module tests that the enhanced memory pool is correctly integrated
with the core DilatedAttention implementation.
"""

import pytest
import torch

from dilated_attention_pytorch.dilated_attention import DilatedAttention


class TestDilatedAttentionMemoryPool:
    """Test memory pool integration with DilatedAttention."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return {
            "segment_lengths": [1024, 2048, 4096],
            "dilation_rates": [1, 2, 4],
            "batch_size": 2,
            "seq_len": 4096,
            "num_heads": 8,
            "head_dim": 64,
        }

    def test_memory_pool_initialization(self, test_config):
        """Test that memory pool initializes correctly."""
        # With memory pool enabled (default)
        attention_with_pool = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
        )
        assert attention_with_pool.enable_memory_pool is True
        assert attention_with_pool._memory_pool is not None

        # With memory pool disabled
        attention_no_pool = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=False,
        )
        assert attention_no_pool.enable_memory_pool is False
        assert attention_no_pool._memory_pool is None

    def test_lightweight_vs_full_pool(self, test_config):
        """Test lightweight vs full memory pool configurations."""
        # Lightweight pool
        attention_light = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=True,
        )
        assert attention_light.lightweight_pool is True
        assert attention_light._memory_pool is not None

        # Full pool
        attention_full = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=False,
        )
        assert attention_full.lightweight_pool is False
        assert attention_full._memory_pool is not None

    def test_forward_with_memory_pool(self, test_config, device):
        """Test forward pass with memory pool enabled."""
        attention = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=True,
        )

        # Create test tensors
        query = torch.randn(
            test_config["batch_size"],
            test_config["seq_len"],
            test_config["num_heads"],
            test_config["head_dim"],
            device=device,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # Forward pass
        output = attention(query, key, value)
        assert output.shape == query.shape
        assert output.device == query.device
        assert output.dtype == query.dtype

    def test_output_consistency(self, test_config, device):
        """Test that outputs are consistent with and without memory pool."""
        # Create models
        attention_no_pool = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=False,
        )

        attention_with_pool = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=True,
        )

        # Create test tensors
        torch.manual_seed(42)
        query = torch.randn(
            test_config["batch_size"],
            test_config["seq_len"],
            test_config["num_heads"],
            test_config["head_dim"],
            device=device,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # Get outputs
        with torch.no_grad():
            output_no_pool = attention_no_pool(query, key, value)
            output_with_pool = attention_with_pool(query, key, value)

        # Check outputs are close (allowing for minor numerical differences)
        torch.testing.assert_close(
            output_no_pool, output_with_pool, rtol=1e-5, atol=1e-5
        )

    def test_memory_pool_profiling(self, test_config, device):
        """Test memory pool with profiling enabled."""
        attention = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            enable_profiling=True,
            lightweight_pool=False,  # Use full pool for profiling
        )

        # Create test tensors
        query = torch.randn(
            test_config["batch_size"],
            test_config["seq_len"],
            test_config["num_heads"],
            test_config["head_dim"],
            device=device,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # Forward pass
        output = attention(query, key, value)
        assert output.shape == query.shape

        # Check that memory pool has stats when profiling is enabled
        if attention._memory_pool:
            stats = attention._memory_pool.get_stats()
            assert "enhanced_pool" in stats
            assert stats["enhanced_pool"]["bucketed_allocations"] > 0

    @pytest.mark.parametrize("seq_len", [4096, 8192, 16384])
    def test_different_sequence_lengths(self, test_config, device, seq_len):
        """Test memory pool with different sequence lengths."""
        # Skip if sequence length is not divisible by largest segment
        if seq_len % max(test_config["segment_lengths"]) != 0:
            pytest.skip(f"Sequence length {seq_len} not divisible by segment length")

        attention = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=True,
        )

        # Create test tensors
        query = torch.randn(
            test_config["batch_size"],
            seq_len,
            test_config["num_heads"],
            test_config["head_dim"],
            device=device,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # Forward pass
        output = attention(query, key, value)
        assert output.shape == query.shape

    def test_causal_attention_with_pool(self, test_config, device):
        """Test causal attention with memory pool."""
        attention = DilatedAttention(
            segment_lengths=test_config["segment_lengths"],
            dilation_rates=test_config["dilation_rates"],
            enable_memory_pool=True,
            lightweight_pool=True,
        )

        # Create test tensors
        query = torch.randn(
            test_config["batch_size"],
            test_config["seq_len"],
            test_config["num_heads"],
            test_config["head_dim"],
            device=device,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        # Forward pass with causal masking
        output = attention(query, key, value, is_causal=True)
        assert output.shape == query.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
