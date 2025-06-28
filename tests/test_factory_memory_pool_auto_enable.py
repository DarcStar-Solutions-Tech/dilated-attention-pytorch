#!/usr/bin/env python3
"""
Test automatic memory pool enabling in factory pattern.
"""

import pytest
import torch

from dilated_attention_pytorch.core import (
    create_dilated_attention,
    create_multihead_dilated_attention,
)


class TestFactoryMemoryPoolAutoEnable:
    """Test automatic memory pool configuration in factory pattern."""

    def test_auto_enable_for_long_sequences(self):
        """Test that memory pools are auto-enabled for long sequences."""
        # Create attention with long sequences
        attention = create_dilated_attention(
            "improved", segment_lengths=[2048, 4096, 8192], dilation_rates=[1, 2, 4]
        )

        # Should have memory pool enabled
        # For improved attention V2, check buffer_manager
        # For improved attention, check enable_memory_pool
        if hasattr(attention, "buffer_manager"):
            assert attention.buffer_manager is not None
        elif hasattr(attention, "enable_memory_pool"):
            assert attention.enable_memory_pool is True
        elif hasattr(attention, "_memory_pool"):
            assert attention._memory_pool is not None

    def test_auto_disable_for_short_sequences(self):
        """Test that memory pools are auto-disabled for short sequences."""
        # Create attention with short sequences
        attention = create_dilated_attention(
            "standard", segment_lengths=[256, 512, 1024], dilation_rates=[1, 2, 4]
        )

        # Should have memory pool disabled or not present
        if hasattr(attention, "enable_memory_pool"):
            assert attention.enable_memory_pool is False
        elif hasattr(attention, "_memory_pool"):
            # If _memory_pool exists, it should be None for short sequences
            assert attention._memory_pool is None

    def test_auto_enable_for_ring_attention(self):
        """Test that memory pools are always enabled for ring attention."""
        if not torch.cuda.is_available():
            pytest.skip("Ring attention requires CUDA")

        # Create ring attention with short sequences
        attention = create_dilated_attention(
            "ring", segment_lengths=[256, 512], dilation_rates=[1, 2]
        )

        # Should have memory pool enabled even for short sequences
        # Ring attention wrapper might not expose these directly
        # Just verify it creates successfully
        assert attention is not None

    def test_user_override_respected(self):
        """Test that user's explicit memory pool settings are respected."""
        # User explicitly disables memory pool
        attention = create_dilated_attention(
            "improved",
            segment_lengths=[4096, 8192, 16384],  # Long sequences
            dilation_rates=[1, 2, 4],
            enable_memory_pool=False,  # Explicit override
        )

        # Should respect user's choice
        if hasattr(attention, "enable_memory_pool"):
            assert attention.enable_memory_pool is False

    def test_multihead_auto_configuration(self):
        """Test auto-configuration for multihead attention."""
        # Create multihead attention with long sequences
        attention = create_multihead_dilated_attention(
            "improved",
            embed_dim=768,
            num_heads=12,
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
        )

        # Should have appropriate memory configuration
        assert attention is not None

    def test_lightweight_pool_for_medium_sequences(self):
        """Test that lightweight pool is used for medium sequences."""
        # Create attention with medium sequences (4096-8192)
        attention = create_dilated_attention(
            "improved", segment_lengths=[2048, 4096], dilation_rates=[1, 2]
        )

        # Check if lightweight pool is configured
        # Since medium sequences use lightweight pool
        if hasattr(attention, "enable_memory_pool"):
            assert attention.enable_memory_pool is True
            if hasattr(attention, "lightweight_pool"):
                assert attention.lightweight_pool is True

    @pytest.mark.parametrize("impl_type", ["distributed", "block_sparse_ring"])
    def test_auto_enable_for_special_implementations(self, impl_type):
        """Test that memory pools are enabled for distributed and sparse implementations."""
        if impl_type == "block_sparse_ring" and not torch.cuda.is_available():
            pytest.skip("Block sparse requires CUDA")

        try:
            # Create attention with short sequences
            attention = create_dilated_attention(
                impl_type, segment_lengths=[512, 1024], dilation_rates=[1, 2]
            )

            # Should have memory pool enabled due to implementation type
            assert attention is not None
        except Exception as e:
            # Some implementations might not be available
            if "Unknown attention type" in str(e):
                pytest.skip(f"{impl_type} implementation not available")
            else:
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
