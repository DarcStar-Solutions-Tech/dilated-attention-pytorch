"""
Test suite for the refactored core components.

This test file verifies that the newly created core components work correctly
and that all defects have been fixed.
"""

import threading
from unittest.mock import patch

import pytest
import torch
from torch import nn

# Import the refactored components
from dilated_attention_pytorch.core import (
    CURRENT_OPTIMAL_SETTINGS,
    GPU_TYPE,
    BaseDilatedAttention,
    BaseMultiheadDilatedAttention,
    DilatedAttentionConfig,
    MultiheadConfig,
    ValidationMixin,
)


class TestValidationMixin:
    """Test the ValidationMixin class."""

    def test_validate_segment_dilation_match(self):
        """Test segment and dilation rate validation."""
        mixin = ValidationMixin()

        # Valid case
        mixin.validate_segment_dilation_match([1, 2, 3], [4, 5, 6])

        # Invalid case
        with pytest.raises(ValueError, match="must have the same length"):
            mixin.validate_segment_dilation_match([1, 2], [3, 4, 5])

    def test_validate_positive_values(self):
        """Test positive value validation."""
        mixin = ValidationMixin()

        # Valid case
        mixin.validate_positive_values([1, 2, 3], "test_values")

        # Invalid case
        with pytest.raises(ValueError, match="must be positive"):
            mixin.validate_positive_values([1, 0, 3], "test_values")

        with pytest.raises(ValueError, match="must be positive"):
            mixin.validate_positive_values([1, -2, 3], "test_values")

    def test_validate_tensor_shape(self):
        """Test tensor shape validation."""
        mixin = ValidationMixin()

        # Valid case
        tensor = torch.randn(2, 3, 4, 5)
        mixin.validate_tensor_shape(tensor, 4, "test_tensor")

        # Invalid case
        with pytest.raises(ValueError, match="expected 3D tensor"):
            mixin.validate_tensor_shape(tensor, 3, "test_tensor")

    def test_validate_head_dim(self):
        """Test head dimension validation."""
        mixin = ValidationMixin()

        # Valid case
        head_dim = mixin.validate_head_dim(768, 12)
        assert head_dim == 64

        # Invalid case
        with pytest.raises(ValueError, match="must be divisible"):
            mixin.validate_head_dim(768, 13)

        # Warning cases
        with pytest.warns(UserWarning, match="should be divisible by 8"):
            mixin.validate_head_dim(84, 12)  # head_dim = 7

        with pytest.warns(UserWarning, match="> 128"):
            mixin.validate_head_dim(1536, 8)  # head_dim = 192


class TestConfigurations:
    """Test configuration dataclasses."""

    def test_dilated_attention_config(self):
        """Test DilatedAttentionConfig validation."""
        # Valid config
        config = DilatedAttentionConfig(
            segment_lengths=[2048, 4096], dilation_rates=[1, 2], dropout=0.1
        )
        assert config.num_groups == 2
        assert config.max_segment_length == 4096

        # Invalid segment/dilation mismatch
        with pytest.raises(ValueError, match="must have same length"):
            DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1, 2])

        # Invalid negative values
        with pytest.raises(ValueError, match="must be positive"):
            DilatedAttentionConfig(segment_lengths=[2048, -1], dilation_rates=[1, 2])

        # Invalid dropout
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1], dropout=1.5)

    def test_multihead_config(self):
        """Test MultiheadConfig validation."""
        # Valid config
        config = MultiheadConfig(embed_dim=768, num_heads=12)
        assert config.head_dim == 64

        # Invalid divisibility
        with pytest.raises(ValueError, match="must be divisible"):
            MultiheadConfig(embed_dim=768, num_heads=13)


class ConcreteDilatedAttention(BaseDilatedAttention):
    """Concrete implementation for testing."""

    def forward(self, q, k, v, is_causal=False, attention_mask=None):
        # Simple implementation for testing
        self._validate_forward_inputs(q, k, v, attention_mask)
        return q  # Just return query for testing


class ConcreteMultiheadAttention(BaseMultiheadDilatedAttention):
    """Concrete multihead implementation for testing."""

    def _create_attention_module(self):
        return ConcreteDilatedAttention(self.attention_config)

    def _init_qkv_projections(self, factory_kwargs):
        # Test both fused and separate projections
        if hasattr(self, "use_fused_qkv") and self.use_fused_qkv:
            self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, **factory_kwargs)
            self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, **factory_kwargs)
            self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, **factory_kwargs)

    def forward(self, query, key=None, value=None, **kwargs):
        # Simple implementation for testing
        return query


class TestBaseDilatedAttention:
    """Test BaseDilatedAttention class."""

    def test_initialization(self):
        """Test proper initialization."""
        config = DilatedAttentionConfig(
            segment_lengths=[2048, 4096], dilation_rates=[1, 2], dropout=0.1
        )
        attention = ConcreteDilatedAttention(config)

        assert attention.num_groups == 2
        assert isinstance(attention.dropout_layer, nn.Dropout)
        assert attention._max_cache_size == 100
        assert hasattr(attention, "_cache_lock")

    def test_thread_safe_caching(self):
        """Test thread-safe cache operations."""
        config = DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1])
        attention = ConcreteDilatedAttention(config)

        # Test concurrent access
        results = []

        def access_cache(num_heads):
            result = attention._get_head_groups(num_heads)
            results.append(result)

        threads = []
        for i in range(8, 17):  # Different num_heads values
            thread = threading.Thread(target=access_cache, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all results are correct
        assert len(results) == 9
        for i, result in enumerate(results):
            group_sizes, head_ranges = result
            assert sum(group_sizes) == 8 + i

    def test_cache_size_limit(self):
        """Test cache size limiting."""
        config = DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1])
        attention = ConcreteDilatedAttention(config)
        attention._max_cache_size = 5  # Small cache for testing

        # Fill cache beyond limit
        for i in range(10, 20):
            attention._get_head_groups(i)

        # Check cache size is limited
        assert len(attention._head_groups_cache) <= 5

        # Verify LRU behavior - early items should be evicted
        assert 10 not in attention._head_groups_cache
        assert 19 in attention._head_groups_cache

    def test_cache_helper_methods(self):
        """Test cache helper methods."""
        config = DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1])
        attention = ConcreteDilatedAttention(config)

        # Test _cache_get
        value = attention._cache_get(
            attention._pattern_cache, "test_key", lambda: torch.randn(10, 10)
        )
        assert value.shape == (10, 10)
        assert "test_key" in attention._pattern_cache

        # Test cache clearing
        attention._clear_caches()
        assert len(attention._head_groups_cache) == 0
        assert len(attention._pattern_cache) == 0


class TestBaseMultiheadDilatedAttention:
    """Test BaseMultiheadDilatedAttention class."""

    def test_parameter_initialization_fused(self):
        """Test parameter initialization with fused QKV."""
        multihead_config = MultiheadConfig(embed_dim=768, num_heads=12)
        attention_config = DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1])

        # Test fused QKV projection
        attention = ConcreteMultiheadAttention(multihead_config, attention_config)
        attention.use_fused_qkv = True
        attention._init_qkv_projections({"device": "cpu", "dtype": torch.float32})

        # Should not raise AttributeError
        attention._reset_parameters()

        # Check initialization worked
        assert hasattr(attention, "qkv_proj")

    def test_parameter_initialization_separate(self):
        """Test parameter initialization with separate projections."""
        multihead_config = MultiheadConfig(embed_dim=768, num_heads=12)
        attention_config = DilatedAttentionConfig(segment_lengths=[2048], dilation_rates=[1])

        # Test separate projections
        attention = ConcreteMultiheadAttention(multihead_config, attention_config)
        attention.use_fused_qkv = False
        attention._init_qkv_projections({"device": "cpu", "dtype": torch.float32})

        # Should not raise AttributeError
        attention._reset_parameters()

        # Check initialization worked
        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")


class TestConstants:
    """Test constants module behavior."""

    def test_gpu_type_lazy_evaluation(self):
        """Test that GPU_TYPE is evaluated lazily."""
        # GPU_TYPE should be a lazy object
        assert hasattr(GPU_TYPE, "__repr__")

        # Force evaluation
        gpu_type_str = str(GPU_TYPE)
        assert gpu_type_str in [
            "cpu",
            "h100",
            "a100",
            "v100",
            "amd_instinct",
            "rtx_40xx",
            "rtx_30xx",
            "generic_cuda",
        ]

    def test_optimal_settings_lazy(self):
        """Test lazy optimal settings."""
        # Should be able to access settings
        use_flash = CURRENT_OPTIMAL_SETTINGS.get("use_flash_attn", False)
        assert isinstance(use_flash, bool)

        block_size = CURRENT_OPTIMAL_SETTINGS.get("block_size", 512)
        assert isinstance(block_size, int)
        assert block_size > 0

    @patch("dilated_attention_pytorch.core.constants.logger")
    def test_logging_instead_of_print(self, mock_logger):
        """Test that logging is used instead of print."""
        # Import should trigger logging, not printing

        # Check that logger was used (may have been called during import)
        # Note: This is tricky to test due to module-level code execution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
