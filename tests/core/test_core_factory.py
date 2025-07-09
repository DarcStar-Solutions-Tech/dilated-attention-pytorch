"""
Test suite for the factory pattern module.

Tests factory functions for creating dilated attention modules.
"""

from unittest.mock import patch

import pytest

from dilated_attention_pytorch.core import (
    BaseDilatedAttention,
    BaseMultiheadDilatedAttention,
    DilatedAttentionConfig,
    create_adaptive_sparse_attention,
    create_block_sparse_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
    register_attention,
    register_multihead_attention,
)
from dilated_attention_pytorch.core.factory import (
    _ATTENTION_REGISTRY,
    _MULTIHEAD_REGISTRY,
    _ensure_implementations_registered,
    _get_config_class,
    _select_best_attention_type,
)

# Ensure implementations are registered for tests
_ensure_implementations_registered()


# Mock implementations for testing
class MockDilatedAttention(BaseDilatedAttention):
    """Mock dilated attention for testing."""

    def forward(self, q, _k, _v, _is_causal=False, _attention_mask=None):
        return q  # Simple mock implementation


class MockMultiheadDilatedAttention(BaseMultiheadDilatedAttention):
    """Mock multihead dilated attention for testing."""

    def _create_attention_module(self):
        return MockDilatedAttention(self.attention_config)

    def _init_qkv_projections(self, factory_kwargs):
        pass  # Mock implementation

    def forward(self, query, _key=None, _value=None, **_kwargs):
        return query  # Simple mock implementation


class TestFactoryRegistration:
    """Test module registration functions."""

    def setup_method(self):
        """Clear registries before each test."""
        # Save current registry state
        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()
        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()

    def test_register_attention(self):
        """Test registering attention implementation."""
        register_attention("test_attention", MockDilatedAttention)

        assert "test_attention" in _ATTENTION_REGISTRY
        assert _ATTENTION_REGISTRY["test_attention"] is MockDilatedAttention

    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)

    def test_register_multihead_attention(self):
        """Test registering multihead attention implementation."""
        register_multihead_attention("multihead_test", MockMultiheadDilatedAttention)

        assert "multihead_test" in _MULTIHEAD_REGISTRY
        assert _MULTIHEAD_REGISTRY["multihead_test"] is MockMultiheadDilatedAttention


class TestFactoryCreation:
    """Test factory creation functions."""

    def setup_method(self):
        """Setup test implementations."""
        # Save current registry state
        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()
        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        # Register test implementations
        register_attention("test", MockDilatedAttention)
        register_attention("standard", MockDilatedAttention)
        register_attention("improved", MockDilatedAttention)
        register_multihead_attention("multihead_test", MockMultiheadDilatedAttention)
        register_multihead_attention(
            "multihead_standard", MockMultiheadDilatedAttention
        )
        register_multihead_attention(
            "multihead_improved", MockMultiheadDilatedAttention
        )

    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)

    def test_create_dilated_attention_basic(self):
        """Test basic dilated attention creation."""
        attention = create_dilated_attention(
            attention_type="test", segment_lengths=[1024, 2048], dilation_rates=[1, 2]
        )

        assert isinstance(attention, MockDilatedAttention)
        assert attention.segment_lengths == [1024, 2048]
        assert attention.dilation_rates == [1, 2]

    def test_create_dilated_attention_defaults(self):
        """Test dilated attention with default parameters."""
        attention = create_dilated_attention(attention_type="test")

        assert attention.segment_lengths == [2048, 4096, 8192]
        assert attention.dilation_rates == [1, 2, 4]

    def test_create_dilated_attention_invalid_type(self):
        """Test error on invalid attention type."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            create_dilated_attention(attention_type="invalid_type")

    def test_create_multihead_dilated_attention(self):
        """Test multihead dilated attention creation."""
        attention = create_multihead_dilated_attention(
            attention_type="test",
            embed_dim=768,
            num_heads=12,
            segment_lengths=[1024],
            dilation_rates=[1],
            dropout=0.1,
        )

        assert isinstance(attention, MockMultiheadDilatedAttention)
        assert attention.embed_dim == 768
        assert attention.num_heads == 12
        assert attention.attention_config.dropout == 0.1

    def test_create_multihead_config_separation(self):
        """Test separation of multihead and attention configs."""
        attention = create_multihead_dilated_attention(
            attention_type="test",
            embed_dim=768,
            num_heads=12,
            bias=False,  # Multihead config
            dropout=0.1,  # Attention config
            layer_norm=True,  # Multihead config
            use_tf32=False,  # Attention config
        )

        # Check multihead config
        assert not attention.bias
        assert hasattr(attention, "q_ln") or hasattr(attention, "layer_norm")

        # Check attention config
        assert attention.attention_config.dropout == 0.1
        assert not attention.attention_config.use_tf32


class TestSpecializedFactories:
    """Test specialized factory functions."""

    def setup_method(self):
        """Setup test implementations."""
        # Save current registry state
        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()
        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        # Register required implementations
        register_attention("block_sparse_ring", MockDilatedAttention)
        register_multihead_attention(
            "multihead_block_sparse_ring", MockMultiheadDilatedAttention
        )

    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)

    def test_create_block_sparse_attention(self):
        """Test block-sparse attention creation."""
        attention = create_block_sparse_attention(
            sparsity_ratio=0.9,
            pattern_type="dilated_sparse",
            embed_dim=768,
            num_heads=12,
        )

        assert isinstance(attention, MockMultiheadDilatedAttention)
        assert attention.attention_config.sparsity_ratio == 0.9
        assert attention.attention_config.pattern_type == "dilated_sparse"

    def test_create_adaptive_sparse_attention(self):
        """Test adaptive sparse attention creation."""
        attention = create_adaptive_sparse_attention(
            embed_dim=768, num_heads=12, min_sparsity=0.2, max_sparsity=0.8
        )

        assert isinstance(attention, MockMultiheadDilatedAttention)
        assert attention.attention_config.enable_adaptive
        assert attention.attention_config.min_sparsity == 0.2
        assert attention.attention_config.max_sparsity == 0.8
        assert attention.attention_config.pattern_type == "learned"


class TestAutoSelection:
    """Test automatic implementation selection."""

    def setup_method(self):
        """Setup test implementations."""
        # Save current registry state
        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()
        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        # Register implementations
        register_attention("standard", MockDilatedAttention)
        register_attention("improved", MockDilatedAttention)
        register_multihead_attention(
            "multihead_standard", MockMultiheadDilatedAttention
        )
        register_multihead_attention(
            "multihead_improved", MockMultiheadDilatedAttention
        )

    @patch("dilated_attention_pytorch.core.factory.GPU_TYPE", "h100")
    @patch("dilated_attention_pytorch.core.factory.HAS_FLASH_ATTN_3", True)
    def test_auto_select_h100_with_fa3(self):
        """Test auto-selection on H100 with Flash Attention 3."""
        attention_type = _select_best_attention_type()
        assert (
            attention_type == "improved"
        )  # H100 with FA3 uses improved for base attention

    @patch("dilated_attention_pytorch.core.factory.GPU_TYPE", "a100")
    @patch("dilated_attention_pytorch.core.factory.HAS_FLASH_ATTN", True)
    @patch("dilated_attention_pytorch.core.factory.HAS_FLASH_ATTN_3", False)
    def test_auto_select_a100_with_fa2(self):
        """Test auto-selection on A100 with Flash Attention 2."""
        attention_type = _select_best_attention_type()
        assert attention_type == "improved"

    @patch("dilated_attention_pytorch.core.factory.GPU_TYPE", "v100")
    def test_auto_select_v100(self):
        """Test auto-selection on V100."""
        attention_type = _select_best_attention_type()
        assert attention_type == "improved"

    @patch("dilated_attention_pytorch.core.factory.GPU_TYPE", "cpu")
    def test_auto_select_cpu(self):
        """Test auto-selection on CPU."""
        attention_type = _select_best_attention_type()
        assert attention_type == "improved"

    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)

    def test_create_with_auto(self):
        """Test creation with auto type."""
        # For auto-selection, we need real implementations registered
        # Clear mock implementations first
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()

        # Force re-registration of real implementations
        import dilated_attention_pytorch.core.factory as factory_module

        factory_module._implementations_registered = False

        # Ensure real implementations are registered
        _ensure_implementations_registered()

        try:
            attention = create_dilated_attention(attention_type="auto")
            # Check that we got a valid attention module (not checking specific type since it's auto-selected)
            assert hasattr(attention, "forward")
            assert hasattr(attention, "segment_lengths")
            assert hasattr(attention, "dilation_rates")
        finally:
            # Restore mock registry
            _ATTENTION_REGISTRY.clear()
            _MULTIHEAD_REGISTRY.clear()
            _ATTENTION_REGISTRY.update(self._saved_attention_registry)
            _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)


class TestConfigSelection:
    """Test configuration class selection."""

    def test_get_config_class_standard(self):
        """Test config class for standard attention."""
        config_class = _get_config_class("standard")
        assert config_class is DilatedAttentionConfig

    def test_get_config_class_ring(self):
        """Test config class for ring attention."""
        from dilated_attention_pytorch.core.config import RingAttentionConfig

        config_class = _get_config_class("ring")
        assert config_class is RingAttentionConfig

    def test_get_config_class_sparse(self):
        """Test config class for sparse attention."""
        from dilated_attention_pytorch.core.config import SparseAttentionConfig

        config_class = _get_config_class("block_sparse_ring")
        assert config_class is SparseAttentionConfig

    def test_get_config_class_default(self):
        """Test default config class for unknown type."""
        config_class = _get_config_class("unknown_type")
        assert config_class is DilatedAttentionConfig


class TestFactoryEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Setup minimal registry."""
        # Save current registry state
        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()
        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        register_attention("test", MockDilatedAttention)
        register_multihead_attention("multihead_test", MockMultiheadDilatedAttention)

    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)

    def test_create_with_extra_kwargs(self):
        """Test creation with extra keyword arguments."""
        attention = create_dilated_attention(
            attention_type="test",
            segment_lengths=[1024],
            dilation_rates=[1],
            custom_param=42,
            another_param="test",
        )

        # Should not raise error
        assert isinstance(attention, MockDilatedAttention)

    def test_empty_registry(self):
        """Test error when registry is empty."""
        _ATTENTION_REGISTRY.clear()

        with pytest.raises(ValueError, match="Unknown attention type"):
            create_dilated_attention(attention_type="test")

    def test_mismatched_multihead_type(self):
        """Test error when multihead type doesn't exist."""
        _MULTIHEAD_REGISTRY.clear()

        with pytest.raises(ValueError, match="Unknown attention type"):
            create_multihead_dilated_attention(attention_type="test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
