#!/usr/bin/env python3
"""
Tests for HilbertAttentionSimple - the PyTorch-based implementation.

This implementation doesn't rely on Triton kernels and should work on all platforms.
"""

import pytest
import torch

from dilated_attention_pytorch.kernels.hilbert_attention_simple import (
    HilbertAttentionSimple,
    create_hilbert_mapping,
    create_hilbert_curve_2d,
)


class TestHilbertMapping:
    """Test Hilbert curve mapping functions."""

    def test_hilbert_curve_2d_small(self):
        """Test 2D Hilbert curve for small grids."""
        # Test 2x2 grid
        mapping = create_hilbert_curve_2d(2)
        assert mapping.shape == (4,)
        assert set(mapping.tolist()) == {0, 1, 2, 3}

        # Test 4x4 grid
        mapping = create_hilbert_curve_2d(4)
        assert mapping.shape == (16,)
        assert set(mapping.tolist()) == set(range(16))

    def test_hilbert_mapping_simple(self):
        """Test Hilbert mapping for various sequence lengths."""
        # Small sequences should return identity
        mapping = create_hilbert_mapping(16)
        assert torch.equal(mapping, torch.arange(16, dtype=torch.long))

        # Medium sequences
        mapping = create_hilbert_mapping(128)
        assert mapping.shape == (128,)
        assert set(mapping.tolist()) == set(range(128))

        # Large sequences (snake pattern)
        mapping = create_hilbert_mapping(1024)
        assert mapping.shape == (1024,)
        assert set(mapping.tolist()) == set(range(1024))

    def test_hilbert_mapping_properties(self):
        """Test properties of Hilbert mapping."""
        for seq_len in [100, 256, 500]:
            mapping = create_hilbert_mapping(seq_len)

            # All indices should be present exactly once
            assert len(mapping) == seq_len
            assert len(set(mapping.tolist())) == seq_len

            # Mapping should be within valid range
            assert mapping.min() >= 0
            assert mapping.max() < seq_len


class TestHilbertAttentionSimple:
    """Test HilbertAttentionSimple module."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def attention_module(self, device):
        """Create test attention module."""
        return HilbertAttentionSimple(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
            dilation_rate=1,
            dropout=0.0,
            use_hilbert=True,
        ).to(device)

    def test_initialization(self, device):
        """Test module initialization."""
        attn = HilbertAttentionSimple(
            hidden_dim=512,
            num_heads=8,
            segment_size=128,
            dilation_rate=2,
            dropout=0.1,
            use_hilbert=True,
        ).to(device)

        assert attn.hidden_dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64
        assert attn.segment_size == 128
        assert attn.dilation_rate == 2
        assert attn.use_hilbert is True
        assert abs(attn.scale - 0.125) < 1e-6  # 1/sqrt(64)

    def test_forward_shape(self, attention_module, device):
        """Test forward pass shapes."""
        batch_sizes = [1, 2, 4]
        seq_lens = [64, 128, 256]

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                x = torch.randn(batch_size, seq_len, 256, device=device)

                # Test with Hilbert ordering
                output = attention_module(x, is_causal=False)
                assert output.shape == (batch_size, seq_len, 256)

                # Test without Hilbert ordering
                attention_module.use_hilbert = False
                output_no_hilbert = attention_module(x, is_causal=False)
                assert output_no_hilbert.shape == (batch_size, seq_len, 256)
                attention_module.use_hilbert = True

    def test_forward_with_padding(self, attention_module, device):
        """Test forward pass with sequences requiring padding."""
        # Sequence not divisible by segment_size (64)
        x = torch.randn(2, 100, 256, device=device)
        output = attention_module(x, is_causal=False)

        # Output should preserve original sequence length
        assert output.shape == (2, 100, 256)
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, attention_module, device):
        """Test gradient flow through the module."""
        x = torch.randn(2, 128, 256, device=device, requires_grad=True)
        output = attention_module(x, is_causal=False)
        loss = output.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Check parameter gradients
        for name, param in attention_module.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in {name} grad"
                assert not torch.isinf(param.grad).any(), f"Inf in {name} grad"

    def test_causal_masking(self, device):
        """Test causal masking behavior."""
        attn = HilbertAttentionSimple(
            hidden_dim=64,
            num_heads=1,
            segment_size=8,
            dilation_rate=1,
            dropout=0.0,
            use_hilbert=False,  # Disable Hilbert for easier interpretation
        ).to(device)

        # Create input where tokens have different values
        # This makes the effect of causal masking visible
        x = torch.randn(1, 8, 64, device=device)

        # Without causal masking
        output_non_causal = attn(x, is_causal=False)

        # With causal masking
        output_causal = attn(x, is_causal=True)

        # The outputs should be different due to causal masking
        assert not torch.allclose(output_non_causal, output_causal, atol=1e-5)

    def test_dilation_behavior(self, device):
        """Test attention with different dilation rates."""
        for dilation_rate in [1, 2, 4]:
            attn = HilbertAttentionSimple(
                hidden_dim=256,
                num_heads=8,
                segment_size=64,
                dilation_rate=dilation_rate,
                dropout=0.0,
                use_hilbert=True,
            ).to(device)

            x = torch.randn(2, 128, 256, device=device)
            output = attn(x, is_causal=False)

            assert output.shape == (2, 128, 256)
            assert not torch.isnan(output).any()

    def test_hilbert_cache(self, attention_module, device):
        """Test Hilbert mapping cache functionality."""
        # First call creates mapping
        mapping1 = attention_module.get_hilbert_mapping(128, device)
        assert 128 in attention_module._hilbert_cache

        # Second call returns cached mapping
        mapping2 = attention_module.get_hilbert_mapping(128, device)
        assert mapping1 is mapping2

        # Different size creates new mapping
        mapping3 = attention_module.get_hilbert_mapping(256, device)
        assert 256 in attention_module._hilbert_cache
        assert mapping3.shape[0] != mapping1.shape[0]

    def test_deterministic_behavior(self, attention_module, device):
        """Test deterministic forward pass."""
        attention_module.eval()  # Set to eval mode

        torch.manual_seed(42)
        x = torch.randn(2, 128, 256, device=device)

        # Run multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                out = attention_module(x, is_causal=False)
                outputs.append(out)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype_support(self, device, dtype):
        """Test support for different data types."""
        if device.type == "cpu" and dtype == torch.float16:
            pytest.skip("CPU doesn't support float16 well")

        attn = (
            HilbertAttentionSimple(
                hidden_dim=128, num_heads=4, segment_size=32, use_hilbert=True
            )
            .to(device)
            .to(dtype)
        )

        x = torch.randn(1, 64, 128, device=device, dtype=dtype)
        output = attn(x, is_causal=False)

        assert output.dtype == dtype
        assert not torch.isnan(output).any()

    def test_hilbert_vs_standard_attention(self, device):
        """Test that Hilbert reordering produces valid attention output."""
        # Create two identical modules
        attn_hilbert = HilbertAttentionSimple(
            hidden_dim=128,
            num_heads=4,
            segment_size=32,
            dilation_rate=1,
            dropout=0.0,
            use_hilbert=True,
        ).to(device)

        attn_standard = HilbertAttentionSimple(
            hidden_dim=128,
            num_heads=4,
            segment_size=32,
            dilation_rate=1,
            dropout=0.0,
            use_hilbert=False,
        ).to(device)

        # Copy weights
        attn_standard.load_state_dict(attn_hilbert.state_dict())

        # Test input
        x = torch.randn(2, 64, 128, device=device)

        with torch.no_grad():
            out_hilbert = attn_hilbert(x)
            out_standard = attn_standard(x)

        # Outputs may differ due to reordering, but should have similar statistics
        assert out_hilbert.shape == out_standard.shape
        assert abs(out_hilbert.mean() - out_standard.mean()) < 0.1
        assert abs(out_hilbert.std() - out_standard.std()) < 0.1
