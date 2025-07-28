#!/usr/bin/env python3
"""
Tests for HilbertAttentionTritonWrapper.

This wrapper provides a q,k,v interface to the HilbertAttentionCore module.
Tests focus on the wrapper functionality rather than the underlying Triton kernels.
"""

import pytest
import torch

# Check if we can import the module
try:
    from dilated_attention_pytorch.kernels.hilbert_attention_triton_wrapper import (
        HilbertAttentionTritonWrapper,
        HilbertAttentionTritonFixed,
    )

    HAS_MODULE = True
except ImportError:
    HAS_MODULE = False


@pytest.mark.skipif(
    not HAS_MODULE, reason="HilbertAttentionTritonWrapper not available"
)
class TestHilbertAttentionTritonWrapper:
    """Test suite for the Hilbert attention wrapper."""

    @pytest.fixture
    def device(self):
        """Get test device (CPU for wrapper tests)."""
        # Use CPU to avoid Triton kernel issues
        return torch.device("cpu")

    def test_initialization(self, device):
        """Test wrapper initialization with various parameters."""
        # Basic initialization
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            dropout=0.1,
            num_heads=8,
            head_dim=64,
        ).to(device)

        assert wrapper.num_heads == 8
        assert wrapper.head_dim == 64
        assert wrapper.attention.segment_size == 64  # Uses first segment length
        assert wrapper.attention.dilation_rate == 1  # Uses first dilation rate

        # Check projection layers
        hidden_dim = 8 * 64  # num_heads * head_dim
        assert wrapper.q_proj.in_features == hidden_dim
        assert wrapper.q_proj.out_features == hidden_dim
        assert wrapper.k_proj.in_features == hidden_dim
        assert wrapper.v_proj.in_features == hidden_dim
        assert wrapper.out_proj.in_features == hidden_dim

    def test_forward_shape(self, device):
        """Test forward pass produces correct output shape."""
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[32],
            dilation_rates=[1],
            dropout=0.0,
            num_heads=4,
            head_dim=16,
        ).to(device)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 16

        # Create input tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward pass
        with torch.no_grad():  # Avoid potential gradient issues
            output = wrapper(q, k, v, is_causal=False)

        # Check output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_gradient_flow(self, device):
        """Test that gradients flow through the wrapper."""
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[16],
            dilation_rates=[1],
            dropout=0.0,
            num_heads=2,
            head_dim=8,
        ).to(device)

        # Small inputs for CPU testing
        q = torch.randn(1, 16, 2, 8, device=device, requires_grad=True)
        k = torch.randn(1, 16, 2, 8, device=device, requires_grad=True)
        v = torch.randn(1, 16, 2, 8, device=device, requires_grad=True)

        # Forward pass
        output = wrapper(q, k, v)
        loss = output.mean()

        # Check that we can compute gradients
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        # Basic sanity check on gradients
        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()

    def test_different_segment_configs(self, device):
        """Test wrapper with different segment length and dilation rate configurations."""
        configs = [
            ([32], [1]),
            ([64, 128], [1, 2]),
            ([128, 256, 512], [1, 2, 4]),
        ]

        for segment_lengths, dilation_rates in configs:
            wrapper = HilbertAttentionTritonWrapper(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                num_heads=4,
                head_dim=16,
            ).to(device)

            # Verify it uses the first values
            assert wrapper.attention.segment_size == segment_lengths[0]
            assert wrapper.attention.dilation_rate == dilation_rates[0]

    def test_backward_compatibility_alias(self, device):
        """Test that HilbertAttentionTritonFixed alias works."""
        # Should be able to instantiate the alias
        wrapper = HilbertAttentionTritonFixed(
            segment_lengths=[32], dilation_rates=[1], num_heads=4, head_dim=16
        ).to(device)

        # Test forward pass
        q = torch.randn(1, 32, 4, 16, device=device)
        k = torch.randn(1, 32, 4, 16, device=device)
        v = torch.randn(1, 32, 4, 16, device=device)

        with torch.no_grad():
            output = wrapper(q, k, v)

        assert output.shape == (1, 32, 4, 16)

    def test_parameter_count(self, device):
        """Test the number of parameters in the wrapper."""
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[64], dilation_rates=[1], num_heads=8, head_dim=64
        ).to(device)

        # Count parameters
        param_count = sum(p.numel() for p in wrapper.parameters())

        # Expected parameters:
        # - attention.qkv_proj: hidden_dim x (3 * hidden_dim)
        # - attention.out_proj: hidden_dim x hidden_dim
        # - wrapper.q_proj: hidden_dim x hidden_dim
        # - wrapper.k_proj: hidden_dim x hidden_dim
        # - wrapper.v_proj: hidden_dim x hidden_dim
        # - wrapper.out_proj: hidden_dim x hidden_dim

        hidden_dim = 8 * 64  # 512
        expected_params = (
            hidden_dim * 3 * hidden_dim  # attention.qkv_proj
            + hidden_dim * hidden_dim  # attention.out_proj
            + hidden_dim * hidden_dim * 4  # wrapper projections
        )

        assert param_count == expected_params

    def test_zero_dropout(self, device):
        """Test that zero dropout produces deterministic results."""
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[32],
            dilation_rates=[1],
            dropout=0.0,
            num_heads=4,
            head_dim=16,
        ).to(device)

        # Set to eval mode to ensure dropout is disabled
        wrapper.eval()

        # Fixed input
        torch.manual_seed(42)
        q = torch.randn(1, 32, 4, 16, device=device)
        k = torch.randn(1, 32, 4, 16, device=device)
        v = torch.randn(1, 32, 4, 16, device=device)

        # Multiple forward passes
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                out = wrapper(q, k, v)
                outputs.append(out)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_batch_and_sequence_dimensions(self, device, batch_size, seq_len):
        """Test wrapper with different batch sizes and sequence lengths."""
        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[16],  # Small segment for CPU
            dilation_rates=[1],
            num_heads=2,
            head_dim=8,
        ).to(device)

        q = torch.randn(batch_size, seq_len, 2, 8, device=device)
        k = torch.randn(batch_size, seq_len, 2, 8, device=device)
        v = torch.randn(batch_size, seq_len, 2, 8, device=device)

        with torch.no_grad():
            output = wrapper(q, k, v)

        assert output.shape == (batch_size, seq_len, 2, 8)
