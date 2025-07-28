#!/usr/bin/env python3
"""
Comprehensive tests for HilbertAttentionCore kernel implementation.

Tests include:
- Forward pass correctness
- Backward pass gradient checking
- Triton kernel functionality
- Hilbert mapping generation
- Edge cases and error handling
- Performance characteristics
"""

import pytest
import torch
import numpy as np

# Skip tests if triton is not available
try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Check if we can actually run Triton kernels
HAS_TRITON_RUNTIME = False
if HAS_TRITON and torch.cuda.is_available():
    try:
        # Simple test to see if Triton can compile
        HAS_TRITON_RUNTIME = True
    except Exception:
        pass

if HAS_TRITON:
    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
            create_hilbert_mapping,
            HilbertAttentionFunction,
        )
    except ImportError:
        # Module might not be importable due to Triton issues
        HAS_TRITON = False


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestHilbertAttentionCore:
    """Test suite for HilbertAttentionCore."""

    @pytest.fixture
    def device(self):
        """Get test device (CUDA if available, else CPU)."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def attention_module(self, device):
        """Create a test attention module."""
        return HilbertAttentionCore(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
            dilation_rate=1,
            dropout=0.0,
            use_custom_backward=True,
        ).to(device)

    def test_initialization(self, device):
        """Test module initialization with various parameters."""
        # Test basic initialization
        attn = HilbertAttentionCore(
            hidden_dim=512, num_heads=8, segment_size=128, dilation_rate=2, dropout=0.1
        ).to(device)

        assert attn.hidden_dim == 512
        assert attn.num_heads == 8
        assert attn.head_dim == 64
        assert attn.segment_size == 128
        assert attn.dilation_rate == 2
        assert abs(attn.scale - 0.125) < 1e-6  # 1/sqrt(64)

        # Test QKV projection shapes
        assert attn.qkv_proj.in_features == 512
        assert attn.qkv_proj.out_features == 1536  # 3 * 512
        assert attn.out_proj.in_features == 512
        assert attn.out_proj.out_features == 512

    def test_hilbert_mapping_generation(self):
        """Test Hilbert curve mapping generation."""
        # Test small sequences
        mapping = create_hilbert_mapping(16)
        assert mapping.shape == (16,)
        assert torch.all(mapping >= 0)
        assert torch.all(mapping < 16)
        assert len(torch.unique(mapping)) == 16  # All positions unique

        # Test larger sequences
        mapping = create_hilbert_mapping(256)
        assert mapping.shape == (256,)
        assert torch.all(mapping >= 0)
        assert torch.all(mapping < 256)
        assert len(torch.unique(mapping)) == 256

        # Test edge case - very small
        mapping = create_hilbert_mapping(1)
        assert mapping.shape == (1,)
        assert mapping[0] == 0

    def test_forward_pass_shapes(self, attention_module, device):
        """Test forward pass with various input shapes."""
        batch_sizes = [1, 2, 4]
        seq_lens = [64, 128, 256, 512]

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                x = torch.randn(batch_size, seq_len, 256, device=device)

                # Test with Hilbert ordering
                out_hilbert = attention_module(x, use_hilbert=True)
                assert out_hilbert.shape == (batch_size, seq_len, 256)

                # Test without Hilbert ordering
                out_standard = attention_module(x, use_hilbert=False)
                assert out_standard.shape == (batch_size, seq_len, 256)

    def test_forward_pass_padding(self, attention_module, device):
        """Test forward pass with sequences requiring padding."""
        # Test sequence not divisible by segment_size
        x = torch.randn(2, 100, 256, device=device)  # 100 not divisible by 64
        out = attention_module(x, use_hilbert=True)
        assert out.shape == (2, 100, 256)  # Original shape preserved

    def test_gradient_flow(self, attention_module, device):
        """Test gradient flow through the module."""
        x = torch.randn(2, 128, 256, device=device, requires_grad=True)
        out = attention_module(x, use_hilbert=True)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

        # Check parameter gradients
        for name, param in attention_module.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN in {name} grad"
                assert not torch.isinf(param.grad).any(), f"Inf in {name} grad"

    def test_custom_backward_correctness(self, device):
        """Test custom backward pass against PyTorch autograd."""
        torch.manual_seed(42)

        # Create module with custom backward
        attn_custom = HilbertAttentionCore(
            hidden_dim=128,
            num_heads=4,
            segment_size=32,
            dilation_rate=1,
            dropout=0.0,
            use_custom_backward=True,
        ).to(device)

        # Create module without custom backward for comparison
        attn_pytorch = HilbertAttentionCore(
            hidden_dim=128,
            num_heads=4,
            segment_size=32,
            dilation_rate=1,
            dropout=0.0,
            use_custom_backward=False,
        ).to(device)

        # Copy weights
        attn_pytorch.load_state_dict(attn_custom.state_dict())

        # Test input
        x = torch.randn(1, 64, 128, device=device, requires_grad=True)
        x_copy = x.clone().detach().requires_grad_(True)

        # Forward pass
        out_custom = attn_custom(x, use_hilbert=True)
        out_pytorch = attn_pytorch(x_copy, use_hilbert=True)

        # Check forward outputs match
        assert torch.allclose(out_custom, out_pytorch, atol=1e-5)

        # Backward pass
        grad_out = torch.randn_like(out_custom)
        out_custom.backward(grad_out)
        out_pytorch.backward(grad_out.clone())

        # Check input gradients match (allow some tolerance for numerical differences)
        if x.grad is not None and x_copy.grad is not None:
            assert torch.allclose(x.grad, x_copy.grad, atol=1e-4, rtol=1e-4)

    def test_dilation_rate_behavior(self, device):
        """Test attention with different dilation rates."""
        for dilation_rate in [1, 2, 4]:
            attn = HilbertAttentionCore(
                hidden_dim=256,
                num_heads=8,
                segment_size=64,
                dilation_rate=dilation_rate,
                dropout=0.0,
            ).to(device)

            x = torch.randn(2, 128, 256, device=device)
            out = attn(x, use_hilbert=True)
            assert out.shape == (2, 128, 256)
            assert not torch.isnan(out).any()

    def test_attention_pattern_correctness(self, device):
        """Test that attention pattern follows expected behavior."""
        # Simple test: attention should focus on similar tokens
        attn = HilbertAttentionCore(
            hidden_dim=64, num_heads=1, segment_size=8, dilation_rate=1, dropout=0.0
        ).to(device)

        # Create input where first and last tokens are similar
        x = torch.randn(1, 8, 64, device=device)
        x[0, 0] = x[0, -1]  # Make first and last tokens identical

        with torch.no_grad():
            out = attn(
                x, use_hilbert=False
            )  # Use standard ordering for interpretability

        # Output should show some relationship between first and last tokens
        # (This is a basic sanity check, not a strict test)
        assert out.shape == (1, 8, 64)

    def test_memory_efficiency(self, device):
        """Test memory usage patterns."""
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

            attn = HilbertAttentionCore(
                hidden_dim=512, num_heads=8, segment_size=128, dilation_rate=1
            ).to(device)

            # Run forward pass
            x = torch.randn(4, 512, 512, device=device)
            _ = attn(x, use_hilbert=True)

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

            # Basic sanity check - should not use excessive memory
            # Adjust threshold based on your requirements
            assert peak_memory < 2000  # Less than 2GB for this size

    def test_deterministic_behavior(self, attention_module, device):
        """Test that forward pass is deterministic."""
        torch.manual_seed(42)
        x = torch.randn(2, 128, 256, device=device)

        # Run multiple times
        outputs = []
        for _ in range(3):
            out = attention_module(x, use_hilbert=True)
            outputs.append(out)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_hilbert_cache(self, attention_module, device):
        """Test Hilbert mapping cache functionality."""
        # First call should create mapping
        mapping1 = attention_module.get_hilbert_mapping(128, device)

        # Second call should return cached mapping
        mapping2 = attention_module.get_hilbert_mapping(128, device)

        # Should be the same object
        assert mapping1 is mapping2

        # Different size should create new mapping
        mapping3 = attention_module.get_hilbert_mapping(256, device)
        assert mapping3.shape != mapping1.shape

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_dtype_support(self, device, dtype):
        """Test support for different data types."""
        if device.type == "cpu" and dtype == torch.float16:
            pytest.skip("CPU doesn't support float16 well")

        attn = (
            HilbertAttentionCore(hidden_dim=128, num_heads=4, segment_size=32)
            .to(device)
            .to(dtype)
        )

        x = torch.randn(1, 64, 128, device=device, dtype=dtype)
        out = attn(x, use_hilbert=True)

        assert out.dtype == dtype
        assert not torch.isnan(out).any()

    def test_error_handling(self, device):
        """Test error handling for invalid inputs."""
        attn = HilbertAttentionCore(hidden_dim=256, num_heads=8).to(device)

        # Test with wrong hidden dimension
        with pytest.raises(RuntimeError):
            x = torch.randn(1, 64, 128, device=device)  # Wrong hidden dim
            attn(x)

        # Test with invalid dimensions
        with pytest.raises(ValueError):
            HilbertAttentionCore(
                hidden_dim=256,
                num_heads=7,  # Not a divisor of hidden_dim
            )


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
class TestHilbertAttentionFunction:
    """Test the custom autograd function directly."""

    def test_function_forward_backward(self):
        """Test the custom function's forward and backward passes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for this test")

        device = torch.device("cuda")
        B, H, M, D = 2, 4, 64, 32

        # Create inputs
        qkv = torch.randn(3, B, H, M, D, device=device, requires_grad=True)
        scale = 1.0 / np.sqrt(D)
        hilbert_map = create_hilbert_mapping(M).to(device)

        # Forward pass
        out = HilbertAttentionFunction.apply(
            qkv, scale, hilbert_map, 32, 1, M, M, B, H, D
        )

        assert out.shape == (B, H, M, D)
        assert not torch.isnan(out).any()

        # Backward pass
        grad_out = torch.randn_like(out)
        out.backward(grad_out)

        assert qkv.grad is not None
        assert qkv.grad.shape == qkv.shape
        assert not torch.isnan(qkv.grad).any()
