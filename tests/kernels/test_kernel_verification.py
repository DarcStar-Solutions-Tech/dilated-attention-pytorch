#!/usr/bin/env python3
"""
Comprehensive verification tests for kernel implementations.

This module tests:
1. HilbertAttentionCore - Main Triton implementation
2. HilbertAttentionSimple - PyTorch fallback
3. HilbertAttentionTritonWrapper - Q,K,V interface wrapper

Tests cover:
- Forward pass correctness
- Backward pass/gradient flow
- Numerical stability
- Edge cases (padding, small dimensions)
- Memory efficiency
"""

import pytest
import torch

# Import all kernel implementations
try:
    from dilated_attention_pytorch.kernels import (
        HilbertAttentionCore,
        HilbertAttentionSimple,
        HilbertAttentionTritonWrapper,
        HilbertAttentionTritonFixed,
        TRITON_AVAILABLE,
        create_hilbert_mapping,
    )
except ImportError:
    # Fallback imports if package not installed
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
    from dilated_attention_pytorch.kernels import (
        HilbertAttentionCore,
        HilbertAttentionSimple,
        HilbertAttentionTritonWrapper,
        HilbertAttentionTritonFixed,
        TRITON_AVAILABLE,
        create_hilbert_mapping,
    )


def get_device():
    """Get appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class TestHilbertMapping:
    """Test Hilbert curve mapping functionality."""

    def test_mapping_basic(self):
        """Test basic Hilbert mapping properties."""
        seq_lengths = [32, 64, 128, 256, 512, 1024]

        for seq_len in seq_lengths:
            mapping = create_hilbert_mapping(seq_len)

            # Check mapping is a permutation
            assert mapping.shape == (seq_len,)
            assert torch.all(torch.sort(mapping)[0] == torch.arange(seq_len))

            # Check mapping is within bounds
            assert torch.all(mapping >= 0)
            assert torch.all(mapping < seq_len)

    def test_mapping_consistency(self):
        """Test that mapping is consistent across calls."""
        seq_len = 256
        mapping1 = create_hilbert_mapping(seq_len)
        mapping2 = create_hilbert_mapping(seq_len)

        assert torch.equal(mapping1, mapping2)

    def test_mapping_edge_cases(self):
        """Test edge cases for Hilbert mapping."""
        # Very small sequences
        assert torch.equal(create_hilbert_mapping(1), torch.tensor([0]))
        assert create_hilbert_mapping(2).shape == (2,)

        # Non-power-of-2 sequences
        for seq_len in [63, 127, 255, 511]:
            mapping = create_hilbert_mapping(seq_len)
            assert mapping.shape == (seq_len,)
            assert len(set(mapping.tolist())) == seq_len  # All unique


class TestHilbertAttentionSimple:
    """Test PyTorch-based Hilbert attention implementation."""

    @pytest.fixture
    def model_configs(self):
        """Different model configurations to test."""
        return [
            # (hidden_dim, num_heads, segment_size, dilation_rate)
            (256, 8, 64, 1),
            (512, 8, 128, 1),
            (768, 12, 128, 2),
            (1024, 16, 256, 4),
        ]

    def test_forward_shape(self, model_configs):
        """Test output shapes are correct."""
        device = get_device()
        batch_size = 2

        for hidden_dim, num_heads, segment_size, dilation_rate in model_configs:
            model = HilbertAttentionSimple(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                use_hilbert=True,
            ).to(device)

            # Test different sequence lengths
            for seq_len in [segment_size, segment_size * 2, segment_size * 3 + 16]:
                x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

                with torch.no_grad():
                    output = model(x)

                assert output.shape == (batch_size, seq_len, hidden_dim)
                assert output.dtype == x.dtype
                assert not torch.isnan(output).any()

    def test_gradient_flow(self, model_configs):
        """Test that gradients flow correctly."""
        device = get_device()
        if device.type == "mps":
            pytest.skip("MPS doesn't support all operations needed for this test")

        batch_size = 2

        for hidden_dim, num_heads, segment_size, dilation_rate in model_configs[
            :2
        ]:  # Test subset
            model = HilbertAttentionSimple(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                use_hilbert=True,
            ).to(device)

            seq_len = segment_size * 2
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device=device, requires_grad=True
            )

            # Forward pass
            output = model(x)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Check gradients exist
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()
            assert x.grad.abs().sum() > 0

            # Check model gradients
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()

    def test_hilbert_vs_standard(self):
        """Test difference between Hilbert and standard attention."""
        device = get_device()
        torch.manual_seed(42)

        hidden_dim, num_heads = 256, 8
        segment_size, dilation_rate = 64, 1
        batch_size, seq_len = 2, 128

        model = HilbertAttentionSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_hilbert=True,  # This flag is used internally
        ).to(device)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Compare outputs (they should be different due to reordering)
        with torch.no_grad():
            # Hilbert attention uses reordering by default
            output_hilbert = model(x, is_causal=False)

            # To test without Hilbert, we'd need to modify the model
            # For now, just verify the output is valid
            assert output_hilbert.shape == x.shape
            assert not torch.isnan(output_hilbert).any()
            assert torch.std(output_hilbert) > 0  # Non-trivial output

    def test_causal_attention(self):
        """Test causal attention mode."""
        device = get_device()

        model = HilbertAttentionSimple(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
            dilation_rate=1,
            use_hilbert=True,
        ).to(device)

        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, 256, device=device)

        with torch.no_grad():
            output_causal = model(x, is_causal=True)
            output_non_causal = model(x, is_causal=False)

        # Outputs should be different
        assert not torch.allclose(output_causal, output_non_causal)

        # Both should be valid
        assert not torch.isnan(output_causal).any()
        assert not torch.isnan(output_non_causal).any()


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestHilbertAttentionCore:
    """Test Triton-based Hilbert attention implementation."""

    @pytest.fixture
    def model_configs(self):
        """Model configurations compatible with Triton constraints."""
        return [
            # (hidden_dim, num_heads, segment_size, dilation_rate)
            (256, 8, 64, 1),
            (512, 8, 128, 1),
            (768, 12, 128, 2),
            (1024, 16, 256, 2),
        ]

    def test_forward_shape(self, model_configs):
        """Test output shapes with Triton kernels."""
        device = get_device()
        if device.type != "cuda":
            pytest.skip("Triton requires CUDA")

        batch_size = 2

        for hidden_dim, num_heads, segment_size, dilation_rate in model_configs:
            model = HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                use_custom_backward=True,
            ).to(device)

            # Test different sequence lengths
            for seq_len in [segment_size, segment_size * 2]:
                x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

                with torch.no_grad():
                    output = model(x, use_hilbert=True)

                assert output.shape == (batch_size, seq_len, hidden_dim)
                assert not torch.isnan(output).any()

    def test_gradient_flow(self, model_configs):
        """Test gradient flow through Triton kernels."""
        device = get_device()
        if device.type != "cuda":
            pytest.skip("Triton requires CUDA")

        batch_size = 2

        for hidden_dim, num_heads, segment_size, dilation_rate in model_configs[:2]:
            model = HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                use_custom_backward=True,
            ).to(device)

            seq_len = segment_size * 2
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device=device, requires_grad=True
            )

            # Forward pass
            output = model(x, use_hilbert=True)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Check gradients
            assert x.grad is not None
            assert not torch.isnan(x.grad).any()
            assert x.grad.abs().sum() > 0

    def test_custom_vs_pytorch_backward(self):
        """Compare custom backward with PyTorch autograd."""
        device = get_device()
        if device.type != "cuda":
            pytest.skip("Triton requires CUDA")

        torch.manual_seed(42)

        hidden_dim, num_heads = 256, 8
        segment_size, dilation_rate = 64, 1
        batch_size, seq_len = 2, 128

        # Model with custom backward
        model_custom = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=True,
        ).to(device)

        # Model without custom backward (uses PyTorch autograd)
        model_pytorch = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        # Copy weights
        model_pytorch.load_state_dict(model_custom.state_dict())

        x = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, requires_grad=True
        )
        x_copy = x.clone().detach().requires_grad_(True)

        # Forward and backward with custom
        output_custom = model_custom(x, use_hilbert=True)
        loss_custom = output_custom.sum()
        loss_custom.backward()

        # Forward and backward with PyTorch
        output_pytorch = model_pytorch(x_copy, use_hilbert=True)
        loss_pytorch = output_pytorch.sum()
        loss_pytorch.backward()

        # Outputs should be very close
        assert torch.allclose(output_custom, output_pytorch, rtol=1e-4, atol=1e-6)

        # Gradients should be close (may have small differences due to numerical precision)
        assert torch.allclose(x.grad, x_copy.grad, rtol=1e-3, atol=1e-5)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        device = get_device()
        if device.type != "cuda":
            pytest.skip("Triton requires CUDA")

        model = HilbertAttentionCore(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
            dilation_rate=1,
        ).to(device)

        batch_size, seq_len = 2, 128

        # Test with very small values
        x_small = torch.randn(batch_size, seq_len, 256, device=device) * 1e-5
        with torch.no_grad():
            output_small = model(x_small, use_hilbert=True)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()

        # Test with larger values
        x_large = torch.randn(batch_size, seq_len, 256, device=device) * 10
        with torch.no_grad():
            output_large = model(x_large, use_hilbert=True)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()


class TestHilbertAttentionWrapper:
    """Test the Q,K,V interface wrapper."""

    def test_wrapper_interface(self):
        """Test wrapper accepts Q,K,V tensors."""
        device = get_device()

        batch_size, seq_len = 2, 128
        num_heads, head_dim = 8, 64

        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[64, 128],
            dilation_rates=[1, 2],
            dropout=0.1,
            num_heads=num_heads,
            head_dim=head_dim,
        ).to(device)

        # Create Q, K, V tensors
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        with torch.no_grad():
            output = wrapper(q, k, v)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_wrapper_gradient_flow(self):
        """Test gradients flow through wrapper."""
        device = get_device()
        if device.type == "mps":
            pytest.skip("MPS doesn't support all operations needed")

        batch_size, seq_len = 2, 128
        num_heads, head_dim = 8, 64

        wrapper = HilbertAttentionTritonWrapper(
            segment_lengths=[64],
            dilation_rates=[1],
            num_heads=num_heads,
            head_dim=head_dim,
        ).to(device)

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )

        output = wrapper(q, k, v)
        loss = output.sum()
        loss.backward()

        # Check all inputs have gradients
        assert q.grad is not None and q.grad.abs().sum() > 0
        assert k.grad is not None and k.grad.abs().sum() > 0
        assert v.grad is not None and v.grad.abs().sum() > 0

    def test_wrapper_fixed_alias(self):
        """Test HilbertAttentionTritonFixed alias works."""
        device = get_device()

        fixed = HilbertAttentionTritonFixed(
            segment_lengths=[64],
            dilation_rates=[1],
            num_heads=8,
            head_dim=64,
        ).to(device)

        assert isinstance(fixed, HilbertAttentionTritonWrapper)

        # Test it works
        q = torch.randn(2, 128, 8, 64, device=device)
        k = torch.randn(2, 128, 8, 64, device=device)
        v = torch.randn(2, 128, 8, 64, device=device)

        with torch.no_grad():
            output = fixed(q, k, v)
        assert output.shape == q.shape


class TestMemoryEfficiency:
    """Test memory efficiency of kernel implementations."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for memory profiling"
    )
    def test_memory_usage(self):
        """Compare memory usage between implementations."""
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        batch_size, seq_len = 4, 1024
        hidden_dim, num_heads = 512, 8
        segment_size = 128

        # Test Simple implementation
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model_simple = HilbertAttentionSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=1,
        ).to(device)

        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        with torch.no_grad():
            _ = model_simple(x)

        memory_simple = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Test Core implementation (if Triton available)
        if TRITON_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model_core = HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=1,
            ).to(device)

            x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

            with torch.no_grad():
                _ = model_core(x)

            memory_core = torch.cuda.max_memory_allocated() / 1024**2  # MB

            print(
                f"Memory usage - Simple: {memory_simple:.2f} MB, Core: {memory_core:.2f} MB"
            )

            # Core should not use significantly more memory
            assert memory_core < memory_simple * 2.0
        else:
            print(f"Memory usage - Simple: {memory_simple:.2f} MB")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_dimensions(self):
        """Test handling of invalid dimensions."""
        _ = get_device()

        # Hidden dim not divisible by num_heads
        with pytest.raises(ValueError):
            HilbertAttentionCore(
                hidden_dim=257,  # Not divisible by 8
                num_heads=8,
                segment_size=64,
            )

    def test_empty_input(self):
        """Test handling of empty tensors."""
        device = get_device()

        model = HilbertAttentionSimple(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
        ).to(device)

        # Zero batch size
        x = torch.randn(0, 128, 256, device=device)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (0, 128, 256)

        # Zero sequence length (should handle padding)
        x = torch.randn(2, 0, 256, device=device)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 0, 256)

    def test_very_long_sequences(self):
        """Test handling of very long sequences."""
        device = get_device()
        if device.type == "mps":
            pytest.skip("MPS may not support very long sequences")

        model = HilbertAttentionSimple(
            hidden_dim=128,  # Smaller for memory
            num_heads=4,
            segment_size=256,
            dilation_rate=2,
        ).to(device)

        # Test with long sequence
        batch_size = 1
        seq_len = 4096
        x = torch.randn(batch_size, seq_len, 128, device=device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, seq_len, 128)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
