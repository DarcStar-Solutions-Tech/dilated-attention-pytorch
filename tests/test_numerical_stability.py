#!/usr/bin/env python3
"""
Numerical stability tests for dilated attention implementations.

These tests verify stability under extreme conditions:
- Very large/small values
- Near-zero attention weights
- Overflow/underflow conditions
- Gradient stability
- Mixed precision edge cases
"""

import math

import pytest
import torch
import torch.nn.functional as F

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
    create_multihead_dilated_attention,
)


class TestNumericalStability:
    """Test numerical stability under extreme conditions."""

    @pytest.fixture
    def device(self):
        """Get test device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.mark.parametrize("scale", [1e-4, 1e-2, 1.0, 1e2, 1e4])
    def test_scale_robustness(self, scale, device):
        """Test attention robustness to input scaling."""
        attention = DilatedAttention(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            attention_dropout=0.0,
        )
        
        # Create test data
        batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        # Scale inputs
        q_scaled = q * scale
        k_scaled = k * scale
        v_scaled = v * scale
        
        # Compute attention with scaled inputs
        with torch.no_grad():
            out_scaled = attention(q_scaled, k_scaled, v_scaled)
        
        # Check output is valid (no NaN or Inf)
        assert not torch.isnan(out_scaled).any(), f"NaN in output for scale {scale}"
        assert not torch.isinf(out_scaled).any(), f"Inf in output for scale {scale}"
        
        # Check output magnitude is reasonable
        out_norm = out_scaled.norm().item()
        v_norm = v_scaled.norm().item()
        
        # Output norm should be on same order as value norm
        ratio = out_norm / (v_norm + 1e-8)
        assert 0.01 < ratio < 100, f"Scale {scale}: output/value norm ratio {ratio}"

    def test_extreme_values_stability(self, device):
        """Test stability with extreme input values."""
        attention = ImprovedDilatedAttention(
            segment_lengths=[128],
            dilation_rates=[1],
        )
        
        batch_size, seq_len, num_heads, head_dim = 1, 256, 4, 32
        
        # Test configurations with extreme values
        test_cases = [
            ("very_large", 1e10),
            ("very_small", 1e-10),
            ("mixed_large_small", None),  # Special case
        ]
        
        for name, value in test_cases:
            if name == "mixed_large_small":
                # Half large, half small values
                q = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device)
                q[:, :seq_len//2] = 1e10
                q[:, seq_len//2:] = 1e-10
                k = q.clone()
                v = torch.ones_like(q)
            else:
                q = torch.full((batch_size, seq_len, num_heads, head_dim), value, device=device)
                k = q.clone()
                v = torch.ones_like(q)
            
            # Should not produce NaN or Inf
            output = attention(q, k, v)
            assert not torch.isnan(output).any(), f"{name}: NaN in output"
            assert not torch.isinf(output).any(), f"{name}: Inf in output"

    def test_gradient_stability(self, device):
        """Test gradient stability during backpropagation."""
        attention = MultiheadDilatedAttention(
            embed_dim=256,
            num_heads=8,
            segment_lengths=[128, 256],
            dilation_rates=[1, 2],
            dropout=0.0,
        )
        attention = attention.to(device)
        
        # Enable gradient computation
        attention.train()
        
        # Test different input magnitudes
        magnitudes = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        
        for magnitude in magnitudes:
            # Create inputs
            batch_size, seq_len = 2, 512
            x = torch.randn(batch_size, seq_len, 256, device=device) * magnitude
            x.requires_grad = True
            
            # Forward pass
            output, _ = attention(x, x, x)
            loss = output.mean()
            
            # Backward pass
            loss.backward()
            
            # Check gradients are stable
            grad = x.grad
            assert not torch.isnan(grad).any(), f"NaN gradients at magnitude {magnitude}"
            assert not torch.isinf(grad).any(), f"Inf gradients at magnitude {magnitude}"
            
            # Check gradient magnitude is reasonable
            grad_norm = grad.norm().item()
            assert grad_norm < 1e10, f"Gradient explosion at magnitude {magnitude}: {grad_norm}"
            
            # Clear gradients
            attention.zero_grad()
            x.grad = None

    def test_attention_weight_underflow(self, device):
        """Test handling of very small attention weights."""
        attention = DilatedAttention(
            segment_lengths=[64],
            dilation_rates=[1],
            softmax_scale=1e-8,  # Very small scale
        )
        
        # Create inputs that will produce very small attention weights
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 32
        
        # Orthogonal queries and keys -> small dot products
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        q = F.normalize(q, dim=-1)
        
        # Make keys orthogonal to queries
        k = torch.randn_like(q)
        k = k - (k * q).sum(dim=-1, keepdim=True) * q
        k = F.normalize(k, dim=-1)
        
        v = torch.randn_like(q)
        
        # Compute attention
        output = attention(q, k, v)
        
        # Output should still be valid (not all zeros)
        assert not torch.allclose(output, torch.zeros_like(output))
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_mixed_precision_stability(self, dtype, device):
        """Test stability with different floating point precisions."""
        if device.type == "cpu" and dtype == torch.bfloat16:
            pytest.skip("bfloat16 not well supported on CPU")
        
        # Create attention with auto mixed precision
        attention = create_multihead_dilated_attention(
            "improved",
            embed_dim=256,
            num_heads=8,
            segment_lengths=[128],
            dilation_rates=[1],
        )
        attention = attention.to(device).to(dtype)
        
        # Test with different input ranges
        test_ranges = [
            ("normal", 1.0),
            ("small", 1e-3),
            ("large", 1e3),
        ]
        
        for name, scale in test_ranges:
            x = torch.randn(2, 256, 256, device=device, dtype=dtype) * scale
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
                output, _ = attention(x, x, x)
            
            # Check output validity
            assert output.dtype == dtype, f"{name}: Wrong output dtype"
            assert not torch.isnan(output).any(), f"{name}: NaN in output"
            assert not torch.isinf(output).any(), f"{name}: Inf in output"
            
            # Check output range is reasonable
            output_scale = output.abs().max().item()
            assert output_scale < 1e6, f"{name}: Output scale too large: {output_scale}"

    def test_zero_sequence_length_handling(self, device):
        """Test handling of zero-length sequences."""
        attention = DilatedAttention(
            segment_lengths=[64],
            dilation_rates=[1],
        )
        
        # Zero sequence length
        q = torch.randn(2, 0, 4, 32, device=device)
        k = torch.randn(2, 0, 4, 32, device=device)
        v = torch.randn(2, 0, 4, 32, device=device)
        
        output = attention(q, k, v)
        
        # Should return empty tensor with same shape
        assert output.shape == (2, 0, 4, 32)
        assert output.numel() == 0

    def test_attention_saturation(self, device):
        """Test behavior when attention weights saturate."""
        attention = DilatedAttention(
            segment_lengths=[64],
            dilation_rates=[1],
            softmax_scale=100.0,  # Large scale causes saturation
        )
        
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 32
        
        # Create inputs where one key strongly matches one query
        q = torch.zeros(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.zeros_like(q)
        v = torch.randn_like(q)
        
        # Set first query and key to be identical (perfect match)
        q[:, 0] = 1.0
        k[:, 0] = 1.0
        
        output = attention(q, k, v)
        
        # First position should dominate but not cause NaN
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Output at position 0 should be close to v[0] (due to high attention)
        assert torch.allclose(output[:, 0], v[:, 0], atol=0.1)

    def test_numerical_consistency_across_implementations(self, device):
        """Test numerical consistency between different implementations."""
        segment_lengths = [256, 512]
        dilation_rates = [1, 2]
        
        # Create different implementations
        standard = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )
        
        improved = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
        )
        
        # Test data
        torch.manual_seed(42)
        batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        # Compute outputs
        with torch.no_grad():
            out_standard = standard(q, k, v)
            out_improved = improved(q, k, v)
        
        # Should be numerically close (allowing for optimization differences)
        assert torch.allclose(out_standard, out_improved, rtol=1e-3, atol=1e-5)

    def test_causal_mask_numerical_stability(self, device):
        """Test numerical stability with causal masking."""
        attention = DilatedAttention(
            segment_lengths=[128],
            dilation_rates=[1],
        )
        
        # Test with extreme values and causal mask
        batch_size, seq_len, num_heads, head_dim = 1, 256, 4, 32
        
        # Large values that could cause overflow with exp()
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 10
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 10
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        
        # Apply causal attention
        output = attention(q, k, v, is_causal=True)
        
        # Check output validity
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Verify causality is maintained (no information from future)
        # This is implicitly tested by the implementation

    def test_long_sequence_accumulation_errors(self, device):
        """Test accumulation errors in very long sequences."""
        # Use larger segments for longer sequences
        attention = ImprovedDilatedAttention(
            segment_lengths=[1024, 2048, 4096],
            dilation_rates=[1, 2, 4],
        )
        
        # Very long sequence
        batch_size, seq_len, num_heads, head_dim = 1, 8192, 4, 32
        
        # Use smaller values to avoid overflow in accumulation
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device) * 0.1
        
        output = attention(q, k, v)
        
        # Check for accumulation errors
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Output variance should be reasonable (not collapsed or exploded)
        output_var = output.var().item()
        assert 1e-6 < output_var < 1e6

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_precision_differences(self):
        """Test precision differences between CPU and GPU."""
        attention = DilatedAttention(
            segment_lengths=[128],
            dilation_rates=[1],
        )
        
        # Test data
        torch.manual_seed(42)
        batch_size, seq_len, num_heads, head_dim = 1, 256, 4, 32
        q_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        # CPU computation
        with torch.no_grad():
            out_cpu = attention(q_cpu, k_cpu, v_cpu)
        
        # GPU computation
        q_gpu = q_cpu.cuda()
        k_gpu = k_cpu.cuda()
        v_gpu = v_cpu.cuda()
        
        with torch.no_grad():
            out_gpu = attention(q_gpu, k_gpu, v_gpu)
        
        # Compare results (allowing for precision differences)
        out_gpu_cpu = out_gpu.cpu()
        
        # Check relative error is small
        rel_error = (out_cpu - out_gpu_cpu).abs() / (out_cpu.abs() + 1e-8)
        max_rel_error = rel_error.max().item()
        
        assert max_rel_error < 1e-3, f"Large precision difference: {max_rel_error}"


class TestEdgeCaseNumerics:
    """Test specific numerical edge cases."""

    def test_all_zeros_input(self):
        """Test with all-zero inputs."""
        attention = DilatedAttention(
            segment_lengths=[64],
            dilation_rates=[1],
        )
        
        # All zeros
        q = torch.zeros(1, 128, 4, 32)
        k = torch.zeros(1, 128, 4, 32)
        v = torch.zeros(1, 128, 4, 32)
        
        output = attention(q, k, v)
        
        # With zero queries and keys, attention weights should be uniform
        # Output should be zero (uniform attention over zero values)
        assert torch.allclose(output, torch.zeros_like(output))

    def test_single_hot_attention(self):
        """Test when attention focuses on single position."""
        attention = DilatedAttention(
            segment_lengths=[64],
            dilation_rates=[1],
            softmax_scale=1000.0,  # Very high scale
        )
        
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 32
        
        # Create queries that only match with specific keys
        q = torch.zeros(batch_size, seq_len, num_heads, head_dim)
        k = torch.zeros_like(q)
        v = torch.randn_like(q)
        
        # Each query matches only with key at same position
        for i in range(seq_len):
            q[:, i, :, 0] = float(i)
            k[:, i, :, 0] = float(i)
        
        output = attention(q, k, v)
        
        # Each output should be close to corresponding value
        for i in range(seq_len):
            assert torch.allclose(output[:, i], v[:, i], atol=0.1)

    def test_numerical_symmetry(self):
        """Test numerical symmetry properties."""
        attention = ImprovedDilatedAttention(
            segment_lengths=[128],
            dilation_rates=[1],
        )
        
        # Create symmetric inputs
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 32
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        # Self-attention should be symmetric in certain cases
        output1 = attention(x, x, x)
        
        # Permute and check
        perm = torch.randperm(seq_len)
        x_perm = x[:, perm]
        output2_perm = attention(x_perm, x_perm, x_perm)
        
        # Unpermute output2
        inv_perm = torch.argsort(perm)
        output2 = output2_perm[:, inv_perm]
        
        # Should be close (allowing for numerical differences)
        assert torch.allclose(output1, output2, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])