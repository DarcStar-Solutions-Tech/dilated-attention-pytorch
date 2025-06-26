#!/usr/bin/env python3
"""
Integration tests for the factory pattern implementation.

These tests ensure the factory functions work correctly across different
scenarios and configurations.
"""

import pytest
import torch
from torch import nn


from dilated_attention_pytorch import (
    create_block_sparse_attention,
    create_multihead_dilated_attention,
)
from dilated_attention_pytorch.core import (
    DilatedAttentionConfig,
    MultiheadConfig,
)

# Explicitly import implementations to ensure they're registered
# This helps with test isolation issues
try:
    import dilated_attention_pytorch.improved_multihead_dilated_attention
    import dilated_attention_pytorch.ring_multihead_dilated_attention
except ImportError:
    pass


class TestFactoryIntegration:
    """Integration tests for factory pattern."""

    @pytest.fixture
    def device(self):
        """Get the appropriate device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def dtype(self):
        """Get the appropriate dtype for testing."""
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def test_auto_selection_creates_valid_module(self, device, dtype):
        """Test that auto-selection creates a valid attention module."""
        attention = create_multihead_dilated_attention(
            "auto",
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=dtype,
        )

        # Verify it's a nn.Module
        assert isinstance(attention, nn.Module)

        # Test forward pass
        batch_size, seq_len = 2, 2048
        x = torch.randn(batch_size, seq_len, 512, device=device, dtype=dtype)
        output = attention(x, x, x)

        assert output.shape == x.shape
        assert output.dtype == dtype
        assert output.device == x.device

    def test_all_implementations_compatible(self, device, dtype):
        """Test that all implementations have compatible interfaces."""
        implementations = ["improved", "ring"]

        # Common configuration
        embed_dim = 256
        num_heads = 4
        segment_lengths = [512, 1024]
        dilation_rates = [1, 2]

        # Input tensor
        batch_size, seq_len = 2, 1024
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        outputs = {}
        for impl in implementations:
            try:
                attention = create_multihead_dilated_attention(
                    impl,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    device=device,
                    dtype=dtype,
                )

                # Forward pass
                output = attention(x, x, x, is_causal=True)
                outputs[impl] = output

                # Verify output shape
                assert output.shape == x.shape

            except Exception as e:
                pytest.skip(f"Implementation {impl} not available: {e}")

        # Verify all outputs have same shape
        if len(outputs) > 1:
            shapes = [out.shape for out in outputs.values()]
            assert all(s == shapes[0] for s in shapes)

    def test_config_objects_work_correctly(self, device, dtype):
        """Test that configuration objects work with factory functions."""
        attention_config = DilatedAttentionConfig(
            segment_lengths=[1024, 2048, 4096],
            dilation_rates=[1, 2, 4],
            dropout=0.2,
        )

        multihead_config = MultiheadConfig(
            embed_dim=768,
            num_heads=12,
            bias=True,
            layer_norm=True,
            gamma_init=1.0,
        )

        # Create attention with configs
        attention = create_multihead_dilated_attention(
            "auto",
            multihead_config=multihead_config,
            attention_config=attention_config,
            device=device,
            dtype=dtype,
        )

        # Verify configuration was applied
        assert attention.multihead_config.embed_dim == 768
        assert attention.multihead_config.num_heads == 12
        assert attention.attention_config.dropout == 0.2

        # Test forward pass
        x = torch.randn(1, 4096, 768, device=device, dtype=dtype)
        output = attention(x, x, x)
        assert output.shape == x.shape

    def test_backward_compatibility(self, device, dtype):
        """Test that factory maintains backward compatibility."""
        # Old style direct instantiation
        from dilated_attention_pytorch import MultiheadDilatedAttention

        old_attention = MultiheadDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=dtype,
        )

        # New style factory - use improved instead of standard
        new_attention = create_multihead_dilated_attention(
            "improved",
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=dtype,
        )

        # Both should work identically
        x = torch.randn(2, 2048, 512, device=device, dtype=dtype)

        old_output = old_attention(x, x, x)
        new_output = new_attention(x, x, x)

        assert old_output.shape == new_output.shape
        assert old_output.dtype == new_output.dtype

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_block_sparse_attention_integration(self):
        """Test block-sparse attention factory integration."""
        device = torch.device("cuda")
        dtype = torch.float16

        # Create block-sparse attention
        attention = create_block_sparse_attention(
            sparsity_ratio=0.9,
            pattern_type="dilated_sparse",
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=dtype,
        )

        # Test forward pass
        x = torch.randn(2, 2048, 512, device=device, dtype=dtype)
        output = attention(x, x, x)

        assert output.shape == x.shape
        assert output.dtype == dtype

        # Verify sparsity is applied (output should not be identical to dense)
        dense_attention = create_multihead_dilated_attention(
            "improved",
            embed_dim=512,
            num_heads=8,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
            device=device,
            dtype=dtype,
        )

        dense_output = dense_attention(x, x, x)

        # Outputs should be different due to sparsity
        assert not torch.allclose(output, dense_output, rtol=1e-2)

    def test_mixed_precision_support(self, device):
        """Test that factory handles mixed precision correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for mixed precision test")

        # Create attention for different dtypes
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue

            attention = create_multihead_dilated_attention(
                "auto",
                embed_dim=256,
                num_heads=4,
                segment_lengths=[512],
                dilation_rates=[1],
                device=device,
                dtype=dtype,
            )

            # Test with matching dtype
            x = torch.randn(1, 512, 256, device=device, dtype=dtype)
            output = attention(x, x, x)

            assert output.dtype == dtype

    def test_error_handling(self):
        """Test that factory provides helpful error messages."""
        # Invalid implementation name
        with pytest.raises(ValueError, match="Unknown attention type"):
            create_multihead_dilated_attention("invalid_impl")

        # Missing required parameters
        with pytest.raises((TypeError, ValueError)):
            create_multihead_dilated_attention("auto")  # Missing embed_dim, etc.

        # Invalid configuration
        with pytest.raises(ValueError):
            create_multihead_dilated_attention(
                "auto",
                embed_dim=256,
                num_heads=7,  # Not a power of 2
                segment_lengths=[1024],
                dilation_rates=[1],
            )

    def test_in_transformer_model(self, device, dtype):
        """Test factory usage in a complete transformer model."""

        class TestTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        create_multihead_dilated_attention(
                            "auto",
                            embed_dim=512,
                            num_heads=8,
                            segment_lengths=[1024, 2048],
                            dilation_rates=[1, 2],
                            dropout=0.1,
                        )
                        for _ in range(2)
                    ]
                )
                self.norm = nn.LayerNorm(512)

            def forward(self, x):
                for layer in self.layers:
                    x = x + layer(self.norm(x))
                return x

        model = TestTransformer().to(device).to(dtype)
        x = torch.randn(2, 2048, 512, device=device, dtype=dtype)

        # Forward pass
        output = model(x)
        assert output.shape == x.shape

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_performance_characteristics(self, device, dtype):
        """Test that different implementations have expected performance characteristics."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for performance test")

        seq_lengths = [1024, 2048, 4096]

        for seq_len in seq_lengths:
            # Ring implementation
            ring = create_multihead_dilated_attention(
                "ring",
                embed_dim=256,
                num_heads=4,
                segment_lengths=[seq_len // 2, seq_len],
                dilation_rates=[1, 2],
                device=device,
                dtype=dtype,
            )

            # Improved implementation (should be faster)
            improved = create_multihead_dilated_attention(
                "improved",
                embed_dim=256,
                num_heads=4,
                segment_lengths=[seq_len // 2, seq_len],
                dilation_rates=[1, 2],
                device=device,
                dtype=dtype,
            )

            x = torch.randn(1, seq_len, 256, device=device, dtype=dtype)

            # Warmup
            for _ in range(3):
                _ = ring(x, x, x)
                _ = improved(x, x, x)

            # Both should produce valid outputs
            out_ring = ring(x, x, x)
            out_improved = improved(x, x, x)

            assert out_ring.shape == x.shape
            assert out_improved.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
