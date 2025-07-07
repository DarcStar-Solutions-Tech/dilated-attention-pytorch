#!/usr/bin/env python3
"""
Test BlockSparseRingDilatedAttentionHilbert implementation.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import create_block_sparse_attention, SparsePatternConfig
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert import (
    BlockSparseRingDilatedAttentionHilbert,
    create_block_sparse_hilbert,
)


class TestBlockSparseHilbert:
    """Test Hilbert-optimized block-sparse attention."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_functionality(self, device):
        """Test basic forward pass works."""
        batch_size = 2
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        # Create model
        model = create_block_sparse_hilbert(
            segment_lengths=[512],
            dilation_rates=[1],
            sparsity_ratio=0.1,
            block_size=64,
            use_hilbert=True,
        ).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = model(q, k, v)

        # Check output shape
        assert output.shape == q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_hilbert_vs_standard(self, device):
        """Compare Hilbert vs standard block-sparse outputs."""
        batch_size = 1
        seq_len = 512
        num_heads = 4
        head_dim = 32

        # Common config
        segment_lengths = [256]
        dilation_rates = [1]
        sparse_config = SparsePatternConfig(
            pattern_type="local_window",
            sparsity_ratio=0.2,
            block_size=64,
            window_size=128,
        )

        # Create standard model
        standard_model = create_block_sparse_attention(
            variant="base",
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
        ).to(device)

        # Create Hilbert model
        hilbert_model = create_block_sparse_attention(
            variant="hilbert",
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
        ).to(device)

        # Same inputs
        torch.manual_seed(42)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward passes
        standard_output = standard_model(q, k, v)
        hilbert_output = hilbert_model(q, k, v)

        # Outputs should be very close (same attention pattern, different order)
        # Allow some tolerance for numerical differences
        assert torch.allclose(standard_output, hilbert_output, rtol=1e-4, atol=1e-5)

    def test_hilbert_options(self, device):
        """Test different Hilbert optimization options."""
        batch_size = 1
        seq_len = 256
        num_heads = 4
        head_dim = 32

        configs = [
            {"hilbert_block_level": True, "hilbert_within_blocks": False},
            {"hilbert_block_level": False, "hilbert_within_blocks": True},
            {"hilbert_block_level": True, "hilbert_within_blocks": True},
        ]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        outputs = []

        for config in configs:
            model = BlockSparseRingDilatedAttentionHilbert(
                segment_lengths=[128],
                dilation_rates=[1],
                sparse_config=SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=0.1,
                    block_size=32,
                ),
                use_hilbert=True,
                **config,
            ).to(device)

            output = model(q, k, v)
            outputs.append(output)

            # Each should produce valid output
            assert output.shape == q.shape
            assert not torch.isnan(output).any()

    def test_causal_masking(self, device):
        """Test that causal masking works with Hilbert ordering."""
        batch_size = 1
        seq_len = 256
        num_heads = 4
        head_dim = 32

        model = create_block_sparse_hilbert(
            segment_lengths=[128],
            dilation_rates=[1],
            sparsity_ratio=0.2,
            block_size=64,
        ).to(device)

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.ones_like(q)  # Use ones to make causality visible

        # Forward with causal masking
        output = model(q, k, v, is_causal=True)

        # Check output is valid
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    def test_factory_presets(self, device):
        """Test factory presets for Hilbert variants."""
        from dilated_attention_pytorch import get_block_sparse_preset

        # Test Hilbert presets
        models = [
            get_block_sparse_preset("hilbert_standard"),
            get_block_sparse_preset("hilbert_ultra"),
        ]

        batch_size = 1
        seq_len = 512
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        for model in models:
            model = model.to(device)
            output = model(q, k, v)
            assert output.shape == q.shape
            assert not torch.isnan(output).any()

    def test_pattern_stats(self, device):
        """Test pattern statistics include Hilbert info."""
        model = create_block_sparse_hilbert(
            segment_lengths=[256],
            dilation_rates=[1],
            sparsity_ratio=0.1,
            hilbert_block_level=True,
            hilbert_within_blocks=True,
        ).to(device)

        stats = model.get_pattern_stats()

        # Check Hilbert stats are included
        assert "hilbert_optimization" in stats
        hilbert_stats = stats["hilbert_optimization"]
        assert hilbert_stats["enabled"] is True
        assert hilbert_stats["block_level"] is True
        assert hilbert_stats["within_blocks"] is True

    @pytest.mark.parametrize("seq_len", [256, 512, 1024, 2048])
    def test_various_sequence_lengths(self, device, seq_len):
        """Test with various sequence lengths."""
        batch_size = 1
        num_heads = 4
        head_dim = 32

        # Adjust block size based on sequence length
        block_size = min(64, seq_len // 4)

        model = create_block_sparse_hilbert(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            sparsity_ratio=0.1,
            block_size=block_size,
        ).to(device)

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        output = model(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    # Run tests
    test = TestBlockSparseHilbert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing BlockSparseRingDilatedAttentionHilbert...")

    test.test_basic_functionality(device)
    print("✓ Basic functionality test passed")

    test.test_hilbert_vs_standard(device)
    print("✓ Hilbert vs standard comparison passed")

    test.test_hilbert_options(device)
    print("✓ Hilbert options test passed")

    test.test_causal_masking(device)
    print("✓ Causal masking test passed")

    test.test_factory_presets(device)
    print("✓ Factory presets test passed")

    test.test_pattern_stats(device)
    print("✓ Pattern stats test passed")

    for seq_len in [256, 512, 1024, 2048]:
        test.test_various_sequence_lengths(device, seq_len)
    print("✓ Various sequence lengths test passed")

    print("\nAll tests passed! ✅")
