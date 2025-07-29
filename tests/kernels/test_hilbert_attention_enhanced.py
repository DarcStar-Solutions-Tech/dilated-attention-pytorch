#!/usr/bin/env python3
"""
Test the enhanced Hilbert attention implementation.
"""

import torch
import pytest
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention
from dilated_attention_pytorch.kernels.hilbert_attention_enhanced import (
    HilbertAttentionEnhanced,
)


def test_enhanced_initialization():
    """Test enhanced implementation can be initialized."""
    model = HilbertAttentionEnhanced(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
    )

    assert model.hidden_dim == 768
    assert model.num_heads == 12
    assert model.head_dim == 64
    assert model.segment_size == 128
    assert model.dilation_rate == 1


def test_config_selection():
    """Test optimal configuration selection."""
    model = HilbertAttentionEnhanced(
        hidden_dim=768,
        num_heads=12,
    )

    # Test different sequence lengths
    configs = {
        512: model._get_optimal_config(512),
        1024: model._get_optimal_config(1024),
        2048: model._get_optimal_config(2048),
        4096: model._get_optimal_config(4096),
        8192: model._get_optimal_config(8192),
        16384: model._get_optimal_config(16384),
    }

    # Verify configs are reasonable
    for seq_len, config in configs.items():
        assert "block_m" in config
        assert "block_n" in config
        assert "block_d" in config
        assert config["block_m"] > 0
        assert config["block_n"] > 0
        assert config["block_d"] > 0

        # Check multi-row processing for larger sequences
        if seq_len >= 4096:
            assert config.get("rows_per_block", 1) >= 1
            assert config.get("fused_block_n", 0) >= config["block_n"]


def test_forward_pass_basic():
    """Test basic forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HilbertAttentionEnhanced(
        hidden_dim=256,
        num_heads=8,
        segment_size=64,
        dilation_rate=1,
    ).to(device)

    # Test input
    batch_size = 2
    seq_len = 128
    x = torch.randn(batch_size, seq_len, 256, device=device)

    # Forward pass
    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, seq_len, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_sparse_attention():
    """Test strided sparse attention."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with different dilation rates
    for dilation_rate in [2, 4, 8]:
        model = HilbertAttentionEnhanced(
            hidden_dim=256,
            num_heads=8,
            segment_size=64,
            dilation_rate=dilation_rate,
        ).to(device)

        x = torch.randn(1, 256, 256, device=device)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 256, 256)
        assert not torch.isnan(out).any()


def test_8k_optimization():
    """Test special 8K sequence optimization."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for 8K optimization test")

    device = torch.device("cuda")

    model = HilbertAttentionEnhanced(
        hidden_dim=768,
        num_heads=12,
        enable_8k_optimization=True,
    ).to(device)

    # Test with 8192 sequence length
    x = torch.randn(1, 8192, 768, device=device, dtype=torch.float16)
    model = model.half()

    # Get config for 8K
    config = model._get_optimal_config(8192)

    # Verify special 8K config is used
    if model.compute_capability >= 7:
        # Volta+ should have special config
        assert config["block_n"] == 128  # Better grid alignment

    # Test forward pass
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 8192, 768)


def test_compatibility_with_original():
    """Test that enhanced version produces similar results to original."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create both models with same parameters
    hidden_dim = 256
    num_heads = 8

    model_orig = HilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=64,
        dilation_rate=1,
        dropout=0.0,
    ).to(device)

    model_enhanced = HilbertAttentionEnhanced(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=64,
        dilation_rate=1,
        dropout=0.0,
    ).to(device)

    # Copy weights
    model_enhanced.qkv_proj.weight.data = model_orig.qkv_proj.weight.data.clone()
    model_enhanced.out_proj.weight.data = model_orig.out_proj.weight.data.clone()

    # Test input
    x = torch.randn(2, 128, hidden_dim, device=device)

    # Forward passes
    with torch.no_grad():
        out_orig = model_orig(x, use_hilbert=False)  # Disable Hilbert for consistency
        out_enhanced = model_enhanced(x, use_hilbert=False)

    # Results should be very close
    assert torch.allclose(out_orig, out_enhanced, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
@pytest.mark.parametrize("dilation_rate", [1, 2, 4])
def test_various_configurations(seq_len, dilation_rate):
    """Test various sequence lengths and dilation rates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HilbertAttentionEnhanced(
        hidden_dim=256,
        num_heads=8,
        segment_size=128,
        dilation_rate=dilation_rate,
    ).to(device)

    # Adjust batch size for memory
    batch_size = 1 if seq_len >= 2048 else 2

    x = torch.randn(batch_size, seq_len, 256, device=device)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (batch_size, seq_len, 256)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == "__main__":
    print("Testing enhanced Hilbert attention implementation...")

    test_enhanced_initialization()
    print("✓ Initialization test passed")

    test_config_selection()
    print("✓ Configuration selection test passed")

    test_forward_pass_basic()
    print("✓ Basic forward pass test passed")

    test_sparse_attention()
    print("✓ Sparse attention test passed")

    if torch.cuda.is_available():
        test_8k_optimization()
        print("✓ 8K optimization test passed")

    test_compatibility_with_original()
    print("✓ Compatibility test passed")

    print("\nAll tests passed!")
