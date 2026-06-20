#!/usr/bin/env python3
"""
Test the consolidated Hilbert kernel implementation.
"""

import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_consolidated_kernel():
    """Test that the consolidated kernel works correctly."""
    from dilated_attention_pytorch.kernels import UnifiedHilbertAttention

    # Test various configurations
    configs = [
        (
            128,
            256,
            8,
            64,
            1,
        ),  # (seq_len, hidden_dim, num_heads, segment_size, dilation_rate)
        (256, 512, 16, 128, 2),
        (512, 768, 12, 256, 4),
    ]

    for seq_len, hidden_dim, num_heads, segment_size, dilation_rate in configs:
        # Create module
        module = UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda()

        # Test input
        x = torch.randn(2, seq_len, hidden_dim, device="cuda")

        # Forward pass
        with torch.no_grad():
            output = module(x, use_hilbert=True)

        # Verify output shape
        assert output.shape == x.shape

        # Verify no NaN/Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"✓ Config: seq={seq_len}, hidden={hidden_dim}, heads={num_heads}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_optimal_block_sizes():
    """Test the optimal block size selection."""
    from dilated_attention_pytorch.kernels import UnifiedHilbertAttention

    module = UnifiedHilbertAttention(
        hidden_dim=256,
        num_heads=8,
        segment_size=128,
        dilation_rate=2,
    ).cuda()

    device = torch.device("cuda")

    # Test different sequence lengths
    seq_lengths = [64, 256, 512, 1024, 2048]

    for seq_len in seq_lengths:
        block_m, block_n, block_d = module.get_optimal_block_sizes(seq_len, device)

        # Verify minimum sizes
        assert block_m >= 16
        assert block_n >= 16
        assert block_d >= 16

        # Verify they increase with sequence length
        if seq_len <= 256:
            assert block_m <= 32
        elif seq_len <= 1024:
            assert block_m == 64
        else:
            assert block_m >= 64

        print(
            f"✓ seq_len={seq_len}: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_D={block_d}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hilbert_mapping():
    """Test the improved Hilbert mapping generation."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        create_hilbert_mapping,
    )

    # Test small sequence (identity)
    mapping = create_hilbert_mapping(32)
    assert mapping.shape == (32,)
    assert torch.equal(mapping, torch.arange(32, dtype=torch.int32))

    # Test medium sequence (true Hilbert)
    mapping = create_hilbert_mapping(256)
    assert mapping.shape == (256,)
    assert mapping.min() == 0
    assert mapping.max() == 255
    assert len(mapping.unique()) == 256  # All positions mapped

    # Test large sequence (snake pattern)
    mapping = create_hilbert_mapping(1024)
    assert mapping.shape == (1024,)
    assert mapping.min() == 0
    assert mapping.max() == 1023
    assert len(mapping.unique()) == 1024

    print("✓ Hilbert mapping generation works correctly")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gradient_flow():
    """Test gradient flow through the consolidated kernel."""
    from dilated_attention_pytorch.kernels import UnifiedHilbertAttention

    module = UnifiedHilbertAttention(
        hidden_dim=128,
        num_heads=4,
        segment_size=64,
        dilation_rate=1,
        use_custom_backward=True,
    ).cuda()

    # Test input requiring grad
    x = torch.randn(1, 128, 128, device="cuda", requires_grad=True)

    # Forward pass
    output = module(x, use_hilbert=True)

    # Backward pass
    loss = output.mean()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Check model gradients
    for name, param in module.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()
            print(f"✓ Gradient exists for {name}")


if __name__ == "__main__":
    print("Testing consolidated Hilbert kernel implementation...")
    test_consolidated_kernel()
    test_optimal_block_sizes()
    test_hilbert_mapping()
    test_gradient_flow()
    print("\n✅ All tests passed!")
