#!/usr/bin/env python3
"""
Test backward pass (gradient computation) for per-segment Hilbert ordering.
"""

import torch
import pytest
from dilated_attention_pytorch.ring import HilbertRingAttention, RingAttentionConfig


def test_backward_pass():
    """Test that gradients flow correctly through per-segment Hilbert ordering."""
    # Skip if no GPU available
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this test")

    print("Testing backward pass with per-segment Hilbert ordering...\n")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 4
    head_dim = 32
    _ = num_heads * head_dim

    device = torch.device("cuda")
    print(f"Device: {device}")

    # Create attention modules
    config_hilbert = RingAttentionConfig(
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
        use_hilbert=True,
        hilbert_curve_level=8,
        dropout=0.0,
    )

    config_no_hilbert = RingAttentionConfig(
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
        use_hilbert=False,
        dropout=0.0,
    )

    attention_hilbert = HilbertRingAttention(config_hilbert, device=device)
    attention_no_hilbert = HilbertRingAttention(config_no_hilbert, device=device)

    # Create input tensors with requires_grad=True
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    # Forward pass with Hilbert
    output_hilbert = attention_hilbert(q.clone(), k.clone(), v.clone())

    # Forward pass without Hilbert
    output_no_hilbert = attention_no_hilbert(q.clone(), k.clone(), v.clone())

    # Create target and loss
    target = torch.randn_like(output_hilbert)
    loss_hilbert = ((output_hilbert - target) ** 2).mean()
    loss_no_hilbert = ((output_no_hilbert - target) ** 2).mean()

    # Backward pass
    loss_hilbert.backward()
    loss_no_hilbert.backward()

    # Check gradients exist
    assert q.grad is not None, "Query gradients should exist"
    assert k.grad is not None, "Key gradients should exist"
    assert v.grad is not None, "Value gradients should exist"

    print("✓ Gradients computed successfully")

    # Check gradient magnitudes are reasonable
    q_grad_norm = q.grad.norm().item()
    k_grad_norm = k.grad.norm().item()
    v_grad_norm = v.grad.norm().item()

    print("\nGradient norms:")
    print(f"  Query: {q_grad_norm:.6f}")
    print(f"  Key:   {k_grad_norm:.6f}")
    print(f"  Value: {v_grad_norm:.6f}")

    # Verify gradients are not zero or exploding
    for name, grad_norm in [
        ("Query", q_grad_norm),
        ("Key", k_grad_norm),
        ("Value", v_grad_norm),
    ]:
        assert 1e-6 < grad_norm < 1e3, (
            f"{name} gradient norm {grad_norm} is out of reasonable range"
        )

    print("\n✓ All gradient checks passed!")


def test_gradient_consistency():
    """Test that Hilbert ordering preserves gradient consistency."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for this test")

    batch_size = 1
    seq_len = 512
    num_heads = 2
    head_dim = 32
    device = torch.device("cuda")

    # Create configs
    config_hilbert = RingAttentionConfig(
        segment_lengths=[128, 256, 512],
        dilation_rates=[1, 2, 4],
        use_hilbert=True,
        dropout=0.0,
    )

    config_no_hilbert = RingAttentionConfig(
        segment_lengths=[128, 256, 512],
        dilation_rates=[1, 2, 4],
        use_hilbert=False,
        dropout=0.0,
    )

    # Create modules
    attention_hilbert = HilbertRingAttention(config_hilbert, device=device)
    attention_no_hilbert = HilbertRingAttention(config_no_hilbert, device=device)

    # Set same random seed for both
    torch.manual_seed(42)

    # Multiple trials
    for trial in range(3):
        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = q.clone().detach().requires_grad_(True)  # Self-attention case
        v = q.clone().detach().requires_grad_(True)

        # Forward with Hilbert
        out_h = attention_hilbert(q, k, v)
        loss_h = out_h.sum()
        loss_h.backward()

        # Save Hilbert gradients
        q_grad_h = q.grad.clone()
        q.grad.zero_()

        # Forward without Hilbert
        out_nh = attention_no_hilbert(q, k, v)
        loss_nh = out_nh.sum()
        loss_nh.backward()

        # Compare gradient magnitudes (should be similar)
        q_grad_nh = q.grad

        grad_diff = (q_grad_h - q_grad_nh).abs().mean().item()
        grad_ratio = q_grad_h.norm() / q_grad_nh.norm()

        print(
            f"Trial {trial + 1}: Gradient diff = {grad_diff:.6f}, ratio = {grad_ratio:.4f}"
        )

        # Gradients should have similar magnitudes
        assert 0.5 < grad_ratio < 2.0, (
            f"Gradient ratio {grad_ratio} indicates potential issue"
        )

    print("\n✓ Gradient consistency test passed!")


if __name__ == "__main__":
    test_backward_pass()
    print("\n" + "=" * 50 + "\n")
    test_gradient_consistency()
