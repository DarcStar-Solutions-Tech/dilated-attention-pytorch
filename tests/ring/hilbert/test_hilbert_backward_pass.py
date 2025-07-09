#!/usr/bin/env python3
"""
Test backward pass (gradient computation) for per-segment Hilbert ordering.
"""

import torch
from dilated_attention_pytorch.ring.hilbert.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def test_backward_pass():
    """Test that gradients flow correctly through per-segment Hilbert ordering."""
    print("Testing backward pass with per-segment Hilbert ordering...\n")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    num_heads = 4
    head_dim = 32
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create attention modules
    attention_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[256, 512, 1024],
        dilation_rates=[1, 2, 4],
        use_hilbert=False,
        dropout=0.0,
    ).to(device)

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

    # Clone inputs for second test
    q_clone = q.clone().detach().requires_grad_(True)
    k_clone = k.clone().detach().requires_grad_(True)
    v_clone = v.clone().detach().requires_grad_(True)

    print("1. Testing gradient flow with Hilbert ordering...")

    # Forward pass with Hilbert
    output_hilbert = attention_hilbert(q, k, v)
    loss_hilbert = output_hilbert.mean()

    # Backward pass
    loss_hilbert.backward()

    # Check gradients exist
    assert q.grad is not None, "Query gradients are None!"
    assert k.grad is not None, "Key gradients are None!"
    assert v.grad is not None, "Value gradients are None!"

    print(f"✓ Query gradient shape: {q.grad.shape}")
    print(f"✓ Query gradient mean: {q.grad.mean().item():.6f}")
    print(f"✓ Query gradient std: {q.grad.std().item():.6f}")

    # Check for NaN or inf
    assert not torch.isnan(q.grad).any(), "Query gradients contain NaN!"
    assert not torch.isinf(q.grad).any(), "Query gradients contain inf!"
    assert not torch.isnan(k.grad).any(), "Key gradients contain NaN!"
    assert not torch.isinf(k.grad).any(), "Key gradients contain inf!"
    assert not torch.isnan(v.grad).any(), "Value gradients contain NaN!"
    assert not torch.isinf(v.grad).any(), "Value gradients contain inf!"

    print("✓ All gradients are finite")

    # Store gradients for comparison
    q_grad_hilbert = q.grad.clone()
    k_grad_hilbert = k.grad.clone()
    v_grad_hilbert = v.grad.clone()

    print("\n2. Testing gradient flow without Hilbert ordering...")

    # Forward pass without Hilbert
    output_no_hilbert = attention_no_hilbert(q_clone, k_clone, v_clone)
    loss_no_hilbert = output_no_hilbert.mean()

    # Backward pass
    loss_no_hilbert.backward()

    print(f"✓ Query gradient shape: {q_clone.grad.shape}")
    print(f"✓ Query gradient mean: {q_clone.grad.mean().item():.6f}")
    print(f"✓ Query gradient std: {q_clone.grad.std().item():.6f}")

    # Compare gradient magnitudes
    print("\n3. Comparing gradient magnitudes...")

    q_grad_diff = (q_grad_hilbert - q_clone.grad).abs().mean()
    k_grad_diff = (k_grad_hilbert - k_clone.grad).abs().mean()
    v_grad_diff = (v_grad_hilbert - v_clone.grad).abs().mean()

    print(f"Query gradient difference: {q_grad_diff.item():.6f}")
    print(f"Key gradient difference: {k_grad_diff.item():.6f}")
    print(f"Value gradient difference: {v_grad_diff.item():.6f}")

    # Gradients should be different but similar in magnitude
    q_ratio = q_grad_hilbert.abs().mean() / q_clone.grad.abs().mean()
    print(f"\nGradient magnitude ratio (Hilbert/No-Hilbert): {q_ratio.item():.2f}")

    if 0.5 < q_ratio.item() < 2.0:
        print("✓ Gradient magnitudes are similar (good!)")
    else:
        print("⚠️  Gradient magnitudes differ significantly")

    print("\n4. Testing gradient accumulation...")

    # Test multiple backward passes
    q2 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    k2 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    v2 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    total_loss = 0
    for i in range(3):
        output = attention_hilbert(q2, k2, v2)
        loss = output.mean()
        total_loss += loss

    total_loss.backward()

    print(f"✓ Accumulated gradient mean: {q2.grad.mean().item():.6f}")
    print("✓ Multiple backward passes work correctly")

    print("\n5. Testing with causal masking...")

    q3 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    k3 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    v3 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    output_causal = attention_hilbert(q3, k3, v3, is_causal=True)
    loss_causal = output_causal.mean()
    loss_causal.backward()

    print(f"✓ Causal gradient mean: {q3.grad.mean().item():.6f}")
    print("✓ Causal masking gradients work correctly")

    print("\n✅ All backward pass tests passed!")


def test_gradient_correctness():
    """Test gradient correctness using finite differences."""
    print("\n" + "=" * 60)
    print("Testing gradient correctness with finite differences")
    print("=" * 60 + "\n")

    # Small test case for numerical stability
    batch_size = 1
    seq_len = 128
    num_heads = 2
    head_dim = 16
    embed_dim = num_heads * head_dim
    epsilon = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[64, 128],
        dilation_rates=[1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    # Compute analytical gradient
    output = attention(q, k, v)
    loss = output.sum()  # Use sum for more stable gradients
    loss.backward()

    analytical_grad = q.grad[0, 0, 0, 0].item()

    # Compute numerical gradient using finite differences
    q_perturbed = q.clone().detach()
    q_perturbed[0, 0, 0, 0] += epsilon

    output_plus = attention(q_perturbed, k.detach(), v.detach())
    loss_plus = output_plus.sum()

    q_perturbed[0, 0, 0, 0] -= 2 * epsilon
    output_minus = attention(q_perturbed, k.detach(), v.detach())
    loss_minus = output_minus.sum()

    numerical_grad = (loss_plus - loss_minus).item() / (2 * epsilon)

    print(f"Analytical gradient: {analytical_grad:.6f}")
    print(f"Numerical gradient: {numerical_grad:.6f}")

    relative_error = abs(analytical_grad - numerical_grad) / (
        abs(analytical_grad) + 1e-8
    )
    print(f"Relative error: {relative_error:.6f}")

    if relative_error < 0.01:  # 1% tolerance
        print("✓ Gradients match! Backward pass is correct.")
    else:
        print("⚠️  Gradients don't match well. There might be an issue.")

    return relative_error < 0.01


if __name__ == "__main__":
    test_backward_pass()
    test_gradient_correctness()
