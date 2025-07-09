#!/usr/bin/env python3
"""
Compare gradient correctness between Hilbert and non-Hilbert versions.
"""

import torch
from dilated_attention_pytorch.ring.hilbert.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def gradient_check(attention_module, q, k, v, epsilon=1e-4):
    """Check gradient correctness using finite differences."""
    # Compute analytical gradient
    q_test = q.clone().detach().requires_grad_(True)
    k_test = k.clone().detach()
    v_test = v.clone().detach()

    output = attention_module(q_test, k_test, v_test)
    loss = output.sum()
    loss.backward()

    analytical_grad = q_test.grad[0, 0, 0, 0].item()

    # Compute numerical gradient
    q_plus = q.clone()
    q_plus[0, 0, 0, 0] += epsilon
    output_plus = attention_module(q_plus, k, v)
    loss_plus = output_plus.sum().item()

    q_minus = q.clone()
    q_minus[0, 0, 0, 0] -= epsilon
    output_minus = attention_module(q_minus, k, v)
    loss_minus = output_minus.sum().item()

    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    relative_error = abs(analytical_grad - numerical_grad) / (
        abs(analytical_grad) + 1e-8
    )

    return analytical_grad, numerical_grad, relative_error


def test_gradient_comparison():
    """Compare gradients between Hilbert and non-Hilbert versions."""
    print("Comparing gradient correctness: Hilbert vs Non-Hilbert\n")

    # Small test case
    batch_size = 1
    seq_len = 128
    num_heads = 2
    head_dim = 16
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create modules
    attention_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[64, 128],
        dilation_rates=[1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[64, 128],
        dilation_rates=[1, 2],
        use_hilbert=False,
        dropout=0.0,
    ).to(device)

    # Create inputs
    torch.manual_seed(42)  # For reproducibility
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print("1. Testing without Hilbert ordering:")
    analytical_no_h, numerical_no_h, error_no_h = gradient_check(
        attention_no_hilbert, q, k, v
    )
    print(f"   Analytical: {analytical_no_h:.6f}")
    print(f"   Numerical:  {numerical_no_h:.6f}")
    print(f"   Error:      {error_no_h:.6f} ({error_no_h * 100:.2f}%)")

    print("\n2. Testing with Hilbert ordering:")
    analytical_h, numerical_h, error_h = gradient_check(attention_hilbert, q, k, v)
    print(f"   Analytical: {analytical_h:.6f}")
    print(f"   Numerical:  {numerical_h:.6f}")
    print(f"   Error:      {error_h:.6f} ({error_h * 100:.2f}%)")

    # Test if outputs are differentiable
    print("\n3. Testing differentiability:")

    q_test = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    k_test = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )
    v_test = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    # Test Hilbert version
    output_h = attention_hilbert(q_test, k_test, v_test)

    # Check if output requires grad
    print(f"   Output requires_grad (Hilbert): {output_h.requires_grad}")

    # Try second-order gradients
    loss = output_h.sum()
    grads = torch.autograd.grad(loss, [q_test, k_test, v_test], create_graph=True)

    # Check if gradients are differentiable
    second_loss = sum(g.sum() for g in grads)
    try:
        _ = torch.autograd.grad(second_loss, [q_test, k_test, v_test])
        print("   ✓ Second-order gradients work!")
    except Exception as e:
        print(f"   ✗ Second-order gradients failed: {e}")

    # Test gradient flow through multiple operations
    print("\n4. Testing gradient flow through operations:")

    # Chain multiple attention operations
    x = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
    )

    # Forward through attention
    y = attention_hilbert(x, x, x)

    # Apply some operations
    z = y.mean(dim=-1, keepdim=True)
    z = z * 2.0 + 1.0

    # Backward
    loss = z.sum()
    loss.backward()

    print(f"   Input gradient norm: {x.grad.norm().item():.6f}")
    print(f"   Input gradient mean: {x.grad.mean().item():.6f}")
    print(f"   Input gradient std:  {x.grad.std().item():.6f}")

    if x.grad.norm().item() > 0:
        print("   ✓ Gradients flow correctly through operations")
    else:
        print("   ✗ No gradients detected!")


if __name__ == "__main__":
    test_gradient_comparison()
