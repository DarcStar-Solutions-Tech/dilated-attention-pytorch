#!/usr/bin/env python3
"""
Simple gradient test focusing on the core functionality.
"""

import torch
import torch.nn as nn
from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def test_simple_gradient_flow():
    """Test basic gradient flow through Hilbert attention."""
    print("Testing simple gradient flow...\n")

    # Very simple test case
    batch_size = 1
    seq_len = 256
    num_heads = 4
    head_dim = 32
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention with Hilbert
    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[128, 256],
        dilation_rates=[1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    # Simple test: can we backprop through it?
    for test_name in ["Basic", "With Causal Mask"]:
        print(f"Test: {test_name}")

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

        # Forward
        if test_name == "Basic":
            output = attention(q, k, v)
        else:
            output = attention(q, k, v, is_causal=True)

        # Create a simple loss
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)

        # Backward
        loss.backward()

        # Check gradients
        print(f"  Loss: {loss.item():.6f}")
        print(f"  Q grad norm: {q.grad.norm().item():.6f}")
        print(f"  K grad norm: {k.grad.norm().item():.6f}")
        print(f"  V grad norm: {v.grad.norm().item():.6f}")

        # Verify gradients are reasonable
        for name, grad in [("Q", q.grad), ("K", k.grad), ("V", v.grad)]:
            assert grad is not None, f"{name} gradient is None!"
            assert not torch.isnan(grad).any(), f"{name} gradient contains NaN!"
            assert not torch.isinf(grad).any(), f"{name} gradient contains inf!"
            assert grad.norm().item() > 0, f"{name} gradient is zero!"

        print("  ✓ All gradients look good!\n")


def test_gradient_consistency():
    """Test that gradients are consistent across multiple runs."""
    print("Testing gradient consistency...\n")

    batch_size = 1
    seq_len = 256
    num_heads = 4
    head_dim = 32
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed seed for reproducibility
    torch.manual_seed(42)

    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[128, 256],
        dilation_rates=[1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    # Run twice with same inputs
    grad_norms = []

    for run in range(2):
        # Same inputs each time
        torch.manual_seed(123)
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, requires_grad=True
        )

        output = attention(q, k, v)
        loss = output.sum()
        loss.backward()

        grad_norm = q.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"Run {run + 1}: Gradient norm = {grad_norm:.6f}")

    # Check consistency
    diff = abs(grad_norms[0] - grad_norms[1])
    print(f"\nDifference between runs: {diff:.8f}")

    if diff < 1e-5:
        print("✓ Gradients are consistent!")
    else:
        print("⚠️  Gradients are inconsistent!")


def test_hilbert_ordering_gradient_impact():
    """Test how Hilbert ordering affects gradients."""
    print("\nTesting Hilbert ordering impact on gradients...\n")

    batch_size = 2
    seq_len = 512
    num_heads = 4
    head_dim = 32
    embed_dim = num_heads * head_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same architecture, different Hilbert settings
    attention_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[256, 512],
        dilation_rates=[1, 2],
        use_hilbert=True,
        dropout=0.0,
    ).to(device)

    attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=[256, 512],
        dilation_rates=[1, 2],
        use_hilbert=False,
        dropout=0.0,
    ).to(device)

    # Same inputs
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Forward and backward with Hilbert
    q1 = q.clone().requires_grad_(True)
    k1 = k.clone().requires_grad_(True)
    v1 = v.clone().requires_grad_(True)

    output1 = attention_hilbert(q1, k1, v1)
    loss1 = output1.mean()
    loss1.backward()

    # Forward and backward without Hilbert
    q2 = q.clone().requires_grad_(True)
    k2 = k.clone().requires_grad_(True)
    v2 = v.clone().requires_grad_(True)

    output2 = attention_no_hilbert(q2, k2, v2)
    loss2 = output2.mean()
    loss2.backward()

    # Compare
    print("With Hilbert:")
    print(f"  Output mean: {output1.mean().item():.6f}")
    print(f"  Q grad norm: {q1.grad.norm().item():.6f}")

    print("\nWithout Hilbert:")
    print(f"  Output mean: {output2.mean().item():.6f}")
    print(f"  Q grad norm: {q2.grad.norm().item():.6f}")

    # Gradient similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        q1.grad.flatten(), q2.grad.flatten(), dim=0
    ).item()

    print(f"\nGradient cosine similarity: {cosine_sim:.4f}")

    if cosine_sim > 0.9:
        print("✓ Gradients are highly similar (good!)")
    elif cosine_sim > 0.7:
        print("✓ Gradients are reasonably similar")
    else:
        print("⚠️  Gradients are quite different")


if __name__ == "__main__":
    test_simple_gradient_flow()
    test_gradient_consistency()
    test_hilbert_ordering_gradient_impact()
