#!/usr/bin/env python3
"""
Proper test for Ring V3 LSE numerical stability.
"""

import torch
import torch.nn.functional as F
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.ring_attention_lse import compute_attention_with_lse


def test_lse_vs_naive():
    """Compare LSE implementation against naive softmax."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("Testing LSE vs Naive Softmax Implementation")
    print("=" * 50)

    # Create test tensors
    batch, heads, seq, dim = 2, 4, 128, 32

    # Generate tensors with large values that could cause overflow
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 10
    k = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype) * 10
    v = torch.randn(batch, heads, seq, dim, device=device, dtype=dtype)

    # Compute with LSE method
    output_lse, lse = compute_attention_with_lse(
        q, k, v, scale=1.0 / (dim**0.5), mask=None, dropout=0.0, training=False
    )

    # Compute with naive method (could overflow)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (dim**0.5)

    # Check for overflow in naive method
    if torch.isinf(scores.exp()).any():
        print("✅ Naive method would overflow (as expected)")

        # Use stable computation for comparison
        max_scores = scores.amax(dim=-1, keepdim=True)
        stable_scores = scores - max_scores
        attn_weights = F.softmax(stable_scores, dim=-1)
        output_stable = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        diff = (output_lse - output_stable).abs().max().item()
        print(f"Max difference (LSE vs stable softmax): {diff:.6e}")

        if diff < 1e-5:
            print("✅ LSE produces same result as stable softmax")
        else:
            print("❌ LSE differs from stable softmax")
    else:
        # If no overflow, compare directly
        attn_weights = F.softmax(scores, dim=-1)
        output_naive = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        diff = (output_lse - output_naive).abs().max().item()
        print(f"Max difference (LSE vs naive): {diff:.6e}")

        if diff < 1e-5:
            print("✅ LSE matches naive implementation")


def test_ring_v3_consistency():
    """Test that Ring V3 produces consistent results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n\nTesting Ring V3 Consistency")
    print("=" * 50)

    # Create model
    model = RingDilatedAttentionV3(
        segment_lengths=[128],
        dilation_rates=[1],
        device=device,
        dtype=torch.float32,
    )

    # Test inputs
    torch.manual_seed(42)
    seq_len = 256
    q = torch.randn(1, seq_len, 4, 32, device=device)
    k = torch.randn(1, seq_len, 4, 32, device=device)
    v = torch.randn(1, seq_len, 4, 32, device=device)

    # Run multiple times to check consistency
    outputs = []
    for i in range(3):
        output = model(q, k, v, is_causal=False)
        outputs.append(output)
        print(
            f"Run {i + 1}: mean={output.mean().item():.6f}, std={output.std().item():.6f}"
        )

    # Check consistency
    for i in range(1, len(outputs)):
        diff = (outputs[0] - outputs[i]).abs().max().item()
        print(f"Difference between run 1 and run {i + 1}: {diff:.6e}")

        if diff < 1e-6:
            print(f"  ✅ Consistent")
        else:
            print(f"  ❌ Inconsistent!")

    # Test with causal mask
    print("\nTesting causal mode consistency...")
    output_causal1 = model(q, k, v, is_causal=True)
    output_causal2 = model(q, k, v, is_causal=True)

    diff_causal = (output_causal1 - output_causal2).abs().max().item()
    print(f"Causal mode difference: {diff_causal:.6e}")

    if diff_causal < 1e-6:
        print("✅ Causal mode is consistent")
    else:
        print("❌ Causal mode is inconsistent!")


def test_extreme_values():
    """Test with extreme values to verify stability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n\nTesting Extreme Values")
    print("=" * 50)

    model = RingDilatedAttentionV3(
        segment_lengths=[64],
        dilation_rates=[1],
        device=device,
        dtype=torch.float32,
    )

    seq_len = 128

    # Test 1: Very large values
    print("Test 1: Very large values...")
    q_large = torch.randn(1, seq_len, 4, 32, device=device) * 100
    k_large = torch.randn(1, seq_len, 4, 32, device=device) * 100
    v_large = torch.randn(1, seq_len, 4, 32, device=device)

    try:
        output_large = model(q_large, k_large, v_large, is_causal=False)
        if torch.isnan(output_large).any() or torch.isinf(output_large).any():
            print("  ❌ Output contains NaN or Inf!")
        else:
            print(f"  ✅ Output is stable: mean={output_large.mean().item():.6f}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

    # Test 2: Very small values
    print("\nTest 2: Very small values...")
    q_small = torch.randn(1, seq_len, 4, 32, device=device) * 1e-6
    k_small = torch.randn(1, seq_len, 4, 32, device=device) * 1e-6
    v_small = torch.randn(1, seq_len, 4, 32, device=device)

    output_small = model(q_small, k_small, v_small, is_causal=False)
    if torch.isnan(output_small).any():
        print("  ❌ Output contains NaN!")
    else:
        print(f"  ✅ Output is stable: mean={output_small.mean().item():.6f}")

    # Test 3: Mixed scales
    print("\nTest 3: Mixed scales...")
    q_mixed = torch.randn(1, seq_len, 4, 32, device=device)
    q_mixed[0, :64] *= 100  # First half large
    k_mixed = torch.randn(1, seq_len, 4, 32, device=device)
    k_mixed[0, 64:] *= 100  # Second half large
    v_mixed = torch.randn(1, seq_len, 4, 32, device=device)

    output_mixed = model(q_mixed, k_mixed, v_mixed, is_causal=False)
    if torch.isnan(output_mixed).any() or torch.isinf(output_mixed).any():
        print("  ❌ Output contains NaN or Inf!")
    else:
        print(f"  ✅ Output is stable: mean={output_mixed.mean().item():.6f}")


if __name__ == "__main__":
    test_lse_vs_naive()
    test_ring_v3_consistency()
    test_extreme_values()
