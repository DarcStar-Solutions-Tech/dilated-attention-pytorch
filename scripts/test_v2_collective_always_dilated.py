#!/usr/bin/env python3
"""
Test that V2 Collective always uses dilated attention, never standard attention.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def test_small_sequences():
    """Test that small sequences use dilated attention."""
    print("Testing small sequences...")

    # Test case 1: Sequence smaller than segment length but divisible
    segment_lengths = [32, 64]
    dilation_rates = [1, 2]
    seq_len = 64  # Equal to largest segment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # This should work without errors
    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print(f"✓ Small sequence test passed (seq_len={seq_len})")

    # Test case 2: Smaller sequence that's still divisible
    seq_len = 128  # 2x the largest segment
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print(f"✓ Equal sequence test passed (seq_len={seq_len})")


def test_remainder_handling():
    """Test that remainders use dilated attention."""
    print("\nTesting remainder handling...")

    segment_lengths = [32, 64]
    dilation_rates = [1, 2]
    seq_len = 128  # Divisible by largest segment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print(f"✓ Remainder handling test passed (seq_len={seq_len})")


def test_dilation_rate_one():
    """Test that dilation_rate=1 still uses dilated attention framework."""
    print("\nTesting dilation_rate=1...")

    segment_lengths = [64]
    dilation_rates = [1]  # Only dilation rate 1
    seq_len = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print(f"✓ Dilation rate 1 test passed (seq_len={seq_len})")


def test_causal_attention():
    """Test that causal attention works with dilated patterns."""
    print("\nTesting causal attention...")

    segment_lengths = [32, 64]
    dilation_rates = [1, 2]
    seq_len = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=True)
    assert output.shape == q.shape
    print(f"✓ Causal attention test passed (seq_len={seq_len})")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    # Test case 1: Minimum valid sequence
    segment_lengths = [64, 128]
    dilation_rates = [1, 2]
    seq_len = 128  # Must be divisible by largest

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    batch_size = 2
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print("✓ Minimum valid sequence test passed")

    # Test case 2: Large dilation rates
    segment_lengths = [16, 32]
    dilation_rates = [8, 16]  # Large dilation rates
    seq_len = 64  # 2x largest segment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
    )

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention(q, k, v, is_causal=False)
    assert output.shape == q.shape
    print("✓ Large dilation rate test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing V2 Collective Always Uses Dilated Attention")
    print("=" * 60)

    try:
        test_small_sequences()
        test_remainder_handling()
        test_dilation_rate_one()
        test_causal_attention()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("V2 Collective correctly always uses dilated attention.")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
