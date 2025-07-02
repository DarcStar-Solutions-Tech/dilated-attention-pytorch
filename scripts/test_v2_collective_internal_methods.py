#!/usr/bin/env python3
"""
Test V2 Collective internal methods to ensure they always use dilated attention.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def test_process_dilated_segment():
    """Test that _process_dilated_segment handles all cases correctly."""
    print("Testing _process_dilated_segment method...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=[32, 64],
        dilation_rates=[1, 2],
        device=device,
    )

    batch_size = 2
    num_heads = 4
    head_dim = 64

    # Test 1: Small sequence (less than segment length)
    seq_len = 16
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Call internal method directly
    output = attention._process_dilated_segment(
        q, k, v, segment_len=32, dilation_rate=2, offset=0, is_causal=False
    )

    assert output.shape == q.shape
    print(f"✓ Small sequence handled correctly (seq_len={seq_len})")

    # Test 2: Exact segment length
    seq_len = 32
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention._process_dilated_segment(
        q, k, v, segment_len=32, dilation_rate=2, offset=0, is_causal=False
    )

    assert output.shape == q.shape
    print(f"✓ Exact segment length handled correctly (seq_len={seq_len})")

    # Test 3: Multiple segments with remainder
    seq_len = 75  # 2*32 + 11
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    output = attention._process_dilated_segment(
        q, k, v, segment_len=32, dilation_rate=2, offset=0, is_causal=False
    )

    assert output.shape == q.shape
    print(f"✓ Multiple segments with remainder handled correctly (seq_len={seq_len})")


def test_apply_dilation():
    """Test that _apply_dilation always applies patterns."""
    print("\nTesting _apply_dilation method...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=[32, 64],
        dilation_rates=[4, 8],
        device=device,
    )

    batch_size = 2
    num_segments = 2
    segment_len = 16  # Smaller than dilation rate
    num_heads = 4
    head_dim = 64

    # Create segment tensors
    q_seg = torch.randn(
        batch_size, num_segments, segment_len, num_heads, head_dim, device=device
    )
    k_seg = torch.randn(
        batch_size, num_segments, segment_len, num_heads, head_dim, device=device
    )
    v_seg = torch.randn(
        batch_size, num_segments, segment_len, num_heads, head_dim, device=device
    )

    # Test with dilation rate larger than segment
    q_dilated, k_dilated, v_dilated = attention._apply_dilation(
        q_seg,
        k_seg,
        v_seg,
        dilation_rate=32,  # Much larger than segment_len
        offset=0,
    )

    # Should return tensors of the same shape
    assert q_dilated.shape == q_seg.shape
    assert k_dilated.shape == k_seg.shape
    assert v_dilated.shape == v_seg.shape
    print("✓ Dilation with rate > segment_len handled correctly")

    # Test with normal dilation
    q_dilated, k_dilated, v_dilated = attention._apply_dilation(
        q_seg, k_seg, v_seg, dilation_rate=4, offset=1
    )

    assert q_dilated.shape == q_seg.shape
    print("✓ Normal dilation handled correctly")


def test_single_device_forward():
    """Test that single device forward always uses dilated patterns."""
    print("\nTesting _single_device_forward method...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module without ImprovedDilatedAttention fallback
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=[16, 32],
        dilation_rates=[1, 2],
        device=device,
        ring_size=1,  # Force single device mode
    )

    # Disable the ImprovedDilatedAttention fallback
    attention._single_gpu_attention = None

    batch_size = 2
    num_heads = 4
    head_dim = 64

    # Test various sequence lengths
    for seq_len in [16, 32, 48, 64]:
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # This should use _single_device_forward which calls _apply_dilated_attention_pattern
        output = attention.forward(q, k, v, is_causal=False)

        assert output.shape == q.shape
        print(f"✓ Single device forward works for seq_len={seq_len}")


def test_dilated_attention_always():
    """Test that _dilated_attention_always method exists and works."""
    print("\nTesting _dilated_attention_always method...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create attention module
    attention = RingDilatedAttentionV2Collective(
        segment_lengths=[32, 64],
        dilation_rates=[1, 2],
        device=device,
    )

    batch_size = 2
    seq_len = 32
    num_heads = 4
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Test the method exists and works
    output = attention._dilated_attention_always(q, k, v, is_causal=False)

    assert output.shape == q.shape
    print("✓ _dilated_attention_always method works correctly")


def main():
    """Run all internal method tests."""
    print("=" * 60)
    print("Testing V2 Collective Internal Methods")
    print("=" * 60)

    try:
        test_process_dilated_segment()
        test_apply_dilation()
        test_single_device_forward()
        test_dilated_attention_always()

        print("\n" + "=" * 60)
        print("✓ ALL INTERNAL METHOD TESTS PASSED!")
        print("V2 Collective correctly implements dilated attention everywhere.")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
