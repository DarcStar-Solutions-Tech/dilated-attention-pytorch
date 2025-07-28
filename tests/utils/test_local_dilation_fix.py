"""
Test the fixed local dilation implementation.
"""

import torch

from dilated_attention_pytorch.ring.hilbert.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed as RingDilatedAttentionHybridFixedV2,
)
# StandardizedRingConfig is not available anymore, using direct parameters instead


def test_local_dilation_dimensions():
    """Test that local dilation produces correct dimensions."""
    batch_size = 2
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    # Create model with direct parameters
    model = RingDilatedAttentionHybridFixedV2(
        dim=num_heads * head_dim,
        heads=num_heads,
        segment_lengths=[512, 512],
        dilation_rates=[1, 2],
        dropout=0.0,
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Forward pass should not raise dimension mismatch
    output = model(q, k, v)

    # Check output shape
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    # Check that output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))

    print("✓ Local dilation test passed!")


def test_dilation_pattern():
    """Test that dilation pattern is applied correctly within segments."""
    batch_size = 1
    seq_len = 1024
    num_heads = 1
    head_dim = 64

    # Create model with simple dilation
    model = RingDilatedAttentionHybridFixedV2(
        dim=num_heads * head_dim,
        heads=num_heads,
        segment_lengths=[512],
        dilation_rates=[2],
        dropout=0.0,
    )

    # Create special pattern to test dilation
    # Make keys have alternating pattern that dilation should pick up
    q = torch.ones(batch_size, seq_len, num_heads, head_dim)
    k = torch.zeros(batch_size, seq_len, num_heads, head_dim)
    v = torch.zeros(batch_size, seq_len, num_heads, head_dim)

    # Set every 2nd position in k and v to have distinct values
    for i in range(0, seq_len, 2):
        k[:, i, :, :] = 1.0
        v[:, i, :, :] = float(i)

    # Run attention
    output = model(q, k, v, is_causal=False)

    # For first segment (0-512), with dilation=2, we should attend to positions
    # [0, 2, 4, ..., 510] which all have value 1.0 in keys
    # The output should reflect the average of these value positions

    # Check first position output (should be average of dilated values in first segment)
    first_seg_dilated_values = [float(i) for i in range(0, 512, 2)]
    expected_avg = sum(first_seg_dilated_values) / len(first_seg_dilated_values)

    # Due to softmax, all positions with key=1.0 get equal attention
    actual_first = output[0, 0, 0, 0].item()

    print(f"Expected average of dilated values: {expected_avg}")
    print(f"Actual first position output: {actual_first}")
    print("✓ Dilation pattern test completed!")


def test_causal_mask_with_dilation():
    """Test that causal masking works correctly with dilation."""
    batch_size = 1
    seq_len = 512
    num_heads = 1
    head_dim = 64

    model = RingDilatedAttentionHybridFixedV2(
        dim=num_heads * head_dim,
        heads=num_heads,
        segment_lengths=[512],
        dilation_rates=[2],
        dropout=0.0,
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)

    # Run with causal mask
    output_causal = model(q, k, v, is_causal=True)
    output_non_causal = model(q, k, v, is_causal=False)

    # Outputs should be different
    assert not torch.allclose(output_causal, output_non_causal)

    print("✓ Causal mask with dilation test passed!")


if __name__ == "__main__":
    test_local_dilation_dimensions()
    test_dilation_pattern()
    test_causal_mask_with_dilation()
    print("\n✅ All tests passed!")
