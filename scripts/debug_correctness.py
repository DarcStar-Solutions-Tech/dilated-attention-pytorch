#!/usr/bin/env python3
"""Debug the correctness test showing 0 difference."""

import torch


def test_correctness_detailed():
    """Test correctness with more detail."""
    print("=== Detailed Correctness Test ===")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    # Create module with no dilation for comparison
    module = HilbertAttentionCore(
        hidden_dim=64,
        num_heads=4,
        segment_size=32,  # Use full sequence as one segment
        dilation_rate=1,
        use_custom_backward=False,
    ).cuda()

    # Test input
    torch.manual_seed(42)
    x = torch.randn(2, 32, 64, device="cuda")

    # Get Hilbert mapping to verify it's working
    hilbert_map = module.get_hilbert_mapping(32, x.device)
    print(f"Hilbert mapping (first 16): {hilbert_map[:16].tolist()}")

    # Compare outputs
    with torch.no_grad():
        # First, let's check if Hilbert is actually being used
        module.eval()  # Ensure no dropout

        output_hilbert = module(x, use_hilbert=True)
        output_standard = module(x, use_hilbert=False)

    # Check differences
    diff = torch.abs(output_hilbert - output_standard)
    print("\nDifference stats:")
    print(f"  Max: {diff.max().item():.6f}")
    print(f"  Mean: {diff.mean().item():.6f}")
    print(f"  Min: {diff.min().item():.6f}")
    print(f"  Num zeros: {(diff == 0).sum().item()} / {diff.numel()}")

    # Check if outputs are identical
    are_identical = torch.allclose(output_hilbert, output_standard, atol=1e-6)
    print(f"\nOutputs identical: {are_identical}")

    # Let's also check intermediate values
    print("\nOutput samples (first 5 values):")
    print(f"  Hilbert:  {output_hilbert[0, 0, :5].tolist()}")
    print(f"  Standard: {output_standard[0, 0, :5].tolist()}")

    # Test with a different configuration that should show differences
    print("\n=== Testing with smaller segments (should show differences) ===")
    module2 = HilbertAttentionCore(
        hidden_dim=64,
        num_heads=4,
        segment_size=16,  # Smaller segments
        dilation_rate=1,
        use_custom_backward=False,
    ).cuda()

    with torch.no_grad():
        module2.eval()
        output_hilbert2 = module2(x, use_hilbert=True)
        output_standard2 = module2(x, use_hilbert=False)

    diff2 = torch.abs(output_hilbert2 - output_standard2)
    print("\nWith segment_size=16:")
    print(f"  Max diff: {diff2.max().item():.6f}")
    print(f"  Mean diff: {diff2.mean().item():.6f}")


if __name__ == "__main__":
    test_correctness_detailed()
