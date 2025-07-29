#!/usr/bin/env python3
"""Verify which kernel is being used for sparse patterns."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def verify_kernel_usage():
    """Check which kernel path is taken."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Kernel Usage Verification")
    print("=" * 60)

    # Test with dilation_rate > 1
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,  # Should trigger sparse kernel
        dropout=0.0,
        hilbert_threshold=100,
    ).to(device)

    print("Module created with dilation_rate=4")
    print(f"Triton available: {module._triton_available}")

    # Check module configuration
    x = torch.randn(1, 2048, 768, device=device)

    # Add some debug prints to understand the flow
    print(f"\nInput shape: {x.shape}")
    print(f"Will use Hilbert: {2048 > module.hilbert_threshold}")
    print(f"Should use sparse kernel: dilation_rate={module.dilation_rate} > 1")

    # Let's also check what happens with dilation_rate=1
    module2 = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,  # Should use standard kernel
        dropout=0.0,
        hilbert_threshold=100,
    ).to(device)

    print("\nModule2 created with dilation_rate=1")
    print(f"Should use standard kernel: dilation_rate={module2.dilation_rate} == 1")


if __name__ == "__main__":
    verify_kernel_usage()
