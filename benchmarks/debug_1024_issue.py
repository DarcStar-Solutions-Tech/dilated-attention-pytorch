#!/usr/bin/env python3
"""
Debug the 1024 sequence length issue.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized import (
    UnifiedHilbertAttentionOptimized,
)


def main():
    device = torch.device("cuda")
    dtype = torch.float32

    model = (
        UnifiedHilbertAttentionOptimized(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=1,
            hilbert_threshold=1024,
        )
        .to(device)
        .to(dtype)
    )
    model.eval()

    # Test 1024
    print("Testing seq_len=1024...")
    x = torch.randn(1, 1024, 768, device=device, dtype=dtype)

    try:
        with torch.no_grad():
            out = model(x, use_hilbert=True)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback

        traceback.print_exc()

    # Test without Hilbert
    print("\nTesting seq_len=1024 without Hilbert...")
    try:
        with torch.no_grad():
            out = model(x, use_hilbert=False)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed with error: {e}")

    # Check if it's using PyTorch or Triton
    print(f"\nHilbert threshold: {model.hilbert_threshold}")
    print(f"1024 > threshold: {1024 > model.hilbert_threshold}")
    print(f"Triton available: {model._triton_available}")


if __name__ == "__main__":
    main()
