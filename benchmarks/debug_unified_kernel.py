#!/usr/bin/env python3
"""
Debug the unified kernel issue at seq_len=1024.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention_unified import (
    UnifiedHilbertAttention,
)


def test_seq_length(seq_len, batch_size=1):
    """Test a specific sequence length."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"\nTesting seq_len={seq_len}, batch={batch_size}")

    model = (
        UnifiedHilbertAttention(
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

    # Get kernel config
    config = model._get_kernel_config(seq_len)
    print(
        f"Config: BLOCK_M={config[0]}, BLOCK_N={config[1]}, BLOCK_D={config[2]}, FUSED_SOFTMAX={config[3]}"
    )

    # Create input
    x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

    # Test without Hilbert first
    print("Testing without Hilbert...")
    try:
        with torch.no_grad():
            out = model(x, use_hilbert=False)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")

    # Test with Hilbert
    print("Testing with Hilbert...")
    try:
        with torch.no_grad():
            out = model(x, use_hilbert=True)
        print(f"Success! Output shape: {out.shape}")
    except Exception as e:
        print(f"Failed: {e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Test problematic sequence length
    test_seq_length(1024)

    # Test surrounding lengths
    test_seq_length(512)
    test_seq_length(2048)

    # Test threshold boundary
    test_seq_length(1023)
    test_seq_length(1025)


if __name__ == "__main__":
    main()
