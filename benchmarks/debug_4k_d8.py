#!/usr/bin/env python3
"""
Debug the 4K d=8 error.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def main():
    print("=== Debugging 4K d=8 Error ===")

    # Parameters
    seq_len = 4096
    dilation_rate = 8
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    print(f"Configuration: {seq_len} tokens, dilation={dilation_rate}")
    print(f"Effective length: {seq_len // dilation_rate}")

    try:
        # Create Enhanced model
        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                enable_4k_optimization=True,
            )
            .cuda()
            .eval()
        )

        # Get config
        config = enhanced._get_optimal_config(seq_len)
        print(f"\nConfig returned: {config}")

        # Check if all required keys are present
        required_keys = [
            "block_m",
            "block_n",
            "block_d",
            "num_warps",
            "use_fused_softmax",
            "rows_per_block",
            "fused_block_n",
            "enable_prefetch",
        ]

        for key in required_keys:
            if key not in config:
                print(f"ERROR: Missing key '{key}' in config!")
            else:
                print(f"  {key}: {config[key]}")

        # Try to create input and forward
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )

        print("\nTrying forward pass...")
        with torch.no_grad():
            out = enhanced(x)

        print("Forward pass successful!")
        print(f"Output shape: {out.shape}")

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
