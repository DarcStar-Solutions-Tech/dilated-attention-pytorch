#!/usr/bin/env python3
"""Debug float16 issue in Triton kernels."""

import torch
import traceback


def test_float16_detailed():
    """Test float16 with detailed debugging."""
    print("=== Detailed Float16 Debug ===")

    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            HilbertAttentionCore,
        )

        # Create module
        module = HilbertAttentionCore(
            hidden_dim=64, num_heads=4, segment_size=16
        ).cuda()

        print("Module created successfully")
        print(f"qkv_proj weight dtype: {module.qkv_proj.weight.dtype}")
        print(f"out_proj weight dtype: {module.out_proj.weight.dtype}")

        # Create float16 input
        x_fp16 = torch.randn(2, 32, 64, device="cuda", dtype=torch.float16)
        print(f"\nInput dtype: {x_fp16.dtype}")

        # Try forward pass with detailed error catching
        try:
            with torch.no_grad():
                output = module(x_fp16, use_hilbert=True)
            print("Forward pass successful!")
            print(f"Output dtype: {output.dtype}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            traceback.print_exc()

        # Try with module in float16
        print("\n=== Testing with module converted to float16 ===")
        module_fp16 = module.half()
        print(
            f"qkv_proj weight dtype after .half(): {module_fp16.qkv_proj.weight.dtype}"
        )
        print(
            f"out_proj weight dtype after .half(): {module_fp16.out_proj.weight.dtype}"
        )

        try:
            with torch.no_grad():
                output = module_fp16(x_fp16, use_hilbert=True)
            print("Forward pass successful with fp16 module!")
            print(f"Output dtype: {output.dtype}")
        except Exception as e:
            print(f"Forward pass failed with fp16 module: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"Test setup failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_float16_detailed()
