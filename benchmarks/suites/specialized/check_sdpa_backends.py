#!/usr/bin/env python3
"""
Check available SDPA backends and their capabilities.
"""

import torch
import torch.nn.functional as F


def check_sdpa_backends():
    """Check which SDPA backends are available."""
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nChecking SDPA backends...")

    # Check if SDPA is available
    if hasattr(F, "scaled_dot_product_attention"):
        print("✓ scaled_dot_product_attention is available")

        # Try to check backends (PyTorch 2.0+)
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            print("\nAvailable backends:")

            # Check each backend
            backends = {
                "enable_flash": "Flash Attention",
                "enable_math": "Math (reference)",
                "enable_mem_efficient": "Memory Efficient (xFormers)",
            }

            for attr, name in backends.items():
                if hasattr(torch.backends.cuda.sdp_kernel(), attr):
                    print(f"  - {name}: Available")

        # Test SDPA with different sizes
        print("\nTesting SDPA with different sequence lengths...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for seq_len in [1024, 4096, 8192]:
            try:
                q = torch.randn(1, 8, seq_len, 64, device=device, dtype=torch.float16)
                k = torch.randn(1, 8, seq_len, 64, device=device, dtype=torch.float16)
                v = torch.randn(1, 8, seq_len, 64, device=device, dtype=torch.float16)

                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=True
                ):
                    out = F.scaled_dot_product_attention(q, k, v)

                print(f"  ✓ {seq_len} tokens: Success (shape: {out.shape})")

            except Exception as e:
                print(f"  ✗ {seq_len} tokens: {e}")
    else:
        print("✗ scaled_dot_product_attention not available")
        print("  Please upgrade to PyTorch 2.0 or later")


if __name__ == "__main__":
    check_sdpa_backends()
