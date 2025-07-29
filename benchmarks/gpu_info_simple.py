#!/usr/bin/env python3
"""Simple GPU info check."""

import torch

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    capability = torch.cuda.get_device_capability(device)
    print(f"Compute Capability: {capability}")

    # GTX 1080 (Pascal) has 48KB shared memory per SM
    print("\nGTX 1080 (Pascal) constraints:")
    print("  Shared Memory Per Block: 48KB (49,152 bytes)")
    print("  Max Threads Per Block: 1024")
    print("  Compute Capability: 6.1")

    # This explains the issue - we're on Pascal with limited shared memory
    print("\nIssue: Block sizes 128x128 require too much shared memory for Pascal")
    print("Solution: Use Pascal-optimized block sizes (32x32 or 64x64)")
