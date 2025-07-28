#!/usr/bin/env python3
"""
Quick verification of hardware-specific tuning.
"""

import torch
from dilated_attention_pytorch.kernels import HilbertAttentionCore


def main():
    print("=== Hardware-Specific Tuning Verification ===\n")

    # Get GPU info
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU: {gpu_name}")
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
    else:
        device = torch.device("cpu")
        print("Running on CPU (limited functionality)")

    # Create module
    module = HilbertAttentionCore(
        hidden_dim=768, num_heads=12, segment_size=128, dilation_rate=2
    ).to(device)

    # Test adaptive selection thresholds
    print("\n1. Adaptive Triton Selection Thresholds:")
    print("-" * 40)

    test_lengths = [128, 256, 512, 768, 1024, 1536, 2048, 4096]

    for seq_len in test_lengths:
        uses_triton = module.should_use_triton(seq_len, device)
        print(f"Sequence {seq_len:4d}: {'Triton' if uses_triton else 'PyTorch'}")

    if hasattr(module, "_triton_threshold"):
        print(f"\nThreshold for this GPU: {module._triton_threshold}")

    # Test block size selection
    print("\n2. Optimal Block Sizes by Sequence Length:")
    print("-" * 50)
    print("Seq Len | BLOCK_M | BLOCK_N | BLOCK_D | Notes")
    print("-" * 50)

    for seq_len in test_lengths:
        M, N, D = module.get_optimal_block_sizes(seq_len, device)

        # Determine architecture note
        if capability[0] < 7:
            arch = "Pascal"
        elif capability[0] < 8:
            arch = "Volta/Turing"
        else:
            arch = "Ampere+"

        print(f"{seq_len:7d} | {M:7d} | {N:7d} | {D:7d} | {arch}")

    # Test actual forward pass
    print("\n3. Testing Forward Pass:")
    print("-" * 40)

    batch_size = 2
    test_seq = 512
    hidden_dim = 768

    x = torch.randn(batch_size, test_seq, hidden_dim, device=device)

    try:
        with torch.no_grad():
            output = module(x)
        print("✓ Forward pass successful")
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {output.shape}")

        # Check if Triton was actually used
        uses_triton = module.should_use_triton(test_seq, device)
        print(
            f"  Implementation: {'Triton kernel' if uses_triton else 'PyTorch fallback'}"
        )

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()
