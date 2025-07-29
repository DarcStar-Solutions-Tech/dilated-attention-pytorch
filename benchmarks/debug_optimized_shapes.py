#!/usr/bin/env python3
"""
Debug shape issues in optimized kernel.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized import (
    UnifiedHilbertAttentionOptimized,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")

    # Check head dimension
    hidden_dim = 768
    num_heads = 12
    head_dim = hidden_dim // num_heads
    print(
        f"\nDimensions: hidden_dim={hidden_dim}, num_heads={num_heads}, head_dim={head_dim}"
    )

    # Create model
    model = (
        UnifiedHilbertAttentionOptimized(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
            hilbert_threshold=1024,
        )
        .to(device)
        .to(dtype)
    )

    # Check configurations
    print("\nKernel configurations:")
    for seq_len in [512, 1024, 2048, 4096]:
        config = model._get_kernel_config(seq_len)
        print(
            f"seq_len={seq_len}: BLOCK_M={config[0]}, BLOCK_N={config[1]}, BLOCK_D={config[2]} (head_dim={head_dim})"
        )

    # Test specific case
    print("\nTesting seq_len=2048...")
    x = torch.randn(1, 2048, 768, device=device, dtype=dtype)

    # Process through QKV
    qkv = model.qkv_proj(x)
    qkv = qkv.reshape(1, 2048, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    print(f"Q shape: {q.shape}")
    print(f"K shape: {k.shape}")
    print(f"V shape: {v.shape}")

    # Get kernel config
    BLOCK_M, BLOCK_N, BLOCK_D, USE_FUSED_SOFTMAX, ENABLE_PREFETCH = (
        model._get_kernel_config(2048)
    )
    print(f"\nFor seq_len=2048: BLOCK_D={BLOCK_D}, actual head_dim={head_dim}")
    print(f"BLOCK_D < head_dim: {BLOCK_D < head_dim}")


if __name__ == "__main__":
    main()
