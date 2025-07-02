#!/usr/bin/env python3
"""
Quick test of hybrid optimized implementation.
"""

import torch
import traceback

try:
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized import (
        RingDilatedAttentionHybridOptimized,
    )

    # Simple test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = RingDilatedAttentionHybridOptimized(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=torch.float32,
        enable_memory_pool=True,
        use_flash_attention=True,
        use_pattern_cache=True,
    )

    # Test with small input
    batch_size = 1
    seq_len = 1024
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    print("Running forward pass...")
    output = model(q, k, v)
    print(f"Success! Output shape: {output.shape}")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()
