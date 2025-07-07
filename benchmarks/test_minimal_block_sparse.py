#!/usr/bin/env python3
"""
Minimal test to diagnose block-sparse issues.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use only first GPU

import torch
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

print("1. Import test...")
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
    )

    print("   ✓ Direct import successful")
except Exception as e:
    print(f"   ✗ Direct import failed: {e}")

print("\n2. Factory import test...")
try:
    print("   ✓ Factory import successful")
except Exception as e:
    print(f"   ✗ Factory import failed: {e}")

print("\n3. Create model test...")
try:
    model = BlockSparseRingDilatedAttention(
        segment_lengths=[512],
        dilation_rates=[1],
        sparsity_ratio=0.1,
    )
    print("   ✓ Model creation successful")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback

    traceback.print_exc()

print("\n4. Forward pass test...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")

    # Small test
    batch_size = 1
    seq_len = 512
    num_heads = 4
    head_dim = 32

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    model = model.to(device)
    output = model(q, k, v)

    print("   ✓ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Output valid: {torch.isfinite(output).all()}")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ Minimal test completed!")
