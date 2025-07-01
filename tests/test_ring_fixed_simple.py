#!/usr/bin/env python3
"""Simple test of fixed ring attention."""

import torch
from dilated_attention_pytorch.ring_dilated_attention_v2_fixed import (
    RingDilatedAttentionV2Fixed,
)

# Test parameters
seq_len = 4096
batch_size = 1
num_heads = 8
head_dim = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("Testing Fixed Ring Attention")
print(f"Device: {device}")
print(f"Sequence length: {seq_len}")

# Create model
model = RingDilatedAttentionV2Fixed(
    segment_lengths=[2048, 2048],
    dilation_rates=[1, 2],
    ring_size=1,  # Single GPU test
    device=device,
    dtype=dtype,
)

# Create input
x = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

# Test forward pass
try:
    with torch.no_grad():
        output = model(x, x, x, is_causal=True)
    print("✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
