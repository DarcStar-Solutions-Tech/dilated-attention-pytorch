#!/usr/bin/env python3
"""Simple demo of SDPA with dilated patterns - no timeouts."""

import torch
import torch.nn.functional as F
import time

# Configuration
seq_len = 1024
num_heads = 8
head_dim = 64
segment_length = 256
dilation_rate = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

print("SDPA Dilated Attention Demo")
print("=" * 40)
print(f"Device: {device}")
print(f"Sequence length: {seq_len}")
print(f"Segment length: {segment_length}")
print(f"Dilation rate: {dilation_rate}")

# Create simple dilated mask
mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
num_segments = seq_len // segment_length

for seg in range(num_segments):
    start = seg * segment_length
    end = start + segment_length
    for i in range(start, end, dilation_rate):
        for j in range(start, end, dilation_rate):
            mask[i, j] = 1.0

# Convert to attention mask
attn_mask = torch.where(mask == 1, 0.0, float("-inf"))
attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

# Create random inputs
batch_size = 2
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Run SDPA
torch.cuda.synchronize() if device.type == "cuda" else None
start = time.perf_counter()

output = F.scaled_dot_product_attention(
    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
)

torch.cuda.synchronize() if device.type == "cuda" else None
elapsed = time.perf_counter() - start

print("\nResults:")
print(f"Output shape: {output.shape}")
print(f"Time: {elapsed * 1000:.2f}ms")

# Calculate sparsity
sparsity = (mask == 0).float().mean().item()
print("\nPattern stats:")
print(f"Sparsity: {sparsity * 100:.1f}%")
print(f"Active connections: {(1 - sparsity) * 100:.1f}%")
print(f"Memory reduction: ~{sparsity:.1%}")
