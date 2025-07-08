#!/usr/bin/env python3
"""Test the HilbertAttentionTritonFixed wrapper."""

import torch
from dilated_attention_pytorch.kernels import HilbertAttentionTritonFixed

# Test parameters
batch_size = 2
seq_len = 2048
num_heads = 8
head_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

# Create test tensors
q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

try:
    # Create model with wrapper
    model = HilbertAttentionTritonFixed(
        segment_lengths=[512, 1024, 2048],
        dilation_rates=[1, 2, 4],
        dropout=0.0,
        num_heads=num_heads,
        head_dim=head_dim,
    ).to(device)

    print("✅ Model created successfully")

    # Test forward pass
    with torch.no_grad():
        output = model(q, k, v)

    print("✅ Forward pass successful")
    print(f"Output shape: {output.shape}")

    # Verify output shape matches input
    assert output.shape == q.shape, f"Shape mismatch: {output.shape} != {q.shape}"
    print("✅ Output shape correct")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
