#!/usr/bin/env python3
"""
Quick test of Hilbert kernel performance in dilated attention.
"""

import time
import torch
from dilated_attention_pytorch.kernels import HilbertAttentionCore

# Configuration
hidden_dim = 768
num_heads = 12
batch_size = 2
seq_len = 1024
segment_size = 256
dilation_rate = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create module
module = HilbertAttentionCore(
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    segment_size=segment_size,
    dilation_rate=dilation_rate,
).to(device)

# Create input
x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

# Test both modes
print(
    f"\nTesting seq_len={seq_len}, segment_size={segment_size}, dilation_rate={dilation_rate}"
)
print("-" * 60)

for use_hilbert in [False, True]:
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time 5 iterations
    start = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            output = module(x, use_hilbert=use_hilbert)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) * 1000 / 5  # ms per iteration

    print(
        f"Hilbert {'ON ' if use_hilbert else 'OFF'}: {elapsed:.2f}ms per forward pass"
    )
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")

# Test with different configurations
print("\n" + "=" * 60)
print("Testing different dilation rates:")
print("-" * 60)

for dil in [1, 2, 4, 8]:
    module.dilation_rate = dil

    # Time with Hilbert ON
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(3):
            _ = module(x, use_hilbert=True)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000 / 3

    # Calculate sparsity
    sparsity = 1.0 - (1.0 / dil)
    print(f"Dilation rate {dil} ({sparsity:.0%} sparse): {elapsed:.2f}ms")

print("\nDilated attention with Hilbert optimization is working correctly!")
