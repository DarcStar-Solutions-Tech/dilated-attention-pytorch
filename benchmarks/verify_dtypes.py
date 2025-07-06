#!/usr/bin/env python3
"""
Verify that we're actually using FP32 in the computations.
"""

import torch
from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
    RingDilatedAttentionHybridOptimizedV2,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model with FP32
    model = RingDilatedAttentionHybridOptimizedV2(
        segment_lengths=[2048, 4096],
        dilation_rates=[8, 16],
        dropout=0.0,
        device=device,
        dtype=torch.float32,
    )

    # Create FP32 inputs
    batch_size = 1
    seq_len = 8192
    num_heads = 8
    head_dim = 64

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"Model device: {device}")
    print("Model dtype parameter: torch.float32")
    print("\nInput tensor dtypes:")
    print(f"  Q dtype: {q.dtype}")
    print(f"  K dtype: {k.dtype}")
    print(f"  V dtype: {v.dtype}")
    print("\nInput tensor info:")
    print(f"  Q shape: {q.shape}")
    print(f"  Element size: {q.element_size()} bytes")
    print(f"  Total memory per tensor: {q.numel() * q.element_size() / 1024**2:.2f} MB")

    # Run forward pass
    with torch.no_grad():
        output = model(q, k, v)

    print(f"\nOutput dtype: {output.dtype}")
    print(f"Output shape: {output.shape}")

    # Calculate theoretical memory usage
    total_elements = 3 * q.numel()  # Q, K, V
    memory_mb = total_elements * 4 / 1024**2  # 4 bytes for float32
    print(f"\nTheoretical memory for Q,K,V: {memory_mb:.2f} MB")

    # Check actual memory usage
    if device == "cuda":
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"Actual GPU memory allocated: {allocated:.2f} MB")


if __name__ == "__main__":
    main()
