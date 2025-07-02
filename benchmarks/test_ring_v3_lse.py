#!/usr/bin/env python3
"""
Test Ring V3 with Log-Sum-Exp accumulation.
"""

import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_lse_stability():
    """Test numerical stability of LSE implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # Use float32 to better test stability

    print("Testing Ring V3 with LSE accumulation")
    print("=" * 50)

    # Test with values that could cause instability
    seq_len = 1024
    batch_size = 2
    num_heads = 4
    head_dim = 32

    # Create model
    model = RingDilatedAttentionV3(
        segment_lengths=[256],
        dilation_rates=[1],
        device=device,
        dtype=dtype,
    )

    # Create inputs with different scales to test stability
    torch.manual_seed(42)

    # Normal scale
    q1 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k1 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v1 = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Large values that could cause overflow without LSE
    # Add large constant to test numerical stability
    large_offset = 50.0
    q2 = q1 + large_offset
    k2 = k1 + large_offset
    v2 = v1

    print("Testing with normal scale...")
    output1 = model(q1, k1, v1, is_causal=False)
    print(
        f"  Output stats: mean={output1.mean().item():.6f}, std={output1.std().item():.6f}"
    )
    print(f"  Output range: [{output1.min().item():.6f}, {output1.max().item():.6f}]")

    print("\nTesting with large offset (+50)...")
    output2 = model(q2, k2, v2, is_causal=False)
    print(
        f"  Output stats: mean={output2.mean().item():.6f}, std={output2.std().item():.6f}"
    )
    print(f"  Output range: [{output2.min().item():.6f}, {output2.max().item():.6f}]")

    # Check if outputs are similar (they should be due to softmax normalization)
    diff = (output1 - output2).abs().max().item()
    print(f"\nMax difference between outputs: {diff:.6f}")

    if diff < 1e-3:
        print("✅ LSE implementation is numerically stable!")
    else:
        print("❌ LSE implementation may have stability issues")

    # Test causal mode
    print("\nTesting causal mode...")
    output_causal = model(q1, k1, v1, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")
    print(
        f"  Output stats: mean={output_causal.mean().item():.6f}, std={output_causal.std().item():.6f}"
    )

    # Check for NaN/Inf
    if torch.isnan(output_causal).any() or torch.isinf(output_causal).any():
        print("❌ Found NaN or Inf in causal output!")
    else:
        print("✅ No NaN or Inf in causal output")

    return True


def test_multi_gpu_lse():
    """Test LSE with multiple GPUs."""
    if "RANK" not in os.environ:
        print("\nSkipping multi-GPU test (not in distributed mode)")
        return

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"\nTesting multi-GPU LSE with {world_size} GPUs")
        print("=" * 50)

    # Small test for multi-GPU
    model = RingDilatedAttentionV3(
        segment_lengths=[128],
        dilation_rates=[1],
        device=device,
        dtype=torch.float16,
        ring_size=world_size,
    )

    # Create inputs
    seq_len = 512  # Must be divisible by world_size
    q = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype)
    k = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype)
    v = torch.randn(1, seq_len, 4, 32, device=device, dtype=model.dtype)

    try:
        output = model(q, k, v, is_causal=False)

        if rank == 0:
            print(f"✅ Multi-GPU forward pass succeeded!")
            print(f"   Output shape: {output.shape}")

    except Exception as e:
        print(f"[Rank {rank}] ❌ Error: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    import os

    # Test LSE stability
    test_lse_stability()

    # Test multi-GPU if available
    test_multi_gpu_lse()
