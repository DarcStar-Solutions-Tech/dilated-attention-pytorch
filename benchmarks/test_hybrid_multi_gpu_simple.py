#!/usr/bin/env python3
"""
Simple test to verify hybrid implementation works on multiple GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/test_hybrid_multi_gpu_simple.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def test_hybrid_basic():
    """Basic test of hybrid implementation."""

    if "RANK" not in os.environ:
        print("Testing single GPU mode...")
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing Hybrid implementation with world_size={world_size}")

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        enable_memory_pool=True,
        use_pattern_cache=True,
        use_flash_attention=False,  # Start without flash
    )

    # Test configurations
    test_cases = [
        (512, "Small test"),
        (1024, "Medium test"),
    ]

    for seq_len, name in test_cases:
        print(f"\n[Rank {rank}] {name} (seq_len={seq_len}):")

        # Create inputs
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

        # Forward pass
        try:
            output = model(q, k, v, is_causal=False)

            # Check output
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_mean = output.mean().item()
            output_std = output.std().item()

            print("  ✅ Success!")
            print(f"     Output shape: {output.shape}")
            print(f"     Mean: {output_mean:.6f}, Std: {output_std:.6f}")
            print(f"     NaN: {has_nan}, Inf: {has_inf}")

            # Memory usage
            if device.type == "cuda":
                mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
                print(f"     Memory: {mem_mb:.1f} MB")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback

            traceback.print_exc()

    # Test with dilation > 1
    if world_size > 1:
        print(f"\n[Rank {rank}] Testing with dilation > 1:")

        model_dilated = RingDilatedAttentionHybrid(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
        )

        q = torch.randn(1, 1024, 8, 32, device=device) * 0.1
        k = torch.randn(1, 1024, 8, 32, device=device) * 0.1
        v = torch.randn(1, 1024, 8, 32, device=device) * 0.1

        try:
            output = model_dilated(q, k, v, is_causal=False)
            print(f"  ✅ Dilation test passed! Output shape: {output.shape}")
        except Exception as e:
            print(f"  ❌ Dilation test failed: {e}")

    if world_size > 1:
        dist.destroy_process_group()

    print(f"\n[Rank {rank}] All tests completed!")


if __name__ == "__main__":
    test_hybrid_basic()
