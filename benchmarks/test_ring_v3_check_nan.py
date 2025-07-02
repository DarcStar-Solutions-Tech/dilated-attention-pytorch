#!/usr/bin/env python3
"""
Check where NaN values are introduced in Ring V3.
Run with: torchrun --nproc_per_node=2 test_ring_v3_check_nan.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def check_nan_source():
    """Find where NaN values are introduced."""

    if "RANK" not in os.environ:
        print("Single GPU test mode")
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Multi-GPU mode
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Testing with world_size={world_size}")

    # Test configurations
    configs = [
        (64, 32, "Tiny"),
        (256, 128, "Small"),
        (512, 256, "Medium"),
        (1024, 256, "Large"),
    ]

    for seq_len, bucket_size, name in configs:
        print(f"\n[Rank {rank}] Testing {name} (seq_len={seq_len}):")

        # Create model
        model = RingDilatedAttentionV3(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            bucket_size=bucket_size,
            use_bucketed=True,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Create inputs with controlled values
        torch.manual_seed(42)
        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1  # Small values
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

        # Check inputs
        print(f"  Q stats: mean={q.mean().item():.6f}, std={q.std().item():.6f}")
        print(f"  K stats: mean={k.mean().item():.6f}, std={k.std().item():.6f}")
        print(f"  V stats: mean={v.mean().item():.6f}, std={v.std().item():.6f}")

        # Forward pass
        try:
            output = model(q, k, v, is_causal=False)

            # Check output
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            if has_nan or has_inf:
                print(f"  ❌ Output has NaN: {has_nan}, Inf: {has_inf}")

                # Find where NaN first appears
                nan_mask = torch.isnan(output)
                if nan_mask.any():
                    nan_indices = torch.where(nan_mask)
                    print(
                        f"     First NaN at: {[idx[0].item() for idx in nan_indices]}"
                    )
            else:
                output_mean = output.mean().item()
                output_std = output.std().item()
                print(f"  ✅ Output OK: mean={output_mean:.6f}, std={output_std:.6f}")

        except Exception as e:
            print(f"  ❌ Exception: {e}")

        # Clean up
        del q, k, v, model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Test without bucketing
    print(f"\n[Rank {rank}] Testing without bucketing:")
    model = RingDilatedAttentionV3(
        segment_lengths=[256],
        dilation_rates=[1],
        use_bucketed=False,  # Disable bucketing
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
    )

    q = torch.randn(1, 512, 4, 32, device=device) * 0.1
    k = torch.randn(1, 512, 4, 32, device=device) * 0.1
    v = torch.randn(1, 512, 4, 32, device=device) * 0.1

    output = model(q, k, v, is_causal=False)
    has_nan = torch.isnan(output).any().item()

    if has_nan:
        print("  ❌ Non-bucketed also produces NaN")
    else:
        print(f"  ✅ Non-bucketed works: mean={output.mean().item():.6f}")

    if "RANK" in os.environ:
        dist.destroy_process_group()


if __name__ == "__main__":
    check_nan_source()
