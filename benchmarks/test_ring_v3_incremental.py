#!/usr/bin/env python3
"""
Test Ring V3 with incrementally larger sequences to find performance cliff.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_incremental.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_incremental():
    """Test with incrementally larger sequences."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_incremental.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Ring V3 with Incremental Sequence Lengths")
        print("=" * 50)

    # Test sequences from small to large
    seq_lengths = [256, 512, 1024, 2048, 4096]

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nTesting seq_len={seq_len}")

        dist.barrier()

        # Create model with appropriate bucket size
        bucket_size = min(256, seq_len // 4)

        model = RingDilatedAttentionV3(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            bucket_size=bucket_size,
            use_bucketed=True,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        # Create inputs
        torch.manual_seed(42)
        scale = 0.1 / (seq_len**0.25)  # Adaptive scaling
        q = torch.randn(1, seq_len, 4, 32, device=device) * scale
        k = torch.randn(1, seq_len, 4, 32, device=device) * scale
        v = torch.randn(1, seq_len, 4, 32, device=device) * scale

        # Time the forward pass
        try:
            # Warmup
            _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Timed run
            dist.barrier()
            start = time.time()

            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            elapsed = time.time() - start

            # Check output
            has_nan = torch.isnan(output).any().item()
            output_mean = output.mean().item()

            # Gather results
            results = {
                "elapsed": elapsed,
                "has_nan": has_nan,
                "output_mean": output_mean,
            }

            all_results = [None] * world_size
            dist.all_gather_object(all_results, results)

            if rank == 0:
                avg_time = sum(r["elapsed"] for r in all_results) / len(all_results)
                any_nan = any(r["has_nan"] for r in all_results)

                status = "❌ NaN" if any_nan else "✅"
                print(f"  {status} Time: {avg_time:.3f}s, Mean: {output_mean:.6f}")

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Failed: {e}")
            break

        # Clean up
        del model, q, k, v
        torch.cuda.empty_cache()

    # Test comparison: bucketed vs non-bucketed at 1024 tokens
    if rank == 0:
        print("\n\nComparing Bucketed vs Non-bucketed at 1024 tokens:")
        print("-" * 40)

    seq_len = 1024

    for use_bucketed, name in [(False, "Non-bucketed"), (True, "Bucketed")]:
        dist.barrier()

        model = RingDilatedAttentionV3(
            segment_lengths=[512],
            dilation_rates=[1],
            bucket_size=256 if use_bucketed else None,
            use_bucketed=use_bucketed,
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
        )

        q = torch.randn(1, seq_len, 4, 32, device=device) * 0.05
        k = torch.randn(1, seq_len, 4, 32, device=device) * 0.05
        v = torch.randn(1, seq_len, 4, 32, device=device) * 0.05

        try:
            start = time.time()
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            if rank == 0:
                print(f"{name}: {elapsed:.3f}s")

        except Exception as e:
            if rank == 0:
                print(f"{name}: Failed - {e}")

    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Testing completed")


if __name__ == "__main__":
    test_incremental()
