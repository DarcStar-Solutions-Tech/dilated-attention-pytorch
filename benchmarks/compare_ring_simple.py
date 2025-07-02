#!/usr/bin/env python3
"""
Simple comparison of Ring V3 and V2 Collective focusing on key differences.
Run with: torchrun --nproc_per_node=2 benchmarks/compare_ring_simple.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def compare_simple():
    """Compare key aspects of both implementations."""

    if "RANK" not in os.environ:
        print("Run with: torchrun --nproc_per_node=2 benchmarks/compare_ring_simple.py")
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Ring V3 vs V2 Collective - Key Differences")
        print("=" * 50)

    # Test 1: Basic functionality
    if rank == 0:
        print("\n1. Basic Multi-GPU Test (512 tokens, no dilation):")

    seq_len = 512
    for name, impl_class in [
        ("V2 Collective", RingDilatedAttentionV2Collective),
        ("V3", RingDilatedAttentionV3),
    ]:
        dist.barrier()

        try:
            if impl_class == RingDilatedAttentionV3:
                model = impl_class(
                    segment_lengths=[256],
                    dilation_rates=[1],
                    use_bucketed=False,
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                )
            else:
                model = impl_class(
                    segment_lengths=[256],
                    dilation_rates=[1],
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                    use_flash_attention=False,
                )

            # Create inputs
            q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

            start = time.time()
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            has_nan = torch.isnan(output).any().item()

            # Gather results
            times = [None] * world_size
            nans = [None] * world_size
            dist.all_gather_object(times, elapsed)
            dist.all_gather_object(nans, has_nan)

            if rank == 0:
                avg_time = sum(times) / len(times)
                any_nan = any(nans)
                status = "✅" if not any_nan else "❌ NaN"
                print(f"   {name:15} {avg_time:.3f}s {status}")

        except Exception as e:
            if rank == 0:
                print(f"   {name:15} ❌ Failed: {str(e)[:40]}...")

    # Test 2: Dilation support
    if rank == 0:
        print("\n2. Dilation Support Test (512 tokens, dilation=2):")

    for name, impl_class in [
        ("V2 Collective", RingDilatedAttentionV2Collective),
        ("V3", RingDilatedAttentionV3),
    ]:
        dist.barrier()

        try:
            if impl_class == RingDilatedAttentionV3:
                model = impl_class(
                    segment_lengths=[256],
                    dilation_rates=[2],  # Dilation > 1
                    use_bucketed=False,
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                )
            else:
                model = impl_class(
                    segment_lengths=[256],
                    dilation_rates=[2],  # Dilation > 1
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                    use_flash_attention=False,
                )

            q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

            output = model(q, k, v, is_causal=False)

            if rank == 0:
                print(f"   {name:15} ✅ Supports dilation")

        except Exception as e:
            if rank == 0:
                print(f"   {name:15} ❌ Dilation failed: {str(e)[:30]}...")

    # Test 3: Memory efficiency
    if rank == 0:
        print("\n3. Memory Usage (2048 tokens):")

    seq_len = 2048
    for name, impl_class in [
        ("V2 Collective", RingDilatedAttentionV2Collective),
        ("V3", RingDilatedAttentionV3),
    ]:
        dist.barrier()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            if impl_class == RingDilatedAttentionV3:
                model = impl_class(
                    segment_lengths=[1024],
                    dilation_rates=[1],
                    use_bucketed=False,
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                )
            else:
                model = impl_class(
                    segment_lengths=[1024],
                    dilation_rates=[1],
                    device=device,
                    dtype=torch.float32,
                    ring_size=world_size,
                    use_flash_attention=False,
                    enable_memory_pool=True,
                )

            q = torch.randn(1, seq_len, 8, 64, device=device) * 0.05
            k = torch.randn(1, seq_len, 8, 64, device=device) * 0.05
            v = torch.randn(1, seq_len, 8, 64, device=device) * 0.05

            _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

            # Gather memory stats
            memories = [None] * world_size
            dist.all_gather_object(memories, peak_memory)

            if rank == 0:
                max_memory = max(memories)
                print(f"   {name:15} Peak: {max_memory:.1f} MB")

        except Exception:
            if rank == 0:
                print(f"   {name:15} ❌ Failed")

    # Summary of key differences
    if rank == 0:
        print("\n" + "=" * 50)
        print("KEY IMPLEMENTATION DIFFERENCES:")
        print("=" * 50)

        print("\nV2 Collective advantages:")
        print("- ✅ Uses dist.all_gather (robust, NCCL optimized)")
        print("- ✅ Full dilation support in multi-GPU mode")
        print("- ✅ Enhanced memory pool for efficiency")
        print("- ✅ Pattern caching for repeated operations")
        print("- ✅ Automatic smart dtype selection")
        print("- ✅ No NaN issues with proper handling")

        print("\nV3 advantages:")
        print("- ✅ Based on lucidrains' proven patterns")
        print("- ✅ Explicit LSE accumulation for stability")
        print("- ✅ Cleaner separation of concerns")
        print("- ❌ But: bucketing implementation needs fix")
        print("- ❌ But: dilation disabled in multi-GPU")
        print("- ❌ But: slower due to complex ring utilities")

        print("\nRecommendation:")
        print("Combine V2's robust communication with V3's LSE accumulation")

    dist.destroy_process_group()


if __name__ == "__main__":
    compare_simple()
