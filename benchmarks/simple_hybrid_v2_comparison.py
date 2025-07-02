#!/usr/bin/env python3
"""
Simple performance comparison between Hybrid and V2 Collective.
Run with: torchrun --nproc_per_node=2 benchmarks/simple_hybrid_v2_comparison.py
"""

import os
import gc
import time
import torch
import torch.distributed as dist
from datetime import datetime


def simple_benchmark():
    """Simple benchmark comparing memory usage."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/simple_hybrid_v2_comparison.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Simple Hybrid vs V2 Collective Memory Comparison")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Test with a moderate sequence length
    seq_len = 2048
    segment_len = 1024

    dist.barrier()

    # Test V2 Collective
    if rank == 0:
        print(f"Testing V2 Collective (seq_len={seq_len})...")

    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    v2_model = RingDilatedAttentionV2Collective(
        segment_lengths=[segment_len],
        dilation_rates=[1],
        ring_size=world_size,
        device=device,
        dtype=torch.float16,
    )

    # Create inputs
    q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1

    # Memory before forward
    v2_mem_before = torch.cuda.memory_allocated(device) / (1024**2)

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        v2_output = v2_model(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    v2_time = time.time() - start_time

    # Memory after forward
    v2_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    v2_mem_after = torch.cuda.memory_allocated(device) / (1024**2)

    # Gather results
    v2_stats = {
        "peak_mem": v2_peak_mem,
        "mem_before": v2_mem_before,
        "mem_after": v2_mem_after,
        "time": v2_time,
        "has_nan": torch.isnan(v2_output).any().item(),
    }

    all_v2_stats = [None] * world_size
    dist.all_gather_object(all_v2_stats, v2_stats)

    if rank == 0:
        avg_v2_peak = sum(s["peak_mem"] for s in all_v2_stats) / len(all_v2_stats)
        max_v2_peak = max(s["peak_mem"] for s in all_v2_stats)
        avg_v2_time = sum(s["time"] for s in all_v2_stats) / len(all_v2_stats)
        print("V2 Collective:")
        print(f"  Peak memory: {avg_v2_peak:.1f}MB avg, {max_v2_peak:.1f}MB max")
        print(f"  Time: {avg_v2_time * 1000:.2f}ms")
        print(f"  Memory growth: {avg_v2_peak - all_v2_stats[0]['mem_before']:.1f}MB")

    # Clean up V2
    del v2_model, v2_output
    gc.collect()
    torch.cuda.empty_cache()

    dist.barrier()

    # Test Hybrid
    if rank == 0:
        print(f"\nTesting Hybrid (seq_len={seq_len})...")

    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    hybrid_model = RingDilatedAttentionHybrid(
        segment_lengths=[segment_len],
        dilation_rates=[1],
        ring_size=world_size,
        device=device,
        dtype=torch.float16,
        enable_memory_pool=True,
        use_pattern_cache=True,
        use_flash_attention=False,  # Disable for fair comparison
    )

    # Recreate inputs (to be fair)
    q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1

    # Memory before forward
    hybrid_mem_before = torch.cuda.memory_allocated(device) / (1024**2)

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        hybrid_output = hybrid_model(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    hybrid_time = time.time() - start_time

    # Memory after forward
    hybrid_peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    hybrid_mem_after = torch.cuda.memory_allocated(device) / (1024**2)

    # Gather results
    hybrid_stats = {
        "peak_mem": hybrid_peak_mem,
        "mem_before": hybrid_mem_before,
        "mem_after": hybrid_mem_after,
        "time": hybrid_time,
        "has_nan": torch.isnan(hybrid_output).any().item(),
    }

    all_hybrid_stats = [None] * world_size
    dist.all_gather_object(all_hybrid_stats, hybrid_stats)

    if rank == 0:
        avg_hybrid_peak = sum(s["peak_mem"] for s in all_hybrid_stats) / len(
            all_hybrid_stats
        )
        max_hybrid_peak = max(s["peak_mem"] for s in all_hybrid_stats)
        avg_hybrid_time = sum(s["time"] for s in all_hybrid_stats) / len(
            all_hybrid_stats
        )
        print("Hybrid:")
        print(
            f"  Peak memory: {avg_hybrid_peak:.1f}MB avg, {max_hybrid_peak:.1f}MB max"
        )
        print(f"  Time: {avg_hybrid_time * 1000:.2f}ms")
        print(
            f"  Memory growth: {avg_hybrid_peak - all_hybrid_stats[0]['mem_before']:.1f}MB"
        )

        # Comparison
        print("\nComparison:")
        print(f"  Memory reduction: {(1 - avg_hybrid_peak / avg_v2_peak) * 100:.1f}%")
        print(f"  Time ratio: {avg_hybrid_time / avg_v2_time:.2f}x")
        print(f"\nExpected: Hybrid should use ~{100 / world_size:.0f}% of V2's memory")
        print(
            f"Actual: Hybrid uses {avg_hybrid_peak / avg_v2_peak * 100:.1f}% of V2's memory"
        )

    # Test larger sequence
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Testing with larger sequence...")

    seq_len = 4096
    segment_len = 2048

    # Quick memory test only
    for model_name, model_class in [
        ("V2 Collective", RingDilatedAttentionV2Collective),
        ("Hybrid", RingDilatedAttentionHybrid),
    ]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            if model_name == "Hybrid":
                model = model_class(
                    segment_lengths=[segment_len],
                    dilation_rates=[1],
                    ring_size=world_size,
                    device=device,
                    dtype=torch.float16,
                    enable_memory_pool=True,
                    use_pattern_cache=True,
                    use_flash_attention=False,
                )
            else:
                model = model_class(
                    segment_lengths=[segment_len],
                    dilation_rates=[1],
                    ring_size=world_size,
                    device=device,
                    dtype=torch.float16,
                )

            q = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
            k = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1
            v = torch.randn(1, seq_len, 8, 64, device=device, dtype=torch.float16) * 0.1

            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

            all_peaks = [None] * world_size
            dist.all_gather_object(all_peaks, peak_mem)

            if rank == 0:
                avg_peak = sum(all_peaks) / len(all_peaks)
                print(f"{model_name}: {avg_peak:.1f}MB avg peak memory")

        except Exception as e:
            if rank == 0:
                print(f"{model_name}: Failed - {e}")

        dist.barrier()

    dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Summary:")
        print("- V2 uses all_gather: Each GPU stores full K,V (O(n) memory)")
        print("- Hybrid uses ring passing: Each GPU stores 1/p of K,V (O(n/p) memory)")
        print("- Memory reduction should be approximately (p-1)/p where p = world_size")


if __name__ == "__main__":
    simple_benchmark()
