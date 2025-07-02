#!/usr/bin/env python3
"""
Quick benchmark for Hybrid implementation focusing on memory usage.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_quick.py
"""

import os
import gc
import time
import torch
import torch.distributed as dist

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def quick_test():
    """Quick test focusing on memory scaling."""

    # Initialize distributed if available
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("Hybrid Ring Attention - Quick Memory Test")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print(f"Device: {device}")

    # Test a single configuration
    seq_len = 2048
    segment_len = 1024
    batch_size = 1
    num_heads = 8
    head_dim = 64

    if rank == 0:
        print(f"\nTesting seq_len={seq_len}, segment_len={segment_len}")

    # Synchronize
    if world_size > 1:
        dist.barrier()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=[segment_len],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        # Auto-select dtype based on GPU
        enable_memory_pool=True,
        use_pattern_cache=True,
        use_flash_attention=False,
    )

    if rank == 0:
        print(f"Model dtype: {model.dtype}")

    # Memory after model creation
    mem_model = torch.cuda.memory_allocated(device) / (1024**2)

    # Create inputs
    torch.manual_seed(42)
    q = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )
    k = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )
    v = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
        )
        * 0.1
    )

    # Memory after inputs
    mem_inputs = torch.cuda.memory_allocated(device) / (1024**2)
    input_size = mem_inputs - mem_model

    # Forward pass with timing
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Memory stats
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    mem_after = torch.cuda.memory_allocated(device) / (1024**2)

    # Check output
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    # Gather results
    stats = {
        "rank": rank,
        "time_ms": elapsed * 1000,
        "mem_model": mem_model,
        "mem_inputs": mem_inputs,
        "input_size": input_size,
        "peak_memory": peak_memory,
        "mem_after": mem_after,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }

    if world_size > 1:
        all_stats = [None] * world_size
        dist.all_gather_object(all_stats, stats)
    else:
        all_stats = [stats]

    # Report results
    if rank == 0:
        print("\nResults:")
        print("-" * 60)

        for s in all_stats:
            print(f"GPU {s['rank']}:")
            print(f"  Time: {s['time_ms']:.2f}ms")
            print(
                f"  Memory - Model: {s['mem_model']:.1f}MB, Inputs: {s['input_size']:.1f}MB"
            )
            print(f"  Peak memory: {s['peak_memory']:.1f}MB")
            print(f"  Issues: NaN={s['has_nan']}, Inf={s['has_inf']}")

        # Average stats
        avg_time = sum(s["time_ms"] for s in all_stats) / len(all_stats)
        avg_peak = sum(s["peak_memory"] for s in all_stats) / len(all_stats)
        max_peak = max(s["peak_memory"] for s in all_stats)

        print("\nSummary:")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Average peak memory: {avg_peak:.1f}MB")
        print(f"  Max peak memory: {max_peak:.1f}MB")

        # Theoretical memory calculation
        # Each GPU stores: Full Q + 1/world_size of K,V
        elements_per_tensor = seq_len * batch_size * num_heads * head_dim
        bytes_per_element = 4 if model.dtype == torch.float32 else 2

        q_memory = elements_per_tensor * bytes_per_element / (1024**2)
        kv_memory_per_gpu = (
            2 * elements_per_tensor * bytes_per_element / (1024**2) / world_size
        )
        theoretical = q_memory + kv_memory_per_gpu

        print("\nMemory Analysis:")
        print(f"  Theoretical (Q + K/p + V/p): {theoretical:.1f}MB")
        print(f"  Actual average peak: {avg_peak:.1f}MB")
        print(
            f"  Overhead: {avg_peak - theoretical:.1f}MB ({(avg_peak / theoretical - 1) * 100:.1f}%)"
        )

        if world_size > 1:
            print("\nMemory scaling:")
            print(
                f"  Expected reduction vs single GPU: {(1 - 1 / world_size) * 100:.1f}%"
            )
            # Can't directly compare without single GPU baseline

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    quick_test()
