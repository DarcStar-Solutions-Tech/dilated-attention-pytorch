#!/usr/bin/env python3
"""
Benchmark hybrid ring dilated attention with optimized attention backends.
Tests the performance impact of using Flash/SDPA/xFormers vs standard einsum.
"""

import torch
import torch.distributed as dist
import time
import os
from typing import Dict, List
import json

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)
from dilated_attention_pytorch.ring_attention_lse_optimized import (
    get_attention_backend_info,
)


def setup_distributed():
    """Initialize distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    return rank, world_size


def measure_attention_performance(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: List[int] = [2048, 4096, 8192],
    dilation_rates: List[int] = [1, 2, 4],
    warmup_steps: int = 5,
    benchmark_steps: int = 10,
    device: str = "cuda",
) -> Dict:
    """Measure performance of hybrid attention with optimized backends."""

    # Get backend info
    backend_info = get_attention_backend_info()

    # Create model with optimized attention
    device_obj = torch.device(device) if isinstance(device, str) else device
    model = RingDilatedAttentionHybrid(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device_obj,
        dtype=torch.float32,  # Use float32 for Pascal GPUs
        use_flash_attention=True,  # Enable optimized backends
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(benchmark_steps):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    avg_time = (end_time - start_time) / benchmark_steps
    throughput = (batch_size * seq_len * num_heads) / avg_time / 1e6  # M tokens/s

    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "time_ms": avg_time * 1000,
        "throughput_mtokens": throughput,
        "memory_mb": peak_memory,
        "backend_info": backend_info,
        "model_info": {
            "can_use_flash": model._can_use_flash,
            "flash_backend": model.flash_backend,
            "has_optimized_lse": model.enable_memory_pool,
        },
    }


def compare_with_standard(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: List[int] = [2048, 4096, 8192],
    dilation_rates: List[int] = [1, 2, 4],
    device: str = "cuda",
) -> Dict:
    """Compare optimized vs standard attention performance."""

    device_obj = torch.device(device) if isinstance(device, str) else device

    # Test with optimized backends enabled
    model_optimized = RingDilatedAttentionHybrid(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device_obj,
        dtype=torch.float32,
        use_flash_attention=True,
    )

    # Test with optimized backends disabled
    model_standard = RingDilatedAttentionHybrid(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device_obj,
        dtype=torch.float32,
        use_flash_attention=False,
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Benchmark optimized
    times_optimized = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out_opt = model_optimized(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        times_optimized.append(time.time() - start)

    # Benchmark standard
    times_standard = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out_std = model_standard(q, k, v, is_causal=False)
        torch.cuda.synchronize()
        times_standard.append(time.time() - start)

    # Check outputs match
    max_diff = (out_opt - out_std).abs().max().item()

    avg_optimized = sum(times_optimized) / len(times_optimized)
    avg_standard = sum(times_standard) / len(times_standard)
    speedup = avg_standard / avg_optimized

    return {
        "seq_len": seq_len,
        "optimized_ms": avg_optimized * 1000,
        "standard_ms": avg_standard * 1000,
        "speedup": speedup,
        "max_diff": max_diff,
        "backend_used": get_attention_backend_info(),
    }


def main():
    """Run benchmarks."""
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=== Hybrid Ring Dilated Attention with Optimized Backends ===")
        print(f"World size: {world_size}")

        # Check backend availability
        backend_info = get_attention_backend_info()
        print("\nAvailable backends:")
        for key, value in backend_info.items():
            print(f"  {key}: {value}")

    # Test different sequence lengths
    seq_lengths = [1024, 2048, 4096, 8192, 16384]

    results = []
    comparisons = []

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nTesting sequence length: {seq_len}")

        # Performance benchmark
        result = measure_attention_performance(
            seq_len=seq_len,
            batch_size=2,
            num_heads=8,
            head_dim=64,
            segment_lengths=[min(2048, seq_len)],
            dilation_rates=[1],
        )
        results.append(result)

        if rank == 0:
            print(f"  Time: {result['time_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput_mtokens']:.2f} M tokens/s")
            print(f"  Memory: {result['memory_mb']:.2f} MB")

        # Compare with standard (only on single GPU)
        if world_size == 1 and seq_len <= 4096:
            comparison = compare_with_standard(
                seq_len=seq_len,
                batch_size=2,
                num_heads=8,
                head_dim=64,
                segment_lengths=[min(2048, seq_len)],
                dilation_rates=[1],
            )
            comparisons.append(comparison)

            if rank == 0:
                print(f"  Optimized vs Standard: {comparison['speedup']:.2f}x speedup")
                print(f"  Max difference: {comparison['max_diff']:.2e}")

    # Save results
    if rank == 0:
        output = {
            "world_size": world_size,
            "backend_info": backend_info,
            "performance_results": results,
            "comparison_results": comparisons,
        }

        with open(f"hybrid_optimized_benchmark_results_gpu{world_size}.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\n=== Summary ===")
        print("Performance with optimized attention backends:")
        for result in results:
            print(
                f"  Seq {result['seq_len']}: {result['time_ms']:.2f} ms, "
                f"{result['throughput_mtokens']:.2f} M tokens/s"
            )

        if comparisons:
            print("\nSpeedup vs standard attention:")
            for comp in comparisons:
                print(f"  Seq {comp['seq_len']}: {comp['speedup']:.2f}x faster")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
