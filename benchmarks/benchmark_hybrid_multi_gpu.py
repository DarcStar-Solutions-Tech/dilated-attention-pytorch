#!/usr/bin/env python3
"""
Comprehensive benchmark for hybrid ring dilated attention on multiple GPUs.
Measures throughput, memory usage, and scaling characteristics.

Run with: torchrun --nproc_per_node=<num_gpus> benchmarks/benchmark_hybrid_multi_gpu.py
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any

import torch
import torch.distributed as dist
from torch.cuda import Event


def benchmark_configuration(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
) -> Dict[str, Any]:
    """Benchmark a specific configuration."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=<num_gpus> benchmarks/benchmark_hybrid_multi_gpu.py"
        )
        return {}

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Import the hybrid implementation
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float16,  # Use fp16 for performance
        enable_memory_pool=True,
        use_flash_attention=True,
    ).to(device)

    # Create inputs
    _ = num_heads * head_dim
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )

    # Warmup
    if rank == 0:
        print(f"Warming up ({warmup_iters} iterations)...")

    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

    # Get initial memory
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB

    # Benchmark forward pass
    if rank == 0:
        print(f"Benchmarking forward pass ({benchmark_iters} iterations)...")

    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)

    forward_times = []

    for _ in range(benchmark_iters):
        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        end_event.record()
        torch.cuda.synchronize()

        forward_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds
        forward_times.append(forward_time)

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
    memory_used = peak_memory - initial_memory

    # Calculate statistics
    avg_forward_time = sum(forward_times) / len(forward_times)
    min_forward_time = min(forward_times)
    max_forward_time = max(forward_times)

    # Calculate throughput
    total_tokens = batch_size * seq_len
    tokens_per_second = total_tokens / avg_forward_time

    # Memory per token
    memory_per_token = memory_used / total_tokens * 1024  # KB per token

    # Gather results from all ranks
    local_results = {
        "rank": rank,
        "avg_forward_time": avg_forward_time,
        "min_forward_time": min_forward_time,
        "max_forward_time": max_forward_time,
        "memory_used_mb": memory_used,
        "peak_memory_mb": peak_memory,
        "tokens_per_second": tokens_per_second,
        "memory_per_token_kb": memory_per_token,
    }

    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_results)

    # Aggregate results on rank 0
    if rank == 0:
        # Calculate aggregate statistics
        total_tokens_per_second = sum(r["tokens_per_second"] for r in all_results)
        avg_memory_per_gpu = sum(r["memory_used_mb"] for r in all_results) / world_size
        max_memory_per_gpu = max(r["memory_used_mb"] for r in all_results)

        results = {
            "configuration": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "num_gpus": world_size,
            },
            "performance": {
                "avg_forward_time_ms": avg_forward_time * 1000,
                "min_forward_time_ms": min_forward_time * 1000,
                "max_forward_time_ms": max_forward_time * 1000,
                "total_throughput_tokens_per_sec": total_tokens_per_second,
                "throughput_per_gpu_tokens_per_sec": total_tokens_per_second
                / world_size,
            },
            "memory": {
                "avg_memory_per_gpu_mb": avg_memory_per_gpu,
                "max_memory_per_gpu_mb": max_memory_per_gpu,
                "memory_per_token_kb": memory_per_token,
                "theoretical_memory_scaling": f"O(n/{world_size})",
            },
            "per_gpu_results": all_results,
        }

        return results

    # Don't destroy process group here - it's still needed
    return {}


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""

    # Initialize distributed first
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("Running Hybrid Ring Dilated Attention Benchmarks")
        print(f"Number of GPUs: {world_size}")
        print("=" * 60)

    # Test configurations
    configurations = [
        # Small sequences
        {
            "batch_size": 4,
            "seq_len": 2048,
            "num_heads": 16,
            "head_dim": 64,
            "segment_lengths": [512, 1024],
            "dilation_rates": [1, 2],
        },
        # Medium sequences
        {
            "batch_size": 2,
            "seq_len": 8192,
            "num_heads": 16,
            "head_dim": 64,
            "segment_lengths": [1024, 2048, 4096],
            "dilation_rates": [1, 2, 4],
        },
        # Long sequences
        {
            "batch_size": 1,
            "seq_len": 32768,
            "num_heads": 16,
            "head_dim": 64,
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
        },
        # Very long sequences (if memory permits)
        {
            "batch_size": 1,
            "seq_len": 65536,
            "num_heads": 16,
            "head_dim": 64,
            "segment_lengths": [4096, 8192, 16384],
            "dilation_rates": [1, 2, 4],
        },
    ]

    all_results = []

    for config in configurations:
        if rank == 0:
            print(
                f"\nTesting seq_len={config['seq_len']}, batch_size={config['batch_size']}..."
            )

        try:
            result = benchmark_configuration(**config)
            if result and rank == 0:
                all_results.append(result)

                # Print summary
                perf = result["performance"]
                mem = result["memory"]
                print(f"  Forward time: {perf['avg_forward_time_ms']:.2f} ms")
                print(
                    f"  Throughput: {perf['total_throughput_tokens_per_sec']:,.0f} tokens/sec"
                )
                print(f"  Memory per GPU: {mem['avg_memory_per_gpu_mb']:.1f} MB")
                print(f"  Memory per token: {mem['memory_per_token_kb']:.2f} KB")

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print("  Skipped: Out of memory")
        except Exception as e:
            if rank == 0:
                print(f"  Skipped: {str(e)}")

    # Save results
    if rank == 0 and all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"benchmarks/hybrid_benchmark_results_{world_size}gpu_{timestamp}.json"
        )

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "num_gpus": world_size,
                    "results": all_results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

        # Print scaling summary
        print("\n" + "=" * 60)
        print("SCALING SUMMARY")
        print("=" * 60)

        for result in all_results:
            config = result["configuration"]
            perf = result["performance"]
            mem = result["memory"]

            print(f"\nSeq Length: {config['seq_len']:,}")
            print(
                f"  Throughput/GPU: {perf['throughput_per_gpu_tokens_per_sec']:,.0f} tokens/sec"
            )
            print(f"  Memory/GPU: {mem['avg_memory_per_gpu_mb']:.1f} MB")
            print(f"  Memory/Token: {mem['memory_per_token_kb']:.2f} KB")
            print(f"  Scaling: {mem['theoretical_memory_scaling']}")

    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main entry point."""

    if "RANK" not in os.environ:
        print("This benchmark requires multiple GPUs.")
        print(
            "Run with: torchrun --nproc_per_node=<num_gpus> benchmarks/benchmark_hybrid_multi_gpu.py"
        )
        print("\nExample:")
        print("  torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_multi_gpu.py")
        return

    run_benchmark_suite()


if __name__ == "__main__":
    main()
