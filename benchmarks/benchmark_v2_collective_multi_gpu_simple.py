#!/usr/bin/env python3
"""
Simple multi-GPU benchmark for V2 Collective using torchrun.
Run with: torchrun --nproc_per_node=<num_gpus> benchmark_v2_collective_multi_gpu_simple.py
"""

import os
import gc
import time
import json

import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory / (1024**3)
    free = total - reserved
    return allocated, reserved, free, total


def benchmark_configuration(
    seq_len,
    batch_size,
    num_heads,
    head_dim,
    segment_lengths,
    dilation_rates,
    iterations=10,
):
    """Benchmark a specific configuration."""

    # Get rank and world size
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{rank}")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    if rank == 0:
        print(
            f"\nTesting seq_len={seq_len:,}, batch={batch_size}, world_size={world_size}"
        )
        print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

    try:
        # Create model
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            dtype=torch.float16,
            ring_size=world_size,
            enable_memory_pool=True,
        )

        # Report initial memory
        if rank == 0:
            alloc, reserved, free, total = get_gpu_memory_info()
            print(f"  Initial GPU memory: {free:.2f}GB free of {total:.2f}GB")

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
        )

        # Synchronize
        if world_size > 1:
            dist.barrier()

        # Warmup
        for _ in range(3):
            _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

        if world_size > 1:
            dist.barrier()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Time iterations
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(iterations):
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

        if world_size > 1:
            dist.barrier()

        end_time = time.perf_counter()

        # Calculate metrics
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        throughput = (seq_len * batch_size) / (avg_time / 1000)

        # Report results
        if rank == 0:
            print(f"  ✓ Success on GPU {rank}:")
            print(f"    Time: {avg_time:.1f}ms")
            print(f"    Memory: {peak_memory:.0f}MB")
            print(f"    Throughput: {throughput:,.0f} tokens/sec")

        # Gather memory info from all ranks
        if world_size > 1:
            all_memories = [None] * world_size
            dist.all_gather_object(all_memories, peak_memory)

            if rank == 0:
                total_memory = sum(all_memories)
                avg_memory = total_memory / world_size
                print(f"  Total memory across {world_size} GPUs: {total_memory:.0f}MB")
                print(f"  Average memory per GPU: {avg_memory:.0f}MB")

        # Clean up
        del q, k, v, output, model
        torch.cuda.empty_cache()
        gc.collect()

        return True, avg_time, peak_memory, throughput

    except Exception as e:
        if rank == 0:
            print(f"  ✗ Failed: {str(e)}")
        return False, 0, 0, 0


def main():
    """Main benchmark function."""
    # Initialize distributed if running with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        print("=" * 80)
        print("V2 Collective Multi-GPU Benchmark")
        print("=" * 80)
        print(f"World size: {world_size}")

        # Print GPU info
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {name} ({mem_gb:.1f}GB)")

    # Test configurations
    configs = [
        # Small - test overhead
        (4096, 4, [512, 1024], [1, 2]),
        # Medium
        (16384, 2, [2048, 4096], [1, 2]),
        # Large
        (65536, 1, [4096, 8192], [1, 2]),
        # Very large - single GPU limit
        (262144, 1, [8192, 16384], [1, 2]),
        # Beyond single GPU
        (524288, 1, [16384, 32768], [1, 2]),
    ]

    # Additional configs for multi-GPU
    if world_size > 1:
        configs.extend(
            [
                # Test if we can handle even larger with multiple GPUs
                (1048576, 1, [32768, 65536], [1, 2]),  # 1M tokens
            ]
        )

    results = []

    for seq_len, batch_size, segment_lengths, dilation_rates in configs:
        success, avg_time, peak_memory, throughput = benchmark_configuration(
            seq_len, batch_size, 8, 64, segment_lengths, dilation_rates
        )

        if rank == 0:
            results.append(
                {
                    "world_size": world_size,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "success": success,
                    "avg_time_ms": avg_time,
                    "peak_memory_mb": peak_memory,
                    "throughput": throughput,
                }
            )

    # Save results on rank 0
    if rank == 0:
        from datetime import datetime

        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
        output_file = f"benchmarks/v2_collective_gpu{world_size}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)

        max_seq = max([r["seq_len"] for r in results if r["success"]], default=0)
        print(f"Maximum sequence length with {world_size} GPU(s): {max_seq:,} tokens")

        # Check memory scaling
        if world_size > 1:
            print("\nMemory scaling analysis:")
            for r in results:
                if (
                    r["success"] and r["seq_len"] <= 262144
                ):  # Compare with single GPU limit
                    print(
                        f"  {r['seq_len']:,} tokens: {r['peak_memory_mb']:.0f}MB per GPU"
                    )

    # Clean up distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
