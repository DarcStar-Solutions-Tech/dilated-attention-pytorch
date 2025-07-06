#!/usr/bin/env python3
"""Test distributed scaling with conservative memory settings."""

import os
import torch
import torch.distributed as dist
import time
from datetime import datetime


def init_distributed():
    """Initialize distributed if not already done."""
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank
    return 0, 0


def test_configuration(seq_len, segments, dilations, batch_size=1, iterations=3):
    """Test a specific configuration."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    # Import after device setup
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    # Model parameters
    num_heads = 8
    head_dim = 64
    _ = num_heads * head_dim
    dtype = torch.float32  # Use float32 for stability

    try:
        # Create model with conservative settings
        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            use_hilbert=True,
            hilbert_chunk_size=2048,  # Smaller chunk size
            enable_memory_pool=False,  # Disable memory pool
            use_xformers=False,  # Disable xformers
            enable_profiling=False,
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        throughput = (batch_size * seq_len) / avg_time
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        # Gather results from all ranks
        if dist.is_initialized():
            metrics = torch.tensor([throughput, memory_gb], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            throughput, memory_gb = metrics.tolist()

        return True, throughput, memory_gb, avg_time * 1000

    except Exception as e:
        if rank == 0:
            print(f"Error: {str(e)}")
        return False, None, None, None
    finally:
        torch.cuda.empty_cache()


def main():
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 60)
        print("DISTRIBUTED SCALING TEST")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print("Testing conservative configurations")
        print()

    # Test configurations - start conservative
    test_cases = [
        # (seq_len, segments, dilations, description)
        (32768, [8192], [1], "32K, no dilation"),
        (32768, [8192], [2], "32K, dilation=2"),
        (65536, [16384], [1], "64K, no dilation"),
        (65536, [16384], [2], "64K, dilation=2"),
        (131072, [32768], [1], "128K, no dilation"),
        (131072, [16384, 32768], [1, 2], "128K, multi-segment"),
        (262144, [65536], [1], "256K, single segment"),
        (262144, [32768, 65536], [1, 2], "256K, multi-segment"),
    ]

    results = []

    for seq_len, segments, dilations, desc in test_cases:
        # Check if valid for world size
        if seq_len % world_size != 0:
            if rank == 0:
                print(f"Skipping {desc}: not divisible by {world_size}")
            continue

        if seq_len % max(segments) != 0:
            if rank == 0:
                print(f"Skipping {desc}: not divisible by segment size")
            continue

        if rank == 0:
            print(f"\nTesting {desc}...")
            print(f"  Segments: {segments}, Dilation: {dilations}")

        success, throughput, memory, time_ms = test_configuration(
            seq_len, segments, dilations
        )

        if rank == 0:
            if success:
                print("  ✓ Success!")
                print(f"    Time: {time_ms:.1f} ms")
                print(f"    Throughput: {throughput:,.0f} tokens/sec")
                print(f"    Memory per GPU: {memory:.2f} GB")

                results.append(
                    {
                        "description": desc,
                        "seq_len": seq_len,
                        "segments": segments,
                        "dilations": dilations,
                        "throughput": throughput,
                        "memory_gb": memory,
                        "time_ms": time_ms,
                        "world_size": world_size,
                    }
                )
            else:
                print("  ✗ Failed")

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Compare to single GPU baseline
        print("\nScaling Analysis:")
        print(
            f"{'Sequence':<15} | {'Throughput':<15} | {'Memory/GPU':<12} | {'Time':<10}"
        )
        print("-" * 60)

        for r in results:
            print(
                f"{r['description']:<15} | {r['throughput']:>14,.0f} | "
                f"{r['memory_gb']:>11.2f} | {r['time_ms']:>9.1f} ms"
            )

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"distributed_scaling_{world_size}gpu_{timestamp}.json"

        import json

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
