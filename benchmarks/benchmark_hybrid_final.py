#!/usr/bin/env python3
"""
Final benchmark for hybrid ring dilated attention.
Focused on demonstrating O(n/p) memory scaling.

Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_final.py
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import datetime


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_final.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Hybrid Ring Dilated Attention - Performance Benchmark")
        print(f"GPUs: {world_size}")
        print("=" * 60)
        print("\nDemonstrating O(n/p) memory scaling:")
        print(
            "Memory per GPU should remain roughly constant as sequence length increases"
        )
        print("")

    # Import model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Model config - simple for speed
    model = RingDilatedAttentionHybrid(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        enable_memory_pool=False,
        use_flash_attention=False,
    )

    # Test configurations
    configs = [
        {"seq": 2048, "batch": 1, "heads": 8, "dim": 64},
        {"seq": 4096, "batch": 1, "heads": 8, "dim": 64},
        {"seq": 8192, "batch": 1, "heads": 8, "dim": 64},
    ]

    results = []

    for cfg in configs:
        seq_len = cfg["seq"]
        batch_size = cfg["batch"]
        num_heads = cfg["heads"]
        head_dim = cfg["dim"]

        if rank == 0:
            print(f"Testing seq_len={seq_len}...", end="", flush=True)

        try:
            # Create inputs
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Warmup
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Benchmark
            torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(3):
                start = time.time()
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2

            # Gather results
            all_times = [None] * world_size
            all_memories = [None] * world_size

            dist.all_gather_object(all_times, avg_time)
            dist.all_gather_object(all_memories, peak_memory)

            if rank == 0:
                avg_time_global = sum(all_times) / len(all_times)
                avg_memory = sum(all_memories) / len(all_memories)

                tokens = batch_size * seq_len
                throughput = tokens / avg_time_global
                memory_per_token = avg_memory / tokens * 1024  # KB

                # Expected memory scaling
                expected_scaling = seq_len / world_size

                result = {
                    "seq_len": seq_len,
                    "time_ms": avg_time_global * 1000,
                    "memory_mb": avg_memory,
                    "throughput": throughput,
                    "memory_per_token_kb": memory_per_token,
                    "expected_scaling": expected_scaling,
                }
                results.append(result)

                print(" Done!")
                print(f"  Time: {result['time_ms']:.1f} ms")
                print(f"  Memory/GPU: {result['memory_mb']:.1f} MB")
                print(f"  Memory/token: {result['memory_per_token_kb']:.2f} KB")
                print(f"  Throughput: {result['throughput']:,.0f} tokens/sec")
                print(f"  Expected O(n/p): {expected_scaling:,.0f}")
                print("")

        except Exception as e:
            if rank == 0:
                print(f" Error: {str(e)}")

    # Summary
    if rank == 0 and len(results) > 1:
        print("\n" + "=" * 60)
        print("MEMORY SCALING SUMMARY")
        print("=" * 60)
        print(f"Configuration: {world_size} GPUs")
        print("Segment lengths: [512, 1024], Dilation rates: [1, 2]")
        print("\nMemory per token (should remain ~constant with O(n/p) scaling):")

        for r in results:
            print(f"  Seq {r['seq_len']:,}: {r['memory_per_token_kb']:.2f} KB/token")

        # Calculate scaling ratio
        first_mem = results[0]["memory_per_token_kb"]
        last_mem = results[-1]["memory_per_token_kb"]
        ratio = last_mem / first_mem

        print(f"\nMemory scaling ratio (last/first): {ratio:.2f}")
        if ratio < 1.2:
            print("✅ Excellent O(n/p) scaling!")
        elif ratio < 1.5:
            print("✅ Good O(n/p) scaling")
        else:
            print("⚠️  Memory scaling could be better")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmarks/hybrid_results_{world_size}gpu_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write("Hybrid Ring Dilated Attention Benchmark Results\n")
            f.write(f"GPUs: {world_size}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 60 + "\n\n")

            for r in results:
                f.write(f"Sequence Length: {r['seq_len']:,}\n")
                f.write(f"  Time: {r['time_ms']:.1f} ms\n")
                f.write(f"  Memory/GPU: {r['memory_mb']:.1f} MB\n")
                f.write(f"  Memory/token: {r['memory_per_token_kb']:.2f} KB\n")
                f.write(f"  Throughput: {r['throughput']:,.0f} tokens/sec\n")
                f.write(f"  O(n/p): {r['expected_scaling']:,.0f}\n\n")

        print(f"\nResults saved to {filename}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
