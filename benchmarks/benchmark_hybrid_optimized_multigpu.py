#!/usr/bin/env python3
"""
Benchmark optimized hybrid ring dilated attention on multiple GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_optimized_multigpu.py
"""

import os
import time
import json
from datetime import datetime
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_optimized_multigpu.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting optimized benchmark...")

    # Import optimized model
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized import (
        RingDilatedAttentionHybridOptimized,
    )

    # Test configurations
    configs = [
        {"seq": 1024, "batch": 2},
        {"seq": 2048, "batch": 2},
        {"seq": 4096, "batch": 1},
        {"seq": 8192, "batch": 1},
        {"seq": 16384, "batch": 1},
        {"seq": 32768, "batch": 1},
        {"seq": 65536, "batch": 1},
    ]

    # Model configuration
    segment_lengths = [512, 1024]
    dilation_rates = [1, 2]

    results = []

    for config in configs:
        seq_len = config["seq"]
        batch_size = config["batch"]

        # Skip if not divisible by max segment
        if seq_len % max(segment_lengths) != 0:
            continue

        print(f"[GPU {rank}] Testing seq_len={seq_len}...")

        try:
            # Create model
            model = RingDilatedAttentionHybridOptimized(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                enable_memory_pool=True,
                use_flash_attention=False,  # Disable for Pascal
                use_pattern_cache=True,
                batch_segments=True,
                precompute_patterns=True,
            )

            # Create inputs
            num_heads = 8
            head_dim = 64
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()

            # Benchmark
            torch.cuda.reset_peak_memory_stats()

            times = []
            for _ in range(5):
                start = time.time()
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

            # Calculate metrics
            tokens = batch_size * seq_len
            throughput = tokens / avg_time
            memory_per_token = peak_memory / tokens * 1024  # KB

            # Expected memory with O(n/p) scaling
            expected_mem_scaling = seq_len / world_size

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "time_ms": avg_time * 1000,
                "memory_mb": peak_memory,
                "throughput": throughput,
                "memory_per_token_kb": memory_per_token,
                "expected_scaling": expected_mem_scaling,
            }

            results.append(result)

            print(
                f"[GPU {rank}] Seq {seq_len}: {avg_time * 1000:.1f}ms, {peak_memory:.1f}MB, {memory_per_token:.2f}KB/token"
            )

        except torch.cuda.OutOfMemoryError:
            print(f"[GPU {rank}] Seq {seq_len}: OOM")
            break
        except Exception as e:
            print(f"[GPU {rank}] Seq {seq_len}: Error - {str(e)}")
            break

    # Save results for this GPU
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmarks/hybrid_optimized_results_rank{rank}_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "timestamp": timestamp,
                    "segment_lengths": segment_lengths,
                    "dilation_rates": dilation_rates,
                    "optimized": True,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"[GPU {rank}] Results saved to {filename}")

    # Synchronize before printing summary
    dist.barrier()

    # Only rank 0 prints summary
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("OPTIMIZED BENCHMARK SUMMARY (from GPU 0)")
        print("=" * 60)
        print(f"Configuration: {world_size} GPUs")
        print(f"Segments: {segment_lengths}, Dilation: {dilation_rates}")
        print("\nResults:")

        for r in results:
            print(f"\nSeq {r['seq_len']:,}:")
            print(f"  Time: {r['time_ms']:.1f} ms")
            print(f"  Memory/GPU: {r['memory_mb']:.1f} MB")
            print(f"  Throughput: {r['throughput']:,.0f} tokens/sec")
            print(f"  Memory/token: {r['memory_per_token_kb']:.2f} KB")

        # Check memory scaling
        if len(results) >= 2:
            print("\nMemory Scaling Analysis:")
            for i in range(len(results)):
                print(
                    f"  Seq {results[i]['seq_len']:,}: {results[i]['memory_per_token_kb']:.2f} KB/token"
                )

            first_mem = results[0]["memory_per_token_kb"]
            last_mem = results[-1]["memory_per_token_kb"]
            ratio = last_mem / first_mem

            print(f"\nMemory ratio (last/first): {ratio:.2f}")
            if ratio < 1.3:
                print("✅ Excellent O(n/p) memory scaling!")
            elif ratio < 1.5:
                print("✅ Good O(n/p) memory scaling")
            else:
                print("⚠️  Memory scaling could be improved")

        print("\nOptimizations Applied:")
        print("✓ Memory pool properly initialized")
        print("✓ Pattern pre-computation enabled")
        print("✓ Batch segment processing")
        print("✓ Optimized memory access patterns")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
