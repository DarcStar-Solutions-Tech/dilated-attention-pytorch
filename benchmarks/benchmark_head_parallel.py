#!/usr/bin/env python3
"""
Benchmark head-parallel dilated attention.
Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_head_parallel.py
"""

import os
import time
import json
import torch
import torch.distributed as dist


def main():
    if "RANK" not in os.environ:
        print("Single GPU benchmark...")
        single_gpu_benchmark()
        print(
            "\nFor multi-GPU, run with: torchrun --nproc_per_node=2 benchmarks/benchmark_head_parallel.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    _ = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[GPU {rank}] Starting head-parallel benchmark...")

    # Import implementations
    from dilated_attention_pytorch.head_parallel_dilated_attention import (
        HeadParallelDilatedAttention,
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

        if seq_len % max(segment_lengths) != 0:
            continue

        if rank == 0:
            print(f"\nTesting seq_len={seq_len}...")

        try:
            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create head-parallel model
            model = HeadParallelDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
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
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2

            # Calculate metrics
            tokens = batch_size * seq_len
            throughput = tokens / avg_time
            memory_per_token = peak_memory / tokens * 1024

            result = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "time_ms": avg_time * 1000,
                "memory_mb": peak_memory,
                "throughput": throughput,
                "memory_per_token_kb": memory_per_token,
            }

            results.append(result)

            if rank == 0:
                print(
                    f"Head-Parallel @ {seq_len}: {avg_time * 1000:.1f}ms, "
                    f"{peak_memory:.1f}MB, {memory_per_token:.2f}KB/token, "
                    f"{throughput:.0f} tok/s"
                )

        except Exception as e:
            if rank == 0:
                print(f"Error at seq_len={seq_len}: {e}")
            break

    # Synchronize before comparison
    dist.barrier()

    # Only rank 0 does comparison
    if rank == 0 and results:
        print("\n" + "=" * 60)
        print("HEAD-PARALLEL vs RING ATTENTION COMPARISON")
        print("=" * 60)

        # Try to load ring attention results for comparison
        import glob

        ring_files = sorted(glob.glob("benchmarks/hybrid_results_rank0_2gpu_*.json"))

        if ring_files:
            with open(ring_files[-1], "r") as f:
                ring_data = json.load(f)
                ring_results = {r["seq_len"]: r for r in ring_data.get("results", [])}

            print(
                f"\n{'Seq Length':>10} | {'Ring (ms)':>12} | {'Head-Par (ms)':>14} | {'Speedup':>10} | {'Mem Saving':>12}"
            )
            print("-" * 70)

            total_speedup = 0
            count = 0

            for hp_result in results:
                seq = hp_result["seq_len"]
                if seq in ring_results:
                    ring = ring_results[seq]
                    speedup = ring["time_ms"] / hp_result["time_ms"]
                    mem_save = (
                        1
                        - hp_result["memory_per_token_kb"] / ring["memory_per_token_kb"]
                    ) * 100

                    print(
                        f"{seq:>10,} | {ring['time_ms']:>12.1f} | {hp_result['time_ms']:>14.1f} | "
                        f"{speedup:>9.2f}x | {mem_save:>11.1f}%"
                    )

                    total_speedup += speedup
                    count += 1

            if count > 0:
                print(f"\nAverage speedup: {total_speedup / count:.2f}x")

        print("\n" + "=" * 60)
        print("KEY ADVANTAGES OF HEAD-PARALLEL:")
        print("=" * 60)
        print("✓ Each GPU processes COMPLETE sequences")
        print("✓ No segment boundary issues")
        print("✓ Single AllGather vs multiple ring passes")
        print("✓ Better GPU utilization (larger ops)")
        print("✓ Preserves dilation pattern locality")

    dist.destroy_process_group()


def single_gpu_benchmark():
    """Benchmark on single GPU for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from dilated_attention_pytorch.head_parallel_dilated_attention import (
        HeadParallelDilatedAttention,
    )

    # Test small sequence
    model = HeadParallelDilatedAttention(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        dropout=0.0,
        device=device,
    )

    seq_len = 4096
    batch_size = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Benchmark
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        output = model(q, k, v)
        torch.cuda.synchronize()
        elapsed = time.time() - start

    print(f"Single GPU test: seq_len={seq_len}, time={elapsed * 1000:.1f}ms")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
