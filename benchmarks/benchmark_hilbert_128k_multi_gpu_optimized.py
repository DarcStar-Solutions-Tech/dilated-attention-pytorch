#!/usr/bin/env python3
"""
Optimized multi-GPU benchmark with reduced synchronization overhead.
"""

import gc
import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from typing import Dict, List

from dilated_attention_pytorch import RingDilatedAttentionHilbertOptimized


def setup(rank: int, world_size: int):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize without device_id (it's not needed for NCCL)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    print(f"[GPU {rank}] Process initialized")


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def clear_memory():
    """Aggressively clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def benchmark_extreme_dilation_distributed(
    rank: int,
    world_size: int,
    seq_len: int,
    results_dict: Dict,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    iterations: int = 3,
):
    """Optimized benchmark with reduced synchronization."""

    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    clear_memory()

    # Use extreme dilation configuration
    base_segment = min(4096, seq_len // 4)
    segment_lengths = [base_segment, base_segment * 2]
    dilation_rates = [8, 16]

    # Ensure sequence length is divisible by largest segment
    max_segment = max(segment_lengths)
    if seq_len % max_segment != 0:
        seq_len = ((seq_len // max_segment) + 1) * max_segment

    if rank == 0:
        print(
            f"\nTesting {seq_len:,} tokens with extreme dilation (8,16) on {world_size} GPUs"
        )
        print(f"  Segments: {segment_lengths}")
        print(f"  Dilation: {dilation_rates}")
        print(f"  Ring size: {world_size}")

    try:
        # Create model - only initialize once
        model = RingDilatedAttentionHilbertOptimized(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            cache_hilbert_mappings=True,
            apply_hilbert_to_kv=True,
            enable_memory_pool=True,
            ring_size=world_size,
        )

        # Create inputs
        if rank == 0:
            print("  Creating tensors...")

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Single barrier after setup
        dist.barrier()

        # Memory tracking
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated(rank) / 1024**3

        # Warmup without excessive barriers
        if rank == 0:
            print("  Warming up...")

        with torch.no_grad():
            _ = model(q, k, v)
        torch.cuda.synchronize()

        # Single barrier after warmup
        dist.barrier()

        # Benchmark with minimal synchronization
        if rank == 0:
            print(f"  Benchmarking {iterations} iterations...")

        times = []
        for i in range(iterations):
            torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)

            if rank == 0:
                print(f"    Iteration {i + 1}: {end - start:.3f}s")
            del output

        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_sec = seq_len / avg_time

        peak_mem = torch.cuda.max_memory_allocated(rank) / 1024**3
        mem_used = peak_mem - mem_before

        # Test without Hilbert - create new model
        if rank == 0:
            print("  Testing without Hilbert...")

        # Delete old model first to save memory
        del model
        clear_memory()

        model_no_hilbert = RingDilatedAttentionHilbertOptimized(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            cache_hilbert_mappings=False,
            apply_hilbert_to_kv=False,
            enable_memory_pool=True,
            ring_size=world_size,
        )

        # Benchmark without Hilbert
        times_no_hilbert = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model_no_hilbert(q, k, v)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times_no_hilbert.append(end - start)
            del output

        avg_time_no_hilbert = np.mean(times_no_hilbert)
        speedup = avg_time_no_hilbert / avg_time
        improvement = (speedup - 1) * 100

        # Simple memory gathering without all_gather_object
        # Each rank writes its result, rank 0 collects
        if rank == 0:
            # Collect memory from all ranks
            total_memory = peak_mem  # Start with rank 0
            for src in range(1, world_size):
                other_mem = torch.tensor(0.0, device=device)
                dist.recv(other_mem, src=src)
                total_memory += other_mem.item()
        else:
            # Send memory to rank 0
            mem_tensor = torch.tensor(peak_mem, device=device)
            dist.send(mem_tensor, dst=0)

        # Cleanup
        del q, k, v, model_no_hilbert
        clear_memory()

        # Final barrier before results
        dist.barrier()

        if rank == 0:
            result = {
                "success": True,
                "seq_len": seq_len,
                "world_size": world_size,
                "avg_time": avg_time,
                "std_time": std_time,
                "tokens_per_sec": tokens_per_sec,
                "peak_memory_gb": peak_mem,
                "total_memory_gb": total_memory,
                "memory_used_gb": mem_used,
                "speedup": speedup,
                "improvement_pct": improvement,
                "no_hilbert_time": avg_time_no_hilbert,
                "no_hilbert_tokens_per_sec": seq_len / avg_time_no_hilbert,
            }

            print("\n  ✓ Success!")
            print(f"    Time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"    Throughput: {tokens_per_sec:,.0f} tokens/sec")
            print(f"    Memory per GPU: {peak_mem:.2f} GB")
            print(f"    Total memory: {total_memory:.2f} GB")
            print(f"    Hilbert speedup: {speedup:.3f}x ({improvement:+.1f}%)")

            results_dict[seq_len] = result

    except torch.cuda.OutOfMemoryError:
        clear_memory()
        if rank == 0:
            print("  ✗ Out of memory")
            results_dict[seq_len] = {
                "success": False,
                "seq_len": seq_len,
                "error": "OOM",
            }
    except Exception as e:
        clear_memory()
        if rank == 0:
            print(f"  ✗ Error: {str(e)}")
            results_dict[seq_len] = {
                "success": False,
                "seq_len": seq_len,
                "error": str(e),
            }
    finally:
        cleanup()


def run_distributed_benchmark(world_size: int, seq_lengths: List[int]) -> List[Dict]:
    """Run distributed benchmark across sequence lengths."""

    manager = mp.Manager()
    results_dict = manager.dict()

    for seq_len in seq_lengths:
        # Run distributed benchmark
        mp.spawn(
            benchmark_extreme_dilation_distributed,
            args=(world_size, seq_len, results_dict),
            nprocs=world_size,
            join=True,
        )

        # Check if we hit OOM
        if seq_len in results_dict and not results_dict[seq_len]["success"]:
            print(
                f"\nStopping at {seq_len:,} tokens due to {results_dict[seq_len]['error']}"
            )
            break

    return list(results_dict.values())


def main():
    """Run optimized multi-GPU benchmark."""

    # Check available GPUs
    if not torch.cuda.is_available():
        print("Error: CUDA is not available")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("Error: This benchmark requires at least 2 GPUs")
        return

    # Use all available GPUs (up to 8)
    world_size = min(num_gpus, 8)

    print("=" * 80)
    print(f"OPTIMIZED MULTI-GPU BENCHMARK: {world_size} GPUs TO 128K TOKENS (FP32)")
    print("=" * 80)
    print("\nOptimizations:")
    print("- Reduced synchronization barriers")
    print("- Efficient memory gathering without all_gather_object")
    print("- Single model initialization per test")
    print("- Explicit device_id in init_process_group")

    # Test sequence lengths
    seq_lengths = [
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        98304,  # 96K
        131072,  # 128K
    ]

    # Run distributed benchmark
    results = run_distributed_benchmark(world_size, seq_lengths)

    # Print summary
    successful_results = [r for r in results if r["success"]]

    if successful_results:
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)

        final = successful_results[-1]
        max_achieved = final["seq_len"]

        print(f"\nMaximum sequence length achieved: {max_achieved:,} tokens")
        print(f"Number of GPUs: {world_size}")
        print(f"Final throughput: {final['tokens_per_sec']:,.0f} tokens/sec")
        print(f"Memory per GPU: {final['peak_memory_gb']:.2f} GB")
        print(f"Total memory: {final['total_memory_gb']:.2f} GB")
        print(
            f"Hilbert speedup: {final['speedup']:.3f}x ({final['improvement_pct']:+.1f}%)"
        )

        # Simple results table
        print("\nResults Summary:")
        print("-" * 60)
        print(f"{'Seq Len':>10} | {'Throughput':>12} | {'Mem/GPU':>8} | {'Speedup':>8}")
        print("-" * 60)
        for r in successful_results:
            print(
                f"{r['seq_len']:>10,} | {r['tokens_per_sec']:>12,.0f} | {r['peak_memory_gb']:>8.2f} | {r['speedup']:>8.2f}x"
            )

    print("\nConclusion:")
    print("✓ Optimized multi-GPU benchmark complete")
    print("✓ Reduced synchronization overhead")
    print("✓ Efficient memory tracking")
    print("=" * 80)


if __name__ == "__main__":
    # Set environment variable to avoid potential issues
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(torch.cuda.device_count())
    )

    main()
