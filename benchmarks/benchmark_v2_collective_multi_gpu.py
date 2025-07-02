#!/usr/bin/env python3
"""
Comprehensive multi-GPU benchmark for V2 Collective.
Tests actual distributed performance, memory scaling, and maximum sequence lengths.
"""

import os
import gc
import time
import json
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup_distributed(rank, world_size):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def get_gpu_memory_info(device_id):
    """Get memory info for specific GPU."""
    torch.cuda.set_device(device_id)
    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    free = total - reserved
    return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}


def benchmark_sequence(
    rank,
    world_size,
    seq_len,
    batch_size,
    num_heads,
    head_dim,
    segment_lengths,
    dilation_rates,
    iterations=10,
):
    """Benchmark a specific configuration on multiple GPUs."""

    # Set up process
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    try:
        # Import here to avoid issues with multiprocessing
        from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )

        # Synchronize before starting
        dist.barrier()

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()

        # Create model
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            dtype=torch.float16,
            ring_size=world_size,
            enable_memory_pool=True,
        )

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

        # Get memory after allocation
        _ = get_gpu_memory_info(rank)

        # Warmup
        for _ in range(3):
            _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

        dist.barrier()

        # Time iterations
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(iterations):
            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

        dist.barrier()
        end_time = time.perf_counter()

        # Calculate metrics
        avg_time = (end_time - start_time) / iterations * 1000  # ms
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
        mem_final = get_gpu_memory_info(rank)

        # Gather results from all ranks
        results = {
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "avg_time_ms": avg_time,
            "peak_memory_mb": peak_memory,
            "memory_allocated_gb": mem_final["allocated"],
            "memory_free_gb": mem_final["free"],
            "throughput_tokens_per_sec": (seq_len * batch_size) / (avg_time / 1000),
            "success": True,
        }

        # Clean up
        del q, k, v, output, model
        torch.cuda.empty_cache()
        gc.collect()

        cleanup_distributed()
        return results

    except Exception as e:
        cleanup_distributed()
        return {
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "success": False,
            "error": str(e),
        }


def run_benchmark(world_size, configurations):
    """Run benchmarks across multiple configurations."""
    all_results = []

    for config in configurations:
        seq_len = config["seq_len"]
        batch_size = config.get("batch_size", 1)
        num_heads = config.get("num_heads", 8)
        head_dim = config.get("head_dim", 64)
        segment_lengths = config["segment_lengths"]
        dilation_rates = config["dilation_rates"]

        print(f"\nTesting {world_size} GPUs: seq_len={seq_len:,}, batch={batch_size}")
        print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

        # Run on all GPUs
        ctx = mp.get_context("spawn")
        processes = []
        results_queue = ctx.Queue()

        def worker(rank):
            result = benchmark_sequence(
                rank,
                world_size,
                seq_len,
                batch_size,
                num_heads,
                head_dim,
                segment_lengths,
                dilation_rates,
            )
            results_queue.put(result)

        # Start processes
        for rank in range(world_size):
            p = ctx.Process(target=worker, args=(rank,))
            p.start()
            processes.append(p)

        # Wait for completion
        for p in processes:
            p.join()

        # Collect results
        gpu_results = []
        while not results_queue.empty():
            gpu_results.append(results_queue.get())

        # Aggregate results
        if all(r["success"] for r in gpu_results):
            avg_time = sum(r["avg_time_ms"] for r in gpu_results) / len(gpu_results)
            total_memory = sum(r["peak_memory_mb"] for r in gpu_results)
            total_throughput = sum(r["throughput_tokens_per_sec"] for r in gpu_results)

            result = {
                "world_size": world_size,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "avg_time_ms": avg_time,
                "total_memory_mb": total_memory,
                "avg_memory_per_gpu_mb": total_memory / world_size,
                "total_throughput": total_throughput,
                "speedup": 1.0,  # Will calculate relative to 1 GPU
                "success": True,
                "gpu_results": gpu_results,
            }

            print(
                f"  ✓ Success: {avg_time:.1f}ms, {total_memory:.0f}MB total, "
                f"{total_throughput:,.0f} tokens/sec"
            )
        else:
            errors = [
                r.get("error", "Unknown") for r in gpu_results if not r["success"]
            ]
            result = {
                "world_size": world_size,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "success": False,
                "errors": errors,
            }
            print(f"  ✗ Failed: {errors[0] if errors else 'Unknown error'}")

        all_results.append(result)

    return all_results


def test_scaling():
    """Test scaling from 1 to N GPUs."""
    # Detect available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")

    if n_gpus < 2:
        print("This test requires at least 2 GPUs")
        return

    # Test configurations
    configs = [
        # Small sequences to test overhead
        {
            "seq_len": 4096,
            "batch_size": 4,
            "segment_lengths": [512, 1024],
            "dilation_rates": [1, 2],
        },
        # Medium sequences
        {
            "seq_len": 16384,
            "batch_size": 2,
            "segment_lengths": [2048, 4096],
            "dilation_rates": [1, 2],
        },
        # Large sequences
        {
            "seq_len": 65536,
            "batch_size": 1,
            "segment_lengths": [4096, 8192],
            "dilation_rates": [1, 2],
        },
        # Very large sequences (multi-GPU enables larger)
        {
            "seq_len": 262144,
            "batch_size": 1,
            "segment_lengths": [8192, 16384],
            "dilation_rates": [1, 2],
        },
        # Test if we can go beyond single GPU limit
        {
            "seq_len": 524288,
            "batch_size": 1,
            "segment_lengths": [16384, 32768],
            "dilation_rates": [1, 2],
        },
    ]

    all_results = {}

    # Test with different GPU counts
    for world_size in [1, 2, min(4, n_gpus), n_gpus]:
        if world_size > n_gpus:
            continue

        print(f"\n{'=' * 80}")
        print(f"Testing with {world_size} GPU(s)")
        print("=" * 80)

        results = run_benchmark(world_size, configs)
        all_results[world_size] = results

    return all_results


def analyze_results(all_results):
    """Analyze and display results."""
    print("\n" + "=" * 80)
    print("MULTI-GPU SCALING ANALYSIS")
    print("=" * 80)

    # Calculate speedups
    if 1 in all_results:
        single_gpu_times = {}
        for result in all_results[1]:
            if result["success"]:
                key = (result["seq_len"], result["batch_size"])
                single_gpu_times[key] = result["avg_time_ms"]

        # Update speedups
        for world_size, results in all_results.items():
            for result in results:
                if result["success"]:
                    key = (result["seq_len"], result["batch_size"])
                    if key in single_gpu_times:
                        result["speedup"] = (
                            single_gpu_times[key] / result["avg_time_ms"]
                        )

    # Display scaling table
    print("\nScaling Results (Speedup relative to 1 GPU):")
    print("-" * 80)
    print(
        f"{'Seq Length':<15} {'Batch':<10} {'1 GPU':<15} {'2 GPUs':<15} {'4 GPUs':<15}"
    )
    print("-" * 80)

    # Organize by sequence length
    seq_lengths = set()
    for results in all_results.values():
        for r in results:
            if r["success"]:
                seq_lengths.add((r["seq_len"], r["batch_size"]))

    for seq_len, batch_size in sorted(seq_lengths):
        line = f"{seq_len:<15,} {batch_size:<10}"

        for world_size in [1, 2, 4]:
            if world_size in all_results:
                result = next(
                    (
                        r
                        for r in all_results[world_size]
                        if r["success"]
                        and r["seq_len"] == seq_len
                        and r["batch_size"] == batch_size
                    ),
                    None,
                )
                if result:
                    speedup = result.get("speedup", 1.0)
                    line += f"{speedup:<15.2f}x"
                else:
                    line += f"{'N/A':<15}"
            else:
                line += f"{'-':<15}"

        print(line)

    # Memory scaling
    print("\n\nMemory Usage (MB per GPU):")
    print("-" * 80)
    print(f"{'Seq Length':<15} {'1 GPU':<15} {'2 GPUs':<15} {'4 GPUs':<15}")
    print("-" * 80)

    for seq_len, batch_size in sorted(seq_lengths):
        line = f"{seq_len:<15,}"

        for world_size in [1, 2, 4]:
            if world_size in all_results:
                result = next(
                    (
                        r
                        for r in all_results[world_size]
                        if r["success"]
                        and r["seq_len"] == seq_len
                        and r["batch_size"] == batch_size
                    ),
                    None,
                )
                if result:
                    mem_per_gpu = result.get("avg_memory_per_gpu_mb", 0)
                    line += f"{mem_per_gpu:<15.0f}"
                else:
                    line += f"{'OOM':<15}"
            else:
                line += f"{'-':<15}"

        print(line)

    # Find maximum sequence lengths
    print("\n\nMaximum Sequence Lengths:")
    print("-" * 60)

    for world_size in sorted(all_results.keys()):
        max_seq = 0
        for result in all_results[world_size]:
            if result["success"]:
                max_seq = max(max_seq, result["seq_len"])

        if max_seq > 0:
            print(f"{world_size} GPU(s): {max_seq:,} tokens")

    return all_results


def main():
    """Main benchmark function."""
    print("=" * 80)
    print("V2 Collective Multi-GPU Performance Verification")
    print("=" * 80)

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"\nDetected {n_gpus} GPU(s):")
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({mem_gb:.1f}GB)")

    if n_gpus < 2:
        print("\n⚠️  This benchmark requires at least 2 GPUs for multi-GPU testing.")
        print("Running single-GPU benchmark only...")

    # Run tests
    results = test_scaling()

    # Analyze results
    analyzed = analyze_results(results)

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_file = f"benchmarks/v2_collective_multi_gpu_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(analyzed, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    # Check if we achieved true ring attention
    if 2 in results:
        two_gpu_results = [r for r in results[2] if r["success"]]
        if two_gpu_results:
            # Check memory scaling
            for r in two_gpu_results:
                if "gpu_results" in r and len(r["gpu_results"]) >= 2:
                    gpu0_mem = r["gpu_results"][0]["peak_memory_mb"]
                    gpu1_mem = r["gpu_results"][1]["peak_memory_mb"]

                    # In true ring attention, each GPU should have ~1/2 the memory
                    if 1 in results:
                        single_result = next(
                            (
                                s
                                for s in results[1]
                                if s["success"] and s["seq_len"] == r["seq_len"]
                            ),
                            None,
                        )
                        if single_result:
                            single_mem = single_result["total_memory_mb"]
                            mem_ratio = (gpu0_mem + gpu1_mem) / single_mem

                            if mem_ratio < 0.6:
                                print(
                                    f"✓ True ring attention achieved for {r['seq_len']:,} tokens!"
                                )
                                print(
                                    f"  Memory: {single_mem:.0f}MB (1 GPU) → "
                                    f"{gpu0_mem:.0f}+{gpu1_mem:.0f}MB (2 GPUs)"
                                )
                            else:
                                print(
                                    f"✗ All-gather pattern detected for {r['seq_len']:,} tokens"
                                )
                                print(
                                    f"  Memory: {single_mem:.0f}MB (1 GPU) → "
                                    f"{gpu0_mem:.0f}+{gpu1_mem:.0f}MB (2 GPUs)"
                                )
                                print(
                                    f"  Ratio: {mem_ratio:.2f} (expect <0.6 for true ring)"
                                )


if __name__ == "__main__":
    main()
