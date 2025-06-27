#!/usr/bin/env python3
"""
Multi-GPU Ring Attention benchmark using both available GPUs.

This script properly initializes PyTorch distributed environment and
demonstrates true Ring Attention benefits across multiple GPUs.
"""

import os
import gc
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import sys
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention_fixed import (
    RingDilatedAttentionFixed,
)

# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed environment."""
    dist.destroy_process_group()


def benchmark_ring_attention_gpu(
    rank: int,
    world_size: int,
    seq_len: int,
    implementation: str,
    results_dict: Optional[Dict] = None,
):
    """Benchmark Ring Attention on a specific GPU."""

    # Setup distributed environment
    setup_distributed(rank, world_size)

    # Configuration
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16

    try:
        # Create module based on implementation
        if implementation == "current":
            module = RingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=world_size,
                device=device,
                dtype=dtype,
            )
        else:  # "fixed"
            module = RingDilatedAttentionFixed(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=world_size,
                device=device,
                dtype=dtype,
            )

        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Create inputs - each GPU has the full sequence
        # This is important for understanding memory usage
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Synchronize before measurement
        torch.cuda.synchronize(device)
        dist.barrier()

        # Warmup
        with torch.no_grad():
            _ = module(q, k, v)

        torch.cuda.synchronize(device)
        dist.barrier()

        # Measure memory after allocation
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)

        # Time the forward pass
        start_time = time.time()
        with torch.no_grad():
            _ = module(q, k, v)

        torch.cuda.synchronize(device)
        dist.barrier()
        end_time = time.time()

        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Only rank 0 reports results
        if rank == 0:
            time_ms = (end_time - start_time) * 1000
            throughput = (seq_len * batch_size) / (end_time - start_time)

            result = {
                "implementation": implementation,
                "seq_len": seq_len,
                "world_size": world_size,
                "time_ms": time_ms,
                "allocated_memory_gb": allocated_memory,
                "peak_memory_gb": peak_memory,
                "throughput_tokens_per_sec": throughput,
                "success": True,
            }

            if results_dict is not None:
                results_dict[f"{implementation}_{seq_len}_{world_size}"] = result

            print(
                f"  ✓ {implementation}: {time_ms:.1f}ms, Peak: {peak_memory:.3f}GB, Allocated: {allocated_memory:.3f}GB"
            )

    except Exception as e:
        if rank == 0:
            print(f"  ✗ {implementation}: {str(e)[:50]}...")
            if results_dict is not None:
                results_dict[f"{implementation}_{seq_len}_{world_size}"] = {
                    "implementation": implementation,
                    "seq_len": seq_len,
                    "world_size": world_size,
                    "success": False,
                    "error": str(e),
                }

    finally:
        # Cleanup
        cleanup_distributed()


def run_single_gpu_baseline(seq_len: int, implementation: str) -> Dict:
    """Run single GPU baseline for comparison."""
    device = torch.device("cuda:0")
    dtype = torch.float16
    batch_size = 1
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Clear memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        # Create module
        if implementation == "current":
            module = RingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=1,
                device=device,
                dtype=dtype,
            )
        else:
            module = RingDilatedAttentionFixed(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=1,
                device=device,
                dtype=dtype,
            )

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        with torch.no_grad():
            _ = module(q, k, v)

        torch.cuda.synchronize()

        # Measure
        allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)

        start_time = time.time()
        with torch.no_grad():
            output = module(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()

        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**3)
        time_ms = (end_time - start_time) * 1000
        throughput = (seq_len * batch_size) / (end_time - start_time)

        print(
            f"  ✓ {implementation} (1 GPU): {time_ms:.1f}ms, Peak: {peak_memory:.3f}GB, Allocated: {allocated_memory:.3f}GB"
        )

        return {
            "implementation": implementation,
            "seq_len": seq_len,
            "world_size": 1,
            "time_ms": time_ms,
            "allocated_memory_gb": allocated_memory,
            "peak_memory_gb": peak_memory,
            "throughput_tokens_per_sec": throughput,
            "success": True,
        }

    except Exception as e:
        print(f"  ✗ {implementation} (1 GPU): {str(e)[:50]}...")
        return {
            "implementation": implementation,
            "seq_len": seq_len,
            "world_size": 1,
            "success": False,
            "error": str(e),
        }
    finally:
        # Cleanup
        del q, k, v
        if "output" in locals():
            del output
        torch.cuda.empty_cache()
        gc.collect()


def main():
    """Main benchmark function."""
    print("Multi-GPU Ring Attention Benchmark")
    print("=" * 80)

    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if num_gpus < 2:
        print("ERROR: This benchmark requires at least 2 GPUs")
        return

    # Test configurations
    test_configs = [
        # (seq_len, use_multi_gpu)
        (4096, False),  # Single GPU baseline
        (4096, True),  # 2 GPU Ring Attention
        (8192, False),  # Single GPU baseline
        (8192, True),  # 2 GPU Ring Attention
        (16384, False),  # Single GPU baseline
        (16384, True),  # 2 GPU Ring Attention
        (32768, False),  # Single GPU baseline
        (32768, True),  # 2 GPU Ring Attention
        (65536, True),  # 2 GPU only (too large for single)
        (131072, True),  # 2 GPU only
    ]

    results = []

    for seq_len, use_multi_gpu in test_configs:
        print(f"\nSequence length: {seq_len:,}, GPUs: {2 if use_multi_gpu else 1}")

        if use_multi_gpu:
            # Test both implementations with multi-GPU
            for impl in ["current", "fixed"]:
                print(f"  Testing {impl} implementation (2 GPUs):")

                # Use multiprocessing to run on multiple GPUs
                manager = mp.Manager()
                results_dict = manager.dict()

                processes = []
                for rank in range(2):
                    p = mp.Process(
                        target=benchmark_ring_attention_gpu,
                        args=(rank, 2, seq_len, impl, results_dict),
                    )
                    p.start()
                    processes.append(p)

                # Wait for all processes
                for p in processes:
                    p.join()

                # Collect results
                for key, result in results_dict.items():
                    results.append(result)
        else:
            # Single GPU baseline
            print("  Testing single GPU baseline:")
            for impl in ["current", "fixed"]:
                result = run_single_gpu_baseline(seq_len, impl)
                results.append(result)

    # Analysis
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    # Compare single vs multi-GPU
    for seq_len in [4096, 8192, 16384, 32768]:
        print(f"\nSequence {seq_len:,}:")

        # Find results
        single_results = [
            r
            for r in results
            if r["seq_len"] == seq_len and r["world_size"] == 1 and r["success"]
        ]
        multi_results = [
            r
            for r in results
            if r["seq_len"] == seq_len and r["world_size"] == 2 and r["success"]
        ]

        if single_results and multi_results:
            for impl in ["current", "fixed"]:
                single = next(
                    (r for r in single_results if r["implementation"] == impl), None
                )
                multi = next(
                    (r for r in multi_results if r["implementation"] == impl), None
                )

                if single and multi:
                    mem_reduction = (
                        1 - multi["peak_memory_gb"] / single["peak_memory_gb"]
                    ) * 100
                    speedup = single["time_ms"] / multi["time_ms"]

                    print(f"  {impl}:")
                    print(
                        f"    1 GPU: {single['peak_memory_gb']:.3f}GB, {single['time_ms']:.1f}ms"
                    )
                    print(
                        f"    2 GPUs: {multi['peak_memory_gb']:.3f}GB, {multi['time_ms']:.1f}ms"
                    )
                    print(f"    Memory reduction: {mem_reduction:.1f}%")
                    print(f"    Speedup: {speedup:.2f}x")

    # Maximum sequence achieved
    print("\n" + "=" * 80)
    print("MAXIMUM SEQUENCES")
    print("=" * 80)

    max_single = max(
        [r["seq_len"] for r in results if r["world_size"] == 1 and r["success"]],
        default=0,
    )
    max_multi = max(
        [r["seq_len"] for r in results if r["world_size"] == 2 and r["success"]],
        default=0,
    )

    print(f"Maximum with 1 GPU: {max_single:,} tokens")
    print(f"Maximum with 2 GPUs: {max_multi:,} tokens")
    print(f"Improvement: {max_multi / max_single:.1f}x")

    # Save results
    output_manager = BenchmarkOutputManager(
        benchmark_type="ring-attention-multi-gpu",
        parameters={
            "num_gpus": 2,
            "implementations": ["current", "fixed"],
            "max_seq_len": max(r["seq_len"] for r in results if r["success"]),
        },
    )

    output_manager.add_result("results", results)
    output_manager.add_result("num_gpus", num_gpus)

    json_path, md_path = output_manager.save_results()
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
