#!/usr/bin/env python3
"""
Standalone benchmark for Hybrid Ring Attention implementation.
Measures performance and memory usage across different configurations.

Single GPU: python benchmarks/benchmark_hybrid_standalone.py
Multi-GPU: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_standalone.py
"""

import os
import gc
import time
import json
import torch
import torch.distributed as dist
from datetime import datetime
from typing import Dict
import numpy as np

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def measure_memory_and_time(
    model,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_warmup: int = 3,
    num_iterations: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Dict:
    """Measure memory usage and execution time for the hybrid model."""

    device = device or torch.cuda.current_device()
    # Use model's dtype if not specified
    dtype = dtype or model.dtype

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Memory before creating inputs
    mem_start = torch.cuda.memory_allocated(device) / (1024**2)

    # Create inputs with controlled values
    torch.manual_seed(42)
    scale = 0.1 / (seq_len**0.25)  # Adaptive scaling

    q = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        * scale
    )
    k = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        * scale
    )
    v = (
        torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        * scale
    )

    # Memory after inputs
    mem_inputs = torch.cuda.memory_allocated(device) / (1024**2)
    input_memory = mem_inputs - mem_start

    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

    # Reset peak memory stats after warmup
    torch.cuda.reset_peak_memory_stats()
    mem_before_forward = torch.cuda.memory_allocated(device) / (1024**2)

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    # Memory statistics
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    _ = torch.cuda.memory_allocated(device) / (1024**2)
    forward_memory = peak_memory - mem_before_forward

    # Validate output
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    output_mean = output.float().mean().item()
    output_std = output.float().std().item()

    # Clean up
    del q, k, v, output
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "avg_time_ms": np.mean(times) * 1000,
        "std_time_ms": np.std(times) * 1000,
        "min_time_ms": np.min(times) * 1000,
        "max_time_ms": np.max(times) * 1000,
        "input_memory_mb": input_memory,
        "forward_memory_mb": forward_memory,
        "peak_memory_mb": peak_memory,
        "throughput_tokens_per_sec": seq_len / np.mean(times),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "output_mean": output_mean,
        "output_std": output_std,
    }


def benchmark_hybrid():
    """Run comprehensive benchmark of hybrid implementation."""

    # Check if distributed
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("Hybrid Ring Attention Benchmark")
        print("=" * 60)
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"World size: {world_size} GPU(s)")
        print(f"Device: {device}")
        print()

    # Test configurations
    test_configs = [
        # (seq_len, segment_len, dilation_rate, batch_size, name)
        (512, 256, 1, 2, "Small"),
        (1024, 512, 1, 2, "Small+"),
        (2048, 1024, 1, 2, "Medium"),
        (4096, 2048, 1, 1, "Large"),
        (8192, 4096, 1, 1, "XLarge"),
        # Test with dilation
        (2048, 512, 2, 2, "Medium-Dilated"),
        (4096, 1024, 2, 1, "Large-Dilated"),
    ]

    all_results = []

    for seq_len, segment_len, dilation_rate, batch_size, config_name in test_configs:
        if rank == 0:
            print(f"\nTesting {config_name}:")
            print(
                f"  Sequence: {seq_len}, Segment: {segment_len}, Dilation: {dilation_rate}"
            )
            print(f"  Batch size: {batch_size}")

        # Synchronize all ranks
        if world_size > 1:
            dist.barrier()

        try:
            # Create model (let it auto-select dtype based on GPU)
            model = RingDilatedAttentionHybrid(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                # dtype auto-selected based on GPU architecture
                enable_memory_pool=True,
                use_pattern_cache=True,
                use_flash_attention=False,  # Disable for consistent comparison
            )

            # Run benchmark (using model's auto-selected dtype)
            result = measure_memory_and_time(
                model,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                # dtype will use model's dtype
            )

            # Add configuration info
            result["config_name"] = config_name
            result["world_size"] = world_size
            result["segment_len"] = segment_len
            result["dilation_rate"] = dilation_rate

            # Gather results from all ranks if distributed
            if world_size > 1:
                all_rank_results = [None] * world_size
                dist.all_gather_object(all_rank_results, result)

                if rank == 0:
                    # Aggregate statistics
                    avg_time = np.mean([r["avg_time_ms"] for r in all_rank_results])
                    max_peak_mem = max(r["peak_memory_mb"] for r in all_rank_results)
                    avg_peak_mem = np.mean(
                        [r["peak_memory_mb"] for r in all_rank_results]
                    )
                    any_nan = any(r["has_nan"] for r in all_rank_results)
                    any_inf = any(r["has_inf"] for r in all_rank_results)

                    # Store aggregated result
                    aggregated = {
                        **result,
                        "avg_time_ms_all_gpus": avg_time,
                        "avg_peak_memory_mb": avg_peak_mem,
                        "max_peak_memory_mb": max_peak_mem,
                        "all_rank_results": all_rank_results,
                        "any_nan": any_nan,
                        "any_inf": any_inf,
                    }
                    all_results.append(aggregated)

                    # Print summary
                    print("  ✅ Success!")
                    print(f"     Time: {avg_time:.2f}ms (avg across GPUs)")
                    print(
                        f"     Memory: {avg_peak_mem:.1f}MB avg, {max_peak_mem:.1f}MB max"
                    )
                    print(
                        f"     Throughput: {seq_len * batch_size / (avg_time / 1000):.0f} tokens/sec"
                    )
                    print(
                        f"     Per-GPU memory: {avg_peak_mem:.1f}MB (expected ~{seq_len * batch_size * 8 * 64 * 4 / (1024**2) / world_size:.1f}MB for K,V)"
                    )

                    if any_nan or any_inf:
                        print(f"     ⚠️  WARNING: NaN={any_nan}, Inf={any_inf}")
            else:
                # Single GPU
                all_results.append(result)

                if rank == 0:
                    print("  ✅ Success!")
                    print(
                        f"     Time: {result['avg_time_ms']:.2f}ms ± {result['std_time_ms']:.2f}ms"
                    )
                    print(f"     Memory: {result['peak_memory_mb']:.1f}MB peak")
                    print(
                        f"     Throughput: {result['throughput_tokens_per_sec']:.0f} tokens/sec"
                    )

                    if result["has_nan"] or result["has_inf"]:
                        print(
                            f"     ⚠️  WARNING: NaN={result['has_nan']}, Inf={result['has_inf']}"
                        )

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Failed: {e}")
                all_results.append(
                    {
                        "config_name": config_name,
                        "seq_len": seq_len,
                        "error": str(e),
                    }
                )

        # Small delay between tests
        time.sleep(0.5)

    # Save results
    if rank == 0:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"benchmarks/hybrid_benchmark_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Benchmark Summary")
        print(f"{'=' * 60}")

        # Summary statistics
        successful = [r for r in all_results if "error" not in r]
        if successful:
            print(f"\nSuccessful configurations: {len(successful)}/{len(all_results)}")

            # Memory efficiency analysis
            print("\nMemory Efficiency:")
            for r in successful:
                seq_len = r["seq_len"]
                batch_size = r["batch_size"]
                peak_mem = r.get("avg_peak_memory_mb", r["peak_memory_mb"])

                # Theoretical memory for K,V storage
                # Each GPU stores: Q (full) + K,V (1/world_size of full)
                kv_elements_per_gpu = seq_len * batch_size * 8 * 64 / world_size
                q_elements = seq_len * batch_size * 8 * 64
                theoretical_mb = (
                    (q_elements + 2 * kv_elements_per_gpu) * 2 / (1024**2)
                )  # *2 for fp16

                efficiency = theoretical_mb / peak_mem * 100
                print(
                    f"  {r['config_name']}: {peak_mem:.1f}MB used, {theoretical_mb:.1f}MB theoretical ({efficiency:.1f}% efficiency)"
                )

            # Performance summary
            print("\nPerformance Summary:")
            print(f"{'Config':<20} {'Avg Time (ms)':<15} {'Throughput (tok/s)':<20}")
            print("-" * 55)

            for r in successful:
                avg_time = r.get("avg_time_ms_all_gpus", r["avg_time_ms"])
                throughput = r["seq_len"] * r["batch_size"] / (avg_time / 1000)
                print(f"{r['config_name']:<20} {avg_time:<15.2f} {throughput:<20.0f}")

        print(f"\nResults saved to: {filename}")

        # Memory scaling analysis
        if world_size > 1:
            print(f"\nMemory Scaling (World size = {world_size}):")
            print(
                f"Expected reduction vs single GPU: {(1 - 1 / world_size) * 100:.1f}%"
            )
            print("Actual reduction: See per-configuration efficiency above")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_hybrid()
