#!/usr/bin/env python3
"""
Comprehensive benchmark comparing optimized Ring Dilated Attention performance.

This script compares:
1. Original Ring Attention (before optimizations)
2. Optimized Ring Attention with all improvements:
   - Pattern caching enabled by default
   - Memory pool with 16MB threshold
   - Flash Attention/xformers backend
   - Smart dtype selection
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np
from typing import Dict
import os
import warnings

# Import different versions
from dilated_attention_pytorch import (
    RingDilatedAttentionV2Collective,
    RingDilatedAttentionV2Flash,
)


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed process group."""
    dist.destroy_process_group()


def benchmark_model(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    num_heads: int,
    head_dim: int,
    num_runs: int = 10,
    warmup_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark a model's performance."""
    device = model.device
    dtype = model.dtype

    # Create inputs
    q = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Measure
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Get memory stats
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        peak_memory = 0

    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "min_time_ms": np.min(times),
        "max_time_ms": np.max(times),
        "peak_memory_mb": peak_memory,
        "throughput": seq_length / (np.mean(times) / 1000),
    }


def test_single_gpu():
    """Test single GPU performance."""
    print("\n" + "=" * 80)
    print("SINGLE GPU PERFORMANCE COMPARISON")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("No GPU available")
        return None

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")

    # Test configurations
    configs = [
        {"batch_size": 1, "seq_length": 2048, "num_heads": 8, "head_dim": 64},
        {"batch_size": 2, "seq_length": 4096, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_length": 8192, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_length": 16384, "num_heads": 8, "head_dim": 64},
    ]

    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    results = {}

    for config in configs:
        print(
            f"\n--- Config: batch={config['batch_size']}, seq_len={config['seq_length']} ---"
        )

        # 1. Original (pattern caching disabled, no optimizations)
        print("\n1. Original Ring Attention (no optimizations):")
        try:
            model_original = RingDilatedAttentionV2Collective(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=torch.float16,  # Force FP16 to show Pascal impact
                use_pattern_cache=False,
                enable_memory_pool=False,
            )
            perf_original = benchmark_model(model_original, **config)
            print(
                f"   Time: {perf_original['mean_time_ms']:.2f} ± {perf_original['std_time_ms']:.2f} ms"
            )
            print(f"   Memory: {perf_original['peak_memory_mb']:.1f} MB")
            print(f"   Throughput: {perf_original['throughput']:.0f} tokens/s")
        except Exception as e:
            print(f"   Error: {e}")
            perf_original = None

        # 2. With pattern caching only
        print("\n2. With Pattern Caching:")
        try:
            model_cached = RingDilatedAttentionV2Collective(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=torch.float16,
                use_pattern_cache=True,
                enable_memory_pool=False,
            )
            perf_cached = benchmark_model(model_cached, **config)
            print(
                f"   Time: {perf_cached['mean_time_ms']:.2f} ± {perf_cached['std_time_ms']:.2f} ms"
            )
            print(f"   Memory: {perf_cached['peak_memory_mb']:.1f} MB")
            print(f"   Throughput: {perf_cached['throughput']:.0f} tokens/s")

            if perf_original:
                speedup = perf_original["mean_time_ms"] / perf_cached["mean_time_ms"]
                print(f"   Speedup vs original: {speedup:.2f}x")
        except Exception as e:
            print(f"   Error: {e}")
            perf_cached = None

        # 3. Current optimized (auto dtype, pattern cache, memory pool)
        print("\n3. Current Optimized (auto dtype, caching, memory pool):")
        try:
            model_current = RingDilatedAttentionV2Collective(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                # dtype auto-selected based on GPU
                use_pattern_cache=True,
                enable_memory_pool=True,
                memory_pool_threshold_mb=16.0,
            )
            perf_current = benchmark_model(model_current, **config)
            print(f"   Using dtype: {model_current.dtype}")
            print(
                f"   Time: {perf_current['mean_time_ms']:.2f} ± {perf_current['std_time_ms']:.2f} ms"
            )
            print(f"   Memory: {perf_current['peak_memory_mb']:.1f} MB")
            print(f"   Throughput: {perf_current['throughput']:.0f} tokens/s")

            if perf_original:
                speedup = perf_original["mean_time_ms"] / perf_current["mean_time_ms"]
                print(f"   Speedup vs original: {speedup:.2f}x")
        except Exception as e:
            print(f"   Error: {e}")
            perf_current = None

        # 4. With Flash Attention backend
        print("\n4. With Flash/xformers Backend:")
        try:
            model_flash = RingDilatedAttentionV2Flash(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                # dtype auto-selected
                use_pattern_cache=True,
                enable_memory_pool=True,
                memory_pool_threshold_mb=16.0,
                use_flash_attention=True,
            )
            perf_flash = benchmark_model(model_flash, **config)
            print(f"   Using backend: {model_flash.flash_backend}")
            print(f"   Using dtype: {model_flash.dtype}")
            print(
                f"   Time: {perf_flash['mean_time_ms']:.2f} ± {perf_flash['std_time_ms']:.2f} ms"
            )
            print(f"   Memory: {perf_flash['peak_memory_mb']:.1f} MB")
            print(f"   Throughput: {perf_flash['throughput']:.0f} tokens/s")

            if perf_original:
                speedup = perf_original["mean_time_ms"] / perf_flash["mean_time_ms"]
                print(f"   Speedup vs original: {speedup:.2f}x")
            if perf_current:
                speedup = perf_current["mean_time_ms"] / perf_flash["mean_time_ms"]
                print(f"   Speedup vs current: {speedup:.2f}x")
        except Exception as e:
            print(f"   Error: {e}")
            perf_flash = None

        # Store results
        key = f"batch{config['batch_size']}_seq{config['seq_length']}"
        results[key] = {
            "original": perf_original,
            "cached": perf_cached,
            "current": perf_current,
            "flash": perf_flash,
        }

    return results


def distributed_worker(
    rank: int,
    world_size: int,
    model_type: str,
    config: dict,
    results_dict: dict,
):
    """Worker process for distributed testing."""
    setup_distributed(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    # Create model based on type
    if model_type == "original":
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            dtype=torch.float16,
            ring_size=world_size,
            use_pattern_cache=False,
            enable_memory_pool=False,
        )
    elif model_type == "current":
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            ring_size=world_size,
            use_pattern_cache=True,
            enable_memory_pool=True,
            memory_pool_threshold_mb=16.0,
        )
    elif model_type == "flash":
        model = RingDilatedAttentionV2Flash(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            ring_size=world_size,
            use_pattern_cache=True,
            enable_memory_pool=True,
            memory_pool_threshold_mb=16.0,
            use_flash_attention=True,
        )

    # Synchronize
    dist.barrier()

    # Benchmark
    perf = benchmark_model(model, **config, num_runs=5)

    # Gather results
    all_perfs = [None] * world_size
    dist.all_gather_object(all_perfs, perf)

    if rank == 0:
        # Average across ranks
        avg_time = np.mean([p["mean_time_ms"] for p in all_perfs])
        max_memory = np.max([p["peak_memory_mb"] for p in all_perfs])

        results_dict[model_type] = {
            "mean_time_ms": avg_time,
            "peak_memory_mb": max_memory,
            "all_times": [p["mean_time_ms"] for p in all_perfs],
        }

    cleanup_distributed()


def test_multi_gpu():
    """Test multi-GPU performance."""
    print("\n" + "=" * 80)
    print("MULTI-GPU PERFORMANCE COMPARISON")
    print("=" * 80)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs for distributed testing")
        return None

    print(f"\nUsing {world_size} GPUs")
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")

    # Test configuration
    config = {
        "batch_size": 1,
        "seq_length": 16384,
        "num_heads": 8,
        "head_dim": 64,
    }

    print(f"\nConfig: batch={config['batch_size']}, seq_len={config['seq_length']}")

    manager = mp.Manager()
    results = {}

    for model_type in ["original", "current", "flash"]:
        print(f"\n--- Testing {model_type} ---")

        model_results = manager.dict()
        processes = []

        for rank in range(world_size):
            p = mp.Process(
                target=distributed_worker,
                args=(rank, world_size, model_type, config, model_results),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if model_type in model_results:
            perf = model_results[model_type]
            results[model_type] = perf
            print(f"Average time: {perf['mean_time_ms']:.2f} ms")
            print(f"Peak memory: {perf['peak_memory_mb']:.1f} MB")
            print(f"Per-GPU times: {[f'{t:.1f}' for t in perf['all_times']]}")

    # Calculate speedups
    if "original" in results:
        print("\n--- Speedups vs Original ---")
        for model_type in ["current", "flash"]:
            if model_type in results:
                speedup = (
                    results["original"]["mean_time_ms"]
                    / results[model_type]["mean_time_ms"]
                )
                print(f"{model_type}: {speedup:.2f}x")

    return results


def print_summary(single_gpu_results: dict, multi_gpu_results: dict):
    """Print comprehensive summary."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 80)

    if single_gpu_results:
        print("\n### Single GPU Performance Gains")
        print("-" * 60)

        total_speedups = []
        for config_key, results in single_gpu_results.items():
            if results["original"] and results["flash"]:
                speedup = (
                    results["original"]["mean_time_ms"]
                    / results["flash"]["mean_time_ms"]
                )
                total_speedups.append(speedup)
                print(f"{config_key}: {speedup:.2f}x speedup")

        if total_speedups:
            print(f"\nAverage speedup: {np.mean(total_speedups):.2f}x")
            print(f"Max speedup: {np.max(total_speedups):.2f}x")

    if multi_gpu_results:
        print("\n### Multi-GPU Performance Gains")
        print("-" * 60)

        if "original" in multi_gpu_results and "flash" in multi_gpu_results:
            speedup = (
                multi_gpu_results["original"]["mean_time_ms"]
                / multi_gpu_results["flash"]["mean_time_ms"]
            )
            print(f"Multi-GPU speedup: {speedup:.2f}x")

    print("\n### Key Optimizations Applied:")
    print("1. ✅ Pattern caching enabled by default")
    print("2. ✅ Memory pool with 16MB threshold")
    print("3. ✅ Smart dtype selection (FP32 for Pascal)")
    print("4. ✅ Flash Attention/xformers backend")
    print("5. ✅ GPU architecture-aware optimization")


def main():
    """Run comprehensive benchmarks."""
    print("Ring Dilated Attention - Optimization Benchmark")
    print("Comparing original vs fully optimized implementation")

    # Test single GPU
    single_gpu_results = test_single_gpu()

    # Test multi GPU
    multi_gpu_results = None
    if torch.cuda.device_count() >= 2:
        multi_gpu_results = test_multi_gpu()

    # Print summary
    print_summary(single_gpu_results, multi_gpu_results)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        mp.set_start_method("spawn", force=True)

    main()
