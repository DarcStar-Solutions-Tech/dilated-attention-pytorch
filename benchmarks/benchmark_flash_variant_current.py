#!/usr/bin/env python3
"""
Benchmark RingDilatedAttentionV2Flash variant on current GPU setup.
Tests both single and multi-GPU performance with various configurations.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import numpy as np
from typing import Dict, List, Optional
import os
import warnings

from dilated_attention_pytorch import RingDilatedAttentionV2Flash


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
    """Test single GPU performance with various configurations."""
    print("\n" + "=" * 80)
    print("SINGLE GPU PERFORMANCE - RingDilatedAttentionV2Flash")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("No GPU available")
        return None

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print(f"\nGPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Memory: {props.total_memory / 1024**3:.1f} GB")

    # Test configurations - various sizes
    configs = [
        # Small configs
        {"batch_size": 1, "seq_length": 2048, "num_heads": 8, "head_dim": 64},
        {"batch_size": 2, "seq_length": 2048, "num_heads": 8, "head_dim": 64},
        # Medium configs
        {"batch_size": 1, "seq_length": 4096, "num_heads": 8, "head_dim": 64},
        {"batch_size": 2, "seq_length": 4096, "num_heads": 8, "head_dim": 64},
        # Larger configs
        {"batch_size": 1, "seq_length": 8192, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_length": 16384, "num_heads": 8, "head_dim": 64},
        # Test with different head configurations
        {"batch_size": 1, "seq_length": 4096, "num_heads": 12, "head_dim": 64},
        {"batch_size": 1, "seq_length": 4096, "num_heads": 16, "head_dim": 64},
    ]

    # Different segment/dilation configurations
    segment_configs = [
        {"segments": [1024, 2048], "dilations": [1, 2]},
        {"segments": [2048, 4096], "dilations": [1, 2]},
        {"segments": [2048, 4096, 8192], "dilations": [1, 2, 4]},
    ]

    results = {}

    for seg_config in segment_configs:
        segment_lengths = seg_config["segments"]
        dilation_rates = seg_config["dilations"]

        print(f"\n{'=' * 60}")
        print(f"Segment lengths: {segment_lengths}")
        print(f"Dilation rates: {dilation_rates}")
        print("=" * 60)

        for config in configs:
            # Skip if sequence length not divisible by largest segment
            if config["seq_length"] % max(segment_lengths) != 0:
                continue

            config_key = (
                f"seg{'-'.join(map(str, segment_lengths))}_"
                f"batch{config['batch_size']}_"
                f"seq{config['seq_length']}_"
                f"heads{config['num_heads']}"
            )

            print(f"\nTesting: {config}")

            try:
                # Create model
                model = RingDilatedAttentionV2Flash(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    device=device,
                    ring_size=1,  # Single GPU
                    use_pattern_cache=True,
                    enable_memory_pool=True,
                    memory_pool_threshold_mb=16.0,
                    use_flash_attention=True,
                )

                print(f"Backend: {model.flash_backend}")
                print(f"Dtype: {model.dtype}")

                # Benchmark
                perf = benchmark_model(model, **config)
                results[config_key] = perf

                print(
                    f"Time: {perf['mean_time_ms']:.2f} ± {perf['std_time_ms']:.2f} ms"
                )
                print(f"Memory: {perf['peak_memory_mb']:.1f} MB")
                print(f"Throughput: {perf['throughput']:.0f} tokens/s")

            except Exception as e:
                print(f"Error: {e}")
                results[config_key] = None

    return results


def distributed_worker(
    rank: int,
    world_size: int,
    config: dict,
    segment_lengths: List[int],
    dilation_rates: List[int],
    results_dict: dict,
):
    """Worker process for distributed testing."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")

    # Create model
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
        std_time = np.std([p["mean_time_ms"] for p in all_perfs])
        max_memory = np.max([p["peak_memory_mb"] for p in all_perfs])
        avg_throughput = np.mean([p["throughput"] for p in all_perfs])

        results_dict["perf"] = {
            "mean_time_ms": avg_time,
            "std_time_ms": std_time,
            "peak_memory_mb": max_memory,
            "throughput": avg_throughput,
            "all_times": [p["mean_time_ms"] for p in all_perfs],
        }
        results_dict["backend"] = model.flash_backend
        results_dict["dtype"] = str(model.dtype)

    dist.destroy_process_group()


def test_multi_gpu():
    """Test multi-GPU performance."""
    print("\n" + "=" * 80)
    print("MULTI-GPU PERFORMANCE - RingDilatedAttentionV2Flash")
    print("=" * 80)

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Need at least 2 GPUs for distributed testing")
        return None

    print(f"\nUsing {world_size} GPUs:")
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # Test configurations
    configs = [
        {"batch_size": 1, "seq_length": 4096, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_length": 8192, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_length": 16384, "num_heads": 8, "head_dim": 64},
        {"batch_size": 2, "seq_length": 4096, "num_heads": 8, "head_dim": 64},
    ]

    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]

    results = {}

    for config in configs:
        # Skip if sequence length not divisible by largest segment
        if config["seq_length"] % max(segment_lengths) != 0:
            continue

        print(f"\n--- Config: {config} ---")

        manager = mp.Manager()
        model_results = manager.dict()
        processes = []

        for rank in range(world_size):
            p = mp.Process(
                target=distributed_worker,
                args=(
                    rank,
                    world_size,
                    config,
                    segment_lengths,
                    dilation_rates,
                    model_results,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if "perf" in model_results:
            perf = model_results["perf"]
            backend = model_results["backend"]
            dtype = model_results["dtype"]

            config_key = f"batch{config['batch_size']}_seq{config['seq_length']}_heads{config['num_heads']}"
            results[config_key] = perf

            print(f"Backend: {backend}")
            print(f"Dtype: {dtype}")
            print(
                f"Average time: {perf['mean_time_ms']:.2f} ± {perf['std_time_ms']:.2f} ms"
            )
            print(f"Peak memory: {perf['peak_memory_mb']:.1f} MB")
            print(f"Throughput: {perf['throughput']:.0f} tokens/s")
            print(f"Per-GPU times: {[f'{t:.1f}' for t in perf['all_times']]} ms")

    return results


def print_summary(
    single_gpu_results: Optional[dict], multi_gpu_results: Optional[dict]
):
    """Print performance summary."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY - RingDilatedAttentionV2Flash")
    print("=" * 80)

    if single_gpu_results:
        print("\n### Single GPU Results")
        print("-" * 60)

        # Group by sequence length
        by_seq_len = {}
        for key, perf in single_gpu_results.items():
            if perf:
                seq_len = int(key.split("_seq")[1].split("_")[0])
                if seq_len not in by_seq_len:
                    by_seq_len[seq_len] = []
                by_seq_len[seq_len].append((key, perf))

        for seq_len in sorted(by_seq_len.keys()):
            print(f"\nSequence Length: {seq_len}")
            for key, perf in by_seq_len[seq_len]:
                print(
                    f"  {key}: {perf['mean_time_ms']:.2f} ms, "
                    f"{perf['throughput']:.0f} tokens/s, "
                    f"{perf['peak_memory_mb']:.1f} MB"
                )

    if multi_gpu_results:
        print("\n### Multi-GPU Results (2 GPUs)")
        print("-" * 60)

        for key, perf in multi_gpu_results.items():
            print(f"\n{key}:")
            print(f"  Time: {perf['mean_time_ms']:.2f} ± {perf['std_time_ms']:.2f} ms")
            print(f"  Throughput: {perf['throughput']:.0f} tokens/s")
            print(f"  Memory: {perf['peak_memory_mb']:.1f} MB")

    print("\n### Key Features Enabled:")
    print("✅ Pattern caching (default)")
    print("✅ Memory pool (16MB threshold)")
    print("✅ Smart dtype selection (FP32 on Pascal)")
    print("✅ Flash Attention with xformers fallback")
    print("✅ GPU architecture-aware optimization")


def main():
    """Run comprehensive benchmarks."""
    print("RingDilatedAttentionV2Flash Performance Benchmark")
    print("Testing on current GPU setup with all optimizations enabled")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Test single GPU
    single_gpu_results = test_single_gpu()

    # Test multi GPU
    multi_gpu_results = None
    if torch.cuda.device_count() >= 2:
        mp.set_start_method("spawn", force=True)
        multi_gpu_results = test_multi_gpu()

    # Print summary
    print_summary(single_gpu_results, multi_gpu_results)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
