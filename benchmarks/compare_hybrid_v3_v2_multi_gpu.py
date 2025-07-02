#!/usr/bin/env python3
"""
Compare performance of Hybrid, V3, and V2 Collective implementations on multiple GPUs.
Run with: torchrun --nproc_per_node=2 benchmarks/compare_hybrid_v3_v2_multi_gpu.py
"""

import os
import gc
import time
import torch
import torch.distributed as dist
from datetime import datetime
from typing import Dict

# Import implementations
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def benchmark_implementation(
    model,
    model_name: str,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_warmup: int = 3,
    num_iterations: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Benchmark a single implementation."""
    device = device or torch.cuda.current_device()

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Scale inputs to prevent overflow
    scale = 0.1 / (seq_len**0.25)
    q = q * scale
    k = k * scale
    v = v * scale

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Memory before
    mem_before = torch.cuda.memory_allocated(device) / (1024**2)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    # Get final output for validation
    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    # Memory stats
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    mem_after = torch.cuda.memory_allocated(device) / (1024**2)

    # Check output validity
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    output_mean = output.float().mean().item()
    output_std = output.float().std().item()

    # Clean up
    del q, k, v, output
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "seq_len": seq_len,
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "mem_before": mem_before,
        "mem_after": mem_after,
        "peak_memory": peak_memory,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "output_mean": output_mean,
        "output_std": output_std,
    }


def compare_implementations():
    """Compare all three implementations on multiple GPUs."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/compare_hybrid_v3_v2_multi_gpu.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Ring Attention Implementation Comparison")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print()

    # Test configurations
    test_configs = [
        # (seq_len, segment_len, dilation_rate, name)
        (1024, 512, 1, "1K seq, dilation=1"),
        (2048, 1024, 1, "2K seq, dilation=1"),
        (4096, 2048, 1, "4K seq, dilation=1"),
        (2048, 512, 2, "2K seq, dilation=2"),
        (4096, 1024, 2, "4K seq, dilation=2"),
    ]

    all_results = []

    for seq_len, segment_len, dilation_rate, config_name in test_configs:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Testing: {config_name}")
            print(
                f"Sequence length: {seq_len:,}, Segment: {segment_len}, Dilation: {dilation_rate}"
            )
            print("-" * 60)

        dist.barrier()

        results_for_config = []

        # Test each implementation
        implementations = []

        # 1. V2 Collective (baseline)
        try:
            v2_model = RingDilatedAttentionV2Collective(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
            )
            implementations.append(("V2 Collective", v2_model))
        except Exception as e:
            if rank == 0:
                print(f"Failed to create V2 Collective: {e}")

        # 2. V3 (true ring, but may have issues)
        try:
            v3_model = RingDilatedAttentionV3(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
                use_bucketed=False,  # Disable bucketing due to issues
            )
            implementations.append(("V3", v3_model))
        except Exception as e:
            if rank == 0:
                print(f"Failed to create V3: {e}")

        # 3. Hybrid (new implementation)
        try:
            hybrid_model = RingDilatedAttentionHybrid(
                segment_lengths=[segment_len],
                dilation_rates=[dilation_rate],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
                enable_memory_pool=True,
                use_pattern_cache=True,
                use_flash_attention=True,
            )
            implementations.append(("Hybrid", hybrid_model))
        except Exception as e:
            if rank == 0:
                print(f"Failed to create Hybrid: {e}")

        # Benchmark each implementation
        for model_name, model in implementations:
            try:
                if rank == 0:
                    print(f"\nBenchmarking {model_name}...", end="", flush=True)

                result = benchmark_implementation(
                    model,
                    model_name,
                    seq_len=seq_len,
                    device=device,
                    dtype=torch.float16,
                )

                # Gather results from all ranks
                all_rank_results = [None] * world_size
                dist.all_gather_object(all_rank_results, result)

                if rank == 0:
                    # Aggregate results
                    avg_time = sum(r["avg_time"] for r in all_rank_results) / len(
                        all_rank_results
                    )
                    max_peak_mem = max(r["peak_memory"] for r in all_rank_results)
                    avg_peak_mem = sum(
                        r["peak_memory"] for r in all_rank_results
                    ) / len(all_rank_results)
                    any_nan = any(r["has_nan"] for r in all_rank_results)
                    any_inf = any(r["has_inf"] for r in all_rank_results)

                    print(" Done!")
                    print(f"  Time: {avg_time * 1000:.2f}ms (avg across GPUs)")
                    print(
                        f"  Memory: {avg_peak_mem:.1f}MB avg, {max_peak_mem:.1f}MB max"
                    )
                    print(f"  Throughput: {seq_len / avg_time:.0f} tokens/sec")

                    if any_nan or any_inf:
                        print(f"  ⚠️  WARNING: NaN={any_nan}, Inf={any_inf}")

                    # Store for comparison
                    results_for_config.append(
                        {
                            "model": model_name,
                            "seq_len": seq_len,
                            "avg_time_ms": avg_time * 1000,
                            "avg_peak_mem_mb": avg_peak_mem,
                            "max_peak_mem_mb": max_peak_mem,
                            "throughput": seq_len / avg_time,
                            "has_issues": any_nan or any_inf,
                        }
                    )

            except Exception as e:
                if rank == 0:
                    print(" Failed!")
                    print(f"  Error: {e}")
                    results_for_config.append(
                        {
                            "model": model_name,
                            "seq_len": seq_len,
                            "error": str(e),
                        }
                    )

            # Clean up
            del model
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()

        if rank == 0 and results_for_config:
            all_results.extend(results_for_config)

            # Compare implementations for this config
            print("\nComparison for this configuration:")
            print("-" * 60)

            valid_results = [r for r in results_for_config if "error" not in r]
            if len(valid_results) >= 2:
                # Find baseline (V2)
                v2_result = next(
                    (r for r in valid_results if r["model"] == "V2 Collective"), None
                )
                if v2_result:
                    print(
                        f"{'Model':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'vs V2 Time':<12} {'vs V2 Mem':<12}"
                    )
                    print("-" * 60)

                    for r in valid_results:
                        time_ratio = r["avg_time_ms"] / v2_result["avg_time_ms"]
                        mem_ratio = r["avg_peak_mem_mb"] / v2_result["avg_peak_mem_mb"]

                        time_str = f"{r['avg_time_ms']:.2f}"
                        mem_str = f"{r['avg_peak_mem_mb']:.1f}"
                        time_cmp = f"{time_ratio:.2f}x"
                        mem_cmp = f"{mem_ratio:.2f}x"

                        if r["has_issues"]:
                            time_str += " ⚠️"

                        print(
                            f"{r['model']:<15} {time_str:<12} {mem_str:<12} {time_cmp:<12} {mem_cmp:<12}"
                        )

    # Final summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Group results by model
        models = {}
        for r in all_results:
            if "error" not in r:
                model = r["model"]
                if model not in models:
                    models[model] = []
                models[model].append(r)

        # Calculate averages
        print(
            f"\n{'Model':<15} {'Avg Time (ms)':<15} {'Avg Memory (MB)':<15} {'Configs Passed':<15}"
        )
        print("-" * 60)

        for model, results in models.items():
            if results:
                avg_time = sum(r["avg_time_ms"] for r in results) / len(results)
                avg_mem = sum(r["avg_peak_mem_mb"] for r in results) / len(results)
                num_passed = len([r for r in results if not r.get("has_issues", False)])
                total = len(results)

                print(
                    f"{model:<15} {avg_time:<15.2f} {avg_mem:<15.1f} {num_passed}/{total:<15}"
                )

        # Memory efficiency comparison
        print("\nMemory Efficiency (vs V2 Collective):")
        print("-" * 40)

        if "V2 Collective" in models and len(models) > 1:
            v2_avg_mem = sum(
                r["avg_peak_mem_mb"] for r in models["V2 Collective"]
            ) / len(models["V2 Collective"])

            for model, results in models.items():
                if model != "V2 Collective" and results:
                    model_avg_mem = sum(r["avg_peak_mem_mb"] for r in results) / len(
                        results
                    )
                    reduction = (1 - model_avg_mem / v2_avg_mem) * 100
                    print(f"{model}: {reduction:+.1f}% memory usage")

        print("\nNotes:")
        print("- V2 Collective: Uses all_gather (O(n) memory per GPU)")
        print("- V3: True ring attention (O(n/p) memory per GPU)")
        print("- Hybrid: True ring + V2 optimizations (O(n/p) memory + features)")
        print(f"- All tests run on {world_size} GPUs")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    compare_implementations()
