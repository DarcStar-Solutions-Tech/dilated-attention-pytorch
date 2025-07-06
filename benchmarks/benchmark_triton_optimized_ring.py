#!/usr/bin/env python3
"""Benchmark the optimized Triton Ring Dilated Attention implementations."""

import torch
import torch.distributed as dist
import time
import os
import gc
import json
from datetime import datetime
import numpy as np


def init_distributed():
    """Initialize distributed if not already done."""
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank
    return 0, 0


def benchmark_implementation(
    model_class,
    model_kwargs,
    seq_len,
    batch_size=1,
    num_heads=8,
    head_dim=64,
    warmup=3,
    iterations=10,
):
    """Benchmark a specific implementation."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32  # fp32 for Pascal

    # Create model
    model = model_class(
        device=device, dtype=dtype, ring_size=world_size, **model_kwargs
    ).eval()

    # Create inputs
    _ = num_heads * head_dim
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    if dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    memory_usage = []

    for _ in range(iterations):
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.no_grad():
            output = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        times.append(time.perf_counter() - start)
        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)

    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_memory = np.mean(memory_usage)
    throughput = (batch_size * seq_len * world_size) / avg_time

    # Cleanup
    del model, q, k, v, output
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": std_time * 1000,
        "throughput_tokens_per_sec": throughput,
        "memory_gb": avg_memory,
    }


def main():
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 80)
        print("TRITON OPTIMIZED RING DILATED ATTENTION BENCHMARK")
        print("=" * 80)
        print(f"World size: {world_size} GPU(s)")
        print("Using fp32 for Pascal architecture")
        print()

    # Import implementations
    try:
        # Baseline
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        # New optimized implementations
        from dilated_attention_pytorch.ring_dilated_attention_triton_optimized import (
            RingDilatedAttentionTritonOptimized,
        )

        # Direct Triton kernel (if available)
        try:
            from dilated_attention_pytorch.ring_dilated_attention_triton_kernel import (
                RingDilatedAttentionTritonKernel,
            )

            has_triton_kernel = True
        except ImportError:
            has_triton_kernel = False
            if rank == 0:
                print("Note: Triton kernel implementation not available")

    except ImportError as e:
        if rank == 0:
            print(f"Import error: {e}")
        return

    # Test configurations
    test_configs = [
        # (seq_len, segments, dilations, description)
        (8192, [2048], [1], "8K, no dilation"),
        (8192, [2048], [4], "8K, dilation=4"),
        (16384, [4096], [1], "16K, no dilation"),
        (16384, [4096], [4], "16K, dilation=4"),
        (32768, [8192], [1], "32K, no dilation"),
        (32768, [8192], [4], "32K, dilation=4"),
        (32768, [4096, 8192], [2, 4], "32K, multi-segment"),
    ]

    # Implementations to test
    implementations = [
        (
            "Baseline (no Hilbert)",
            RingDilatedAttentionHybridHilbert,
            {"use_hilbert": False},
        ),
        ("Python Hilbert", RingDilatedAttentionHybridHilbert, {"use_hilbert": True}),
        (
            "Triton Optimized",
            RingDilatedAttentionTritonOptimized,
            {
                "use_triton_hilbert": True,
                "apply_hilbert_to_dilated": True,
            },
        ),
        (
            "Triton Opt (no Hilbert)",
            RingDilatedAttentionTritonOptimized,
            {
                "use_triton_hilbert": False,
            },
        ),
    ]

    if has_triton_kernel:
        implementations.append(
            ("Triton Kernel Direct", RingDilatedAttentionTritonKernel, {})
        )

    results = []

    for seq_len, segments, dilations, desc in test_configs:
        # Check validity
        if seq_len % world_size != 0:
            continue

        if seq_len % max(segments) != 0:
            continue

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"{desc}")
            print(f"Segments: {segments}, Dilation: {dilations}")
            print(f"{'=' * 60}")

        config_results = {
            "description": desc,
            "seq_len": seq_len,
            "segments": segments,
            "dilations": dilations,
            "world_size": world_size,
            "implementations": {},
        }

        for impl_name, impl_class, impl_kwargs in implementations:
            if rank == 0:
                print(f"\n{impl_name}:")

            try:
                result = benchmark_implementation(
                    impl_class,
                    {
                        "segment_lengths": segments,
                        "dilation_rates": dilations,
                        **impl_kwargs,
                    },
                    seq_len=seq_len,
                )

                if rank == 0:
                    print(
                        f"  Time: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms"
                    )
                    print(
                        f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec"
                    )
                    print(f"  Memory: {result['memory_gb']:.2f} GB")

                config_results["implementations"][impl_name] = result

            except Exception as e:
                if rank == 0:
                    print(f"  ERROR: {str(e)[:60]}...")
                config_results["implementations"][impl_name] = {"error": str(e)}

        results.append(config_results)

    # Summary and analysis
    if rank == 0 and results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Find best performers
        print("\nBest Implementation by Configuration:")
        print("-" * 60)

        for result in results:
            desc = result["description"]
            impls = result["implementations"]

            # Find best throughput
            best_impl = None
            best_throughput = 0

            for impl_name, metrics in impls.items():
                if "throughput_tokens_per_sec" in metrics:
                    if metrics["throughput_tokens_per_sec"] > best_throughput:
                        best_throughput = metrics["throughput_tokens_per_sec"]
                        best_impl = impl_name

            if best_impl:
                print(f"{desc}: {best_impl} ({best_throughput:,.0f} tok/s)")

        # Speedup analysis
        print("\n" + "-" * 60)
        print("Speedup Analysis (vs Baseline):")
        print("-" * 60)

        for result in results:
            desc = result["description"]
            impls = result["implementations"]

            baseline = impls.get("Baseline (no Hilbert)", {})
            baseline_throughput = baseline.get("throughput_tokens_per_sec", 0)

            if baseline_throughput > 0:
                print(f"\n{desc}:")
                for impl_name, metrics in impls.items():
                    if "throughput_tokens_per_sec" in metrics:
                        speedup = (
                            metrics["throughput_tokens_per_sec"] / baseline_throughput
                        )
                        print(f"  {impl_name}: {speedup:.2f}x")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"triton_optimized_benchmark_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "results": results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to: {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
