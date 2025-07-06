#!/usr/bin/env python3
"""Benchmark impact of dilation rates on Hilbert performance.

This script tests how different dilation rates affect the performance
of Hilbert Ring Attention, starting from 16K sequence lengths.
"""

import os
import torch
import torch.distributed as dist
import time
import json
from datetime import datetime
import numpy as np
import argparse
from typing import Dict, List


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


def benchmark_dilation_config(
    seq_len: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    batch_size: int = 2,
    num_heads: int = 12,
    hidden_dim: int = 768,
    iterations: int = 10,
    warmup: int = 3,
) -> Dict:
    """Benchmark a specific dilation configuration."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")

    # Import after device setup
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    head_dim = hidden_dim // num_heads
    results = {}

    # Create model with Hilbert ordering
    try:
        model = RingDilatedAttentionHybridHilbert(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_hilbert=True,
            hilbert_chunk_size=min(4096, seq_len // world_size),
            enable_memory_pool=True,
            use_xformers=True,
            enable_profiling=False,
        ).eval()

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=model.dtype
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

        # Measure forward pass
        forward_times = []
        memory_usage = []

        for _ in range(iterations):
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()

            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)

        # Calculate metrics
        avg_forward_time = np.mean(forward_times)
        avg_memory_gb = np.mean(memory_usage)
        throughput = (batch_size * seq_len) / avg_forward_time

        # Aggregate across GPUs if distributed
        if dist.is_initialized():
            metrics = torch.tensor([avg_forward_time, throughput], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
            avg_forward_time, throughput = metrics.tolist()

        results = {
            "success": True,
            "forward_time_ms": avg_forward_time * 1000,
            "throughput_tokens_per_sec": throughput,
            "memory_gb": avg_memory_gb,
            "effective_batch_size": batch_size * world_size,
        }

        # Test with model without Hilbert for comparison
        model_no_hilbert = RingDilatedAttentionHybridHilbert(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=model.dtype,
            use_hilbert=False,  # Disable Hilbert
            enable_memory_pool=True,
            use_xformers=True,
        ).eval()

        # Benchmark without Hilbert
        no_hilbert_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model_no_hilbert(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            no_hilbert_times.append(time.perf_counter() - start)

        avg_no_hilbert_time = np.mean(no_hilbert_times)
        no_hilbert_throughput = (batch_size * seq_len) / avg_no_hilbert_time

        results["no_hilbert_forward_ms"] = avg_no_hilbert_time * 1000
        results["no_hilbert_throughput"] = no_hilbert_throughput
        results["hilbert_speedup"] = throughput / no_hilbert_throughput

    except torch.cuda.OutOfMemoryError:
        results = {"success": False, "error": "OOM"}
    except Exception as e:
        results = {"success": False, "error": str(e)}

    # Cleanup
    torch.cuda.empty_cache()

    return results


def test_dilation_scaling(
    seq_lengths: List[int],
    world_size: int,
    batch_size: int = 2,
) -> List[Dict]:
    """Test how dilation rates affect performance at different scales."""
    rank, _ = init_distributed()

    # Dilation configurations to test
    dilation_configs = [
        # (name, segment_lengths, dilation_rates)
        ("No dilation", [4096], [1]),
        ("Standard dilation", [2048, 4096, 8192], [1, 2, 4]),
        ("High dilation", [2048, 4096, 8192], [1, 4, 16]),
        ("Many segments", [1024, 2048, 4096, 8192], [1, 2, 4, 8]),
        ("Large segments", [8192, 16384], [1, 2]),
        ("Extreme dilation", [2048, 8192], [1, 8]),
    ]

    results = []

    for seq_len in seq_lengths:
        if seq_len % world_size != 0:
            continue

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Testing sequence length: {seq_len:,} tokens")
            print(f"World size: {world_size} GPUs")
            print(f"{'=' * 60}")

        for config_name, segments, dilations in dilation_configs:
            # Check if config is valid for this sequence length
            max_segment = max(segments)
            if seq_len % max_segment != 0:
                if rank == 0:
                    print(
                        f"\n{config_name}: Skipped (seq_len not divisible by {max_segment})"
                    )
                continue

            if rank == 0:
                print(f"\n{config_name}:")
                print(f"  Segments: {segments}")
                print(f"  Dilation: {dilations}")

            result = benchmark_dilation_config(
                seq_len=seq_len,
                segment_lengths=segments,
                dilation_rates=dilations,
                batch_size=batch_size,
            )

            if rank == 0:
                if result["success"]:
                    print(
                        f"  ✓ Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec"
                    )
                    print(f"    Memory: {result['memory_gb']:.2f} GB")
                    print(f"    Hilbert speedup: {result['hilbert_speedup']:.2f}x")
                else:
                    print(f"  ✗ Failed: {result['error']}")

            result.update(
                {
                    "seq_len": seq_len,
                    "config_name": config_name,
                    "segments": segments,
                    "dilations": dilations,
                    "world_size": world_size,
                }
            )
            results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-seq-len", type=int, default=16384, help="Minimum sequence length to test"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=262144,
        help="Maximum sequence length to test",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    rank, _ = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Sequence lengths to test (exponential scaling)
    seq_lengths = []
    seq_len = args.min_seq_len
    while seq_len <= args.max_seq_len:
        seq_lengths.append(seq_len)
        seq_len *= 2

    if rank == 0:
        print("=" * 80)
        print("DILATION RATES IMPACT ON HILBERT PERFORMANCE")
        print("=" * 80)
        print(f"Testing sequence lengths: {[f'{s:,}' for s in seq_lengths]}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")

    # Run benchmarks
    results = test_dilation_scaling(
        seq_lengths=seq_lengths,
        world_size=world_size,
        batch_size=args.batch_size,
    )

    # Save and analyze results
    if rank == 0 and results:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = (
            f"{args.output_dir}/dilation_hilbert_{world_size}gpu_{timestamp}.json"
        )

        # Save raw results
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

        print(f"\n{'=' * 80}")
        print("ANALYSIS")
        print("=" * 80)

        # Analyze by sequence length
        for seq_len in seq_lengths:
            seq_results = [
                r for r in results if r["seq_len"] == seq_len and r["success"]
            ]
            if not seq_results:
                continue

            print(f"\nSequence Length: {seq_len:,} tokens")
            print("-" * 60)

            # Sort by throughput
            seq_results.sort(key=lambda x: x["throughput_tokens_per_sec"], reverse=True)

            print(
                f"{'Config':<20} | {'Throughput':<15} | {'Memory':<10} | {'Hilbert Speedup':<15}"
            )
            print("-" * 60)

            for r in seq_results[:5]:  # Top 5
                print(
                    f"{r['config_name']:<20} | "
                    f"{r['throughput_tokens_per_sec']:>14,.0f} | "
                    f"{r['memory_gb']:>9.2f} | "
                    f"{r['hilbert_speedup']:>14.2f}x"
                )

        # Find best dilation config overall
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            best_throughput = max(
                successful_results, key=lambda x: x["throughput_tokens_per_sec"]
            )
            best_hilbert = max(successful_results, key=lambda x: x["hilbert_speedup"])

            print(f"\n{'=' * 60}")
            print("BEST CONFIGURATIONS")
            print("=" * 60)

            print("\nHighest Throughput:")
            print(f"  Config: {best_throughput['config_name']}")
            print(f"  Sequence: {best_throughput['seq_len']:,} tokens")
            print(
                f"  Throughput: {best_throughput['throughput_tokens_per_sec']:,.0f} tokens/sec"
            )

            print("\nBest Hilbert Speedup:")
            print(f"  Config: {best_hilbert['config_name']}")
            print(f"  Sequence: {best_hilbert['seq_len']:,} tokens")
            print(f"  Speedup: {best_hilbert['hilbert_speedup']:.2f}x")

        print(f"\nResults saved to: {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
