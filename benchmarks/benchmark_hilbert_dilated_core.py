#!/usr/bin/env python3
"""Benchmark Hilbert Ring Attention with DilatedAttention core."""

import os
import torch
import torch.distributed as dist
import time
import json
from datetime import datetime
import numpy as np
import argparse


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


def benchmark_configuration(
    seq_len: int,
    segment_lengths: list,
    dilation_rates: list,
    num_heads: int = 8,
    hidden_dim: int = 512,
    batch_size: int = 1,
    warmup: int = 3,
    iterations: int = 10,
    test_backends: bool = True,
):
    """Benchmark different configurations."""
    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32

    # Import after setting device
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    head_dim = hidden_dim // num_heads
    results = {}

    if rank == 0:
        print(
            f"\nTesting {seq_len:,} tokens with segments={segment_lengths}, dilation={dilation_rates}"
        )
        print(f"  Batch size: {batch_size}, Heads: {num_heads}, Hidden: {hidden_dim}")
        print(f"  World size: {world_size} GPU(s)")

    # Test configurations
    configs = [
        ("Standard (no Hilbert)", False, False, False),
        ("Hilbert + DilatedAttention", True, False, False),
        ("Hilbert + DilatedAttention + MemPool", True, True, False),
    ]

    if test_backends:
        configs.append(("Hilbert + DilatedAttention + xFormers", True, True, True))

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for config_name, use_hilbert, use_mempool, use_xformers in configs:
        try:
            if rank == 0:
                print(f"\n  {config_name}:")

            # Create model
            model = RingDilatedAttentionHybridHilbert(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=dtype,
                use_hilbert=use_hilbert,
                hilbert_chunk_size=min(4096, seq_len // world_size),
                enable_memory_pool=use_mempool,
                use_xformers=use_xformers,
                enable_profiling=False,
            )

            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = model(q, k, v, is_causal=False)

            if dist.is_initialized():
                dist.barrier()
            torch.cuda.synchronize()

            # Benchmark forward pass
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            forward_time = (time.perf_counter() - start) / iterations

            # Benchmark with causal mask
            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(q, k, v, is_causal=True)
            torch.cuda.synchronize()

            causal_time = (time.perf_counter() - start) / iterations

            # Memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3

            # Collect results
            if dist.is_initialized():
                times = torch.tensor([forward_time, causal_time], device=device)
                dist.all_reduce(times, op=dist.ReduceOp.AVG)
                forward_time, causal_time = times.tolist()

            if rank == 0:
                print(
                    f"    Forward: {forward_time * 1000:.2f} ms ({seq_len / forward_time:,.0f} tokens/sec)"
                )
                print(
                    f"    Causal:  {causal_time * 1000:.2f} ms ({seq_len / causal_time:,.0f} tokens/sec)"
                )
                print(f"    Memory:  {allocated:.2f} GB")

                # Check features
                if hasattr(model, "dilated_attention"):
                    da = model.dilated_attention
                    if hasattr(da, "_memory_pool") and da._memory_pool:
                        print("    ✓ Memory pool active")
                    if hasattr(da, "_pattern_cache") and da._pattern_cache:
                        print(f"    ✓ Pattern cache: {len(da._pattern_cache)} entries")

            results[config_name] = {
                "forward_ms": forward_time * 1000,
                "causal_ms": causal_time * 1000,
                "tokens_per_sec": seq_len / forward_time,
                "memory_gb": allocated,
            }

        except Exception as e:
            if rank == 0:
                print(f"    ERROR: {str(e)[:60]}...")
            results[config_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--test-backends", action="store_true", help="Test different backends"
    )
    args = parser.parse_args()

    rank, _ = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 80)
        print("HILBERT RING ATTENTION WITH DILATED ATTENTION CORE BENCHMARK")
        print("=" * 80)
        print(f"Running on {world_size} GPU(s)")
        print("Testing optimizations: Memory Pool, Pattern Cache, xFormers\n")

    # Test configurations
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates, description)
        (8192, [2048], [1], "8K, single segment"),
        (8192, [2048, 4096], [1, 2], "8K, multi-segment"),
        (16384, [4096], [1], "16K, no dilation"),
        (16384, [4096], [2], "16K, dilation=2"),
    ]

    if world_size == 1:
        # Single GPU can handle larger sequences
        test_configs.extend(
            [
                (32768, [8192], [1], "32K, no dilation"),
                (32768, [4096, 8192], [2, 4], "32K, multi-dilation"),
            ]
        )

        if args.max_seq_len >= 65536:
            test_configs.append((65536, [8192], [2], "64K, dilation=2"))

    all_results = []

    for seq_len, seg_lens, dil_rates, desc in test_configs:
        if seq_len > args.max_seq_len:
            continue
        if seq_len % world_size != 0:
            continue

        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"{desc}")
            print(f"{'=' * 60}")

        results = benchmark_configuration(
            seq_len=seq_len,
            segment_lengths=seg_lens,
            dilation_rates=dil_rates,
            warmup=args.warmup,
            iterations=args.iterations,
            test_backends=args.test_backends,
        )

        if rank == 0 and results:
            all_results.append(
                {
                    "config": desc,
                    "seq_len": seq_len,
                    "segment_lengths": seg_lens,
                    "dilation_rates": dil_rates,
                    "world_size": world_size,
                    "results": results,
                }
            )

    # Summary
    if rank == 0 and all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Performance comparison
        print("\nPerformance Comparison (tokens/sec):")
        print("-" * 60)
        print(
            f"{'Configuration':<30} | {'Standard':<12} | {'Hilbert+DA':<12} | {'Speedup':<8}"
        )
        print("-" * 60)

        for result in all_results:
            config = result["config"]
            res = result["results"]

            standard_tps = res.get("Standard (no Hilbert)", {}).get("tokens_per_sec", 0)
            hilbert_tps = res.get("Hilbert + DilatedAttention", {}).get(
                "tokens_per_sec", 0
            )

            if standard_tps > 0 and hilbert_tps > 0:
                speedup = hilbert_tps / standard_tps
                print(
                    f"{config:<30} | {standard_tps:>12,.0f} | {hilbert_tps:>12,.0f} | {speedup:>7.2f}x"
                )

        # Feature utilization
        print("\nOptimization Features:")
        for opt in ["MemPool", "xFormers"]:
            configs_with_opt = [
                r
                for r in all_results
                if f"Hilbert + DilatedAttention + {opt}" in r["results"]
            ]
            if configs_with_opt:
                avg_speedup = np.mean(
                    [
                        r["results"][f"Hilbert + DilatedAttention + {opt}"][
                            "tokens_per_sec"
                        ]
                        / r["results"]["Hilbert + DilatedAttention"]["tokens_per_sec"]
                        for r in configs_with_opt
                        if "error"
                        not in r["results"][f"Hilbert + DilatedAttention + {opt}"]
                    ]
                )
                print(f"  {opt}: {avg_speedup:.2f}x speedup over base Hilbert+DA")

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"hilbert_dilated_core_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "results": all_results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
