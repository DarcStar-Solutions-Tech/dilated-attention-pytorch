#!/usr/bin/env python3
"""Benchmark Hilbert ring attention across multiple GPUs."""

import os
import torch
import torch.distributed as dist
import time
import json
from datetime import datetime
import argparse


def init_distributed():
    """Initialize distributed environment."""
    if dist.is_initialized():
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        return rank, local_rank
    else:
        return 0, 0


def benchmark_configuration(
    seq_len: int,
    hidden_dim: int = 512,
    num_heads: int = 8,
    batch_size: int = 1,
    warmup: int = 3,
    iterations: int = 10,
):
    """Benchmark standard vs Hilbert ring attention."""

    rank, local_rank = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float32  # Use fp32 to avoid overflow

    # Import after setting device
    from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
        RingDilatedAttentionHybridHilbert,
    )

    head_dim = hidden_dim // num_heads

    if rank == 0:
        print(f"\nTesting {seq_len} tokens on {world_size} GPUs")
        print(f"  Per GPU: {seq_len // world_size} tokens")
        print(f"  Hidden dim: {hidden_dim}, Heads: {num_heads}")

    # Create models
    model_standard = RingDilatedAttentionHybridHilbert(
        segment_lengths=[4096],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=dtype,
        use_hilbert=False,  # Standard
    )

    model_hilbert = RingDilatedAttentionHybridHilbert(
        segment_lengths=[4096],
        dilation_rates=[1],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=dtype,
        use_hilbert=True,  # Hilbert
        hilbert_chunk_size=min(4096, seq_len // world_size),
    )

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    if rank == 0:
        print("  Warming up...")

    for _ in range(warmup):
        with torch.no_grad():
            _ = model_standard(q, k, v, is_causal=False)
            _ = model_hilbert(q, k, v, is_causal=False)

    if dist.is_initialized():
        dist.barrier()

    # Benchmark standard
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model_standard(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / iterations

    if dist.is_initialized():
        dist.barrier()

    # Benchmark Hilbert
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model_hilbert(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    hilbert_time = (time.perf_counter() - start) / iterations

    # Average times across ranks
    if dist.is_initialized():
        times = torch.tensor([standard_time, hilbert_time], device=device)
        dist.all_reduce(times, op=dist.ReduceOp.AVG)
        standard_time, hilbert_time = times.tolist()

    # Memory stats
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    if rank == 0:
        speedup = standard_time / hilbert_time
        print(f"  Standard: {standard_time * 1000:.1f} ms")
        print(f"  Hilbert:  {hilbert_time * 1000:.1f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  Memory:   {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        return {
            "seq_len": seq_len,
            "world_size": world_size,
            "standard_ms": standard_time * 1000,
            "hilbert_ms": hilbert_time * 1000,
            "speedup": speedup,
            "memory_gb": allocated,
        }

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    rank, _ = init_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("=" * 70)
        print("HILBERT RING ATTENTION MULTI-GPU BENCHMARK")
        print("=" * 70)
        print(f"Running on {world_size} GPU(s)")

    # Test configurations
    configs = []
    seq_len = 8192
    while seq_len <= args.max_seq_len:
        # Ensure divisible by world size
        if seq_len % world_size == 0:
            configs.append(seq_len)
        seq_len *= 2

    results = []

    for seq_len in configs:
        try:
            result = benchmark_configuration(
                seq_len=seq_len,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if result:
                results.append(result)
        except Exception as e:
            if rank == 0:
                print(f"  ERROR: {e}")
            break

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        print("\nSequence Length | Standard (ms) | Hilbert (ms) | Speedup")
        print("-" * 60)
        for r in results:
            print(
                f"{r['seq_len']:>15,} | {r['standard_ms']:>13.1f} | "
                f"{r['hilbert_ms']:>12.1f} | {r['speedup']:>7.2f}x"
            )

        # Save results
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"hilbert_multi_gpu_{world_size}gpu_{timestamp}.json"

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

        print(f"\nResults saved to {filename}")

    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
