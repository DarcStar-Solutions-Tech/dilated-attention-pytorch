# \!/usr/bin/env python3
"""
Quick multi-GPU scaling benchmark for hybrid ring dilated attention.
Tests O(n/p) scaling with progressively longer sequences.
"""

import torch
import torch.distributed as dist
import time
import os
from typing import Dict

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def setup_distributed():
    """Initialize distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    return rank, world_size


def benchmark_sequence(
    seq_len: int,
    world_size: int,
    rank: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
) -> Dict:
    """Benchmark a single sequence length."""

    # Ensure divisibility
    if seq_len % world_size != 0:
        seq_len = (seq_len // world_size) * world_size

    # Adaptive segments
    if seq_len <= 16384:
        segment_lengths = [2048, 4096]
        dilation_rates = [1, 2]
    elif seq_len <= 65536:
        segment_lengths = [2048, 4096, 8192]
        dilation_rates = [1, 2, 4]
    else:
        segment_lengths = [4096, 8192, 16384]
        dilation_rates = [1, 2, 4]

    device = torch.device(f"cuda:{rank}")

    # Create model
    model = RingDilatedAttentionHybrid(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        ring_size=world_size,
        use_flash_attention=True,
    )

    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Clear memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Single warmup
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)

    if world_size > 1:
        dist.barrier()

    # Time 3 iterations
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()
    total_time = time.time() - start_time
    avg_time = total_time / 3

    # Get memory
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    memory_per_token = peak_memory_mb * 1024 / (seq_len * batch_size)

    return {
        "seq_len": seq_len,
        "seq_per_gpu": seq_len // world_size,
        "time_ms": avg_time * 1000,
        "memory_mb": peak_memory_mb,
        "memory_per_token_kb": memory_per_token,
    }


def main():
    """Run scaling benchmark."""
    rank, world_size = setup_distributed()

    # Test sequences - scale with GPU count
    if world_size == 1:
        test_sequences = [4096, 8192, 16384, 32768]
    elif world_size == 2:
        test_sequences = [8192, 16384, 32768, 65536]  # Stop at 65K for faster results
    elif world_size == 4:
        test_sequences = [16384, 32768, 65536, 131072, 262144]
    else:
        test_sequences = [32768, 65536, 131072, 262144, 524288]

    if rank == 0:
        print(f"=== Hybrid Ring Attention Scaling - {world_size} GPUs ===")
        print("Testing O(n/p) memory scaling")
        print("")

    results = []

    for seq_len in test_sequences:
        if rank == 0:
            print(
                f"Testing {seq_len:,} tokens ({seq_len // world_size:,} per GPU)...",
                end="",
                flush=True,
            )

        if world_size > 1:
            dist.barrier()

        try:
            result = benchmark_sequence(seq_len, world_size, rank)
            results.append(result)

            if rank == 0:
                print(
                    f" {result['time_ms']:.0f}ms, {result['memory_mb']:.0f}MB, {result['memory_per_token_kb']:.1f}KB/tok"
                )

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(" OOM\!")
            break
        except Exception as e:
            if rank == 0:
                print(f" Error: {e}")
            break

    # Analysis (rank 0 only)
    if rank == 0 and len(results) >= 2:
        print(f"\n{'=' * 60}")
        print("Memory Scaling Analysis:")
        print(
            f"{'Seq Total':>12} {'Per GPU':>10} {'Mem/GPU':>10} {'KB/Token':>10} {'Scaling':>10}"
        )
        print("-" * 60)

        base_mem = results[0]["memory_per_token_kb"]
        for r in results:
            scaling = r["memory_per_token_kb"] / base_mem
            print(
                f"{r['seq_len']:>12,} {r['seq_per_gpu']:>10,} {r['memory_mb']:>9.0f}M {r['memory_per_token_kb']:>10.1f} {scaling:>10.2f}x"
            )

        # Check O(n/p) scaling
        first = results[0]
        last = results[-1]
        seq_ratio = last["seq_len"] / first["seq_len"]
        mem_ratio = last["memory_per_token_kb"] / first["memory_per_token_kb"]

        print(f"\nSequence increased: {seq_ratio:.1f}x")
        print(f"Memory/token ratio: {mem_ratio:.2f}x")

        if mem_ratio < 1.5:
            print("✅ Excellent O(n/p) scaling\!")
        elif mem_ratio < 2.0:
            print("✓ Good O(n/p) scaling")
        else:
            print("⚠️  Suboptimal scaling")

        # Max sequence capability
        max_seq = results[-1]["seq_len"]
        print(f"\nMax tested sequence: {max_seq:,} tokens total")
        print(f"Max per GPU: {max_seq // world_size:,} tokens")
        print(
            f"Theoretical max (8GB GPU): ~{int(8192 / results[-1]['memory_per_token_kb'] * 1024):,} tokens total"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
