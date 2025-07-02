#!/usr/bin/env python3
"""
Multi-GPU benchmark for hybrid ring dilated attention with very long sequences.
Tests true O(n/p) scaling without using all_gather.
"""

import torch
import torch.distributed as dist
import time
import os
import gc
from typing import Dict, List
import json

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


def test_long_sequence(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict:
    """Test a specific sequence length on multi-GPU setup."""

    # Adaptive segment lengths based on sequence length
    if segment_lengths is None:
        if seq_len <= 16384:
            segment_lengths = [2048, 4096]
            dilation_rates = [1, 2]
        elif seq_len <= 65536:
            segment_lengths = [2048, 4096, 8192]
            dilation_rates = [1, 2, 4]
        elif seq_len <= 262144:
            segment_lengths = [2048, 4096, 8192, 16384]
            dilation_rates = [1, 2, 4, 8]
        else:
            segment_lengths = [4096, 8192, 16384, 32768]
            dilation_rates = [1, 2, 4, 8]

    if dilation_rates is None:
        dilation_rates = [1] * len(segment_lengths)

    try:
        # Ensure sequence length is divisible by world size
        if seq_len % world_size != 0:
            seq_len = (seq_len // world_size) * world_size
            print(
                f"[GPU {rank}] Adjusted seq_len to {seq_len} (divisible by {world_size})"
            )

        # Create model
        device = torch.device(f"cuda:{rank}")
        model = RingDilatedAttentionHybrid(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,  # Use float32 for Pascal GPUs
            ring_size=world_size,
            use_flash_attention=True,
            enable_memory_pool=True,
            use_pattern_cache=True,
        )

        # Create inputs - full sequence on each GPU
        # Ring attention will handle the distribution
        print(f"[GPU {rank}] Creating tensors: seq_len={seq_len}")
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Clear memory stats
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup - just 1 iteration for long sequences
        print(f"[GPU {rank}] Warmup...")
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

        if world_size > 1:
            dist.barrier()

        # Benchmark
        print(f"[GPU {rank}] Benchmarking...")
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        forward_time = time.time() - start_time

        # Get memory stats
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        current_memory_mb = torch.cuda.memory_allocated(device) / 1024**2

        # Calculate metrics
        seq_per_gpu = seq_len // world_size
        total_tokens = batch_size * seq_len * num_heads
        throughput = total_tokens / forward_time / 1e6  # M tokens/s
        memory_per_token = (
            peak_memory_mb * 1024 / (seq_len * batch_size)
        )  # KB per token

        result = {
            "status": "success",
            "seq_len": seq_len,
            "seq_per_gpu": seq_per_gpu,
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
            "world_size": world_size,
            "rank": rank,
            "forward_time_ms": forward_time * 1000,
            "throughput_mtokens": throughput,
            "peak_memory_mb": peak_memory_mb,
            "current_memory_mb": current_memory_mb,
            "memory_per_token_kb": memory_per_token,
        }

        print(
            f"[GPU {rank}] Success: {forward_time * 1000:.1f}ms, {peak_memory_mb:.1f}MB"
        )

        # Clean up
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"[GPU {rank}] Error: {type(e).__name__}: {str(e)}")
        return {
            "status": "error",
            "seq_len": seq_len,
            "rank": rank,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def main():
    """Run multi-GPU long sequence benchmarks."""
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=== Multi-GPU Hybrid Ring Attention - Long Sequence Benchmark ===")
        print(f"World size: {world_size} GPUs")
        print("Testing O(n/p) scaling with long sequences")
        print("NO all_gather operations - true ring communication")
        print("")

    # Test increasingly long sequences
    # Start reasonable and scale up based on GPU count
    base_lengths = [
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        131072,  # 128K
        262144,  # 256K
        524288,  # 512K
        1048576,  # 1M
    ]

    # For multi-GPU, we can handle longer sequences
    if world_size > 1:
        test_lengths = [
            length for length in base_lengths if length >= 16384
        ]  # Start from 16K
    else:
        test_lengths = [
            length for length in base_lengths if length <= 131072
        ]  # Cap at 128K for single GPU

    results = []

    for seq_len in test_lengths:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Testing sequence length: {seq_len:,} tokens")
            print(f"Per GPU: {seq_len // world_size:,} tokens")

        # Synchronize before test
        if world_size > 1:
            dist.barrier()

        result = test_long_sequence(
            seq_len=seq_len,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            rank=rank,
            world_size=world_size,
        )

        results.append(result)

        # Synchronize after test
        if world_size > 1:
            dist.barrier()

        # Only rank 0 prints summary
        if rank == 0 and result["status"] == "success":
            print("\n✓ Success across all GPUs!")
            print(f"  Total sequence: {result['seq_len']:,} tokens")
            print(f"  Per GPU: {result['seq_per_gpu']:,} tokens")
            print(f"  Time: {result['forward_time_ms']:.1f} ms")
            print(f"  Throughput: {result['throughput_mtokens']:.2f} M tokens/s")
            print(f"  Memory/GPU: {result['peak_memory_mb']:.1f} MB")
            print(f"  Memory/token: {result['memory_per_token_kb']:.2f} KB")

        # Stop if we hit OOM
        if (
            result["status"] == "error"
            and "out of memory" in result.get("error", "").lower()
        ):
            if rank == 0:
                print("\nStopping due to OOM error")
            break

    # Calculate memory scaling ratio
    if rank == 0 and len([r for r in results if r["status"] == "success"]) >= 2:
        successful = [r for r in results if r["status"] == "success" and r["rank"] == 0]
        if len(successful) >= 2:
            first = successful[0]
            last = successful[-1]
            memory_ratio = last["memory_per_token_kb"] / first["memory_per_token_kb"]
            seq_ratio = last["seq_len"] / first["seq_len"]

            print(f"\n{'=' * 60}")
            print("Memory Scaling Analysis:")
            print(
                f"  First: {first['seq_len']:,} tokens @ {first['memory_per_token_kb']:.2f} KB/token"
            )
            print(
                f"  Last: {last['seq_len']:,} tokens @ {last['memory_per_token_kb']:.2f} KB/token"
            )
            print(f"  Sequence ratio: {seq_ratio:.1f}x")
            print(f"  Memory ratio: {memory_ratio:.2f}x")
            print(
                f"  {'✅ Excellent' if memory_ratio < 1.5 else '⚠️  Suboptimal'} O(n/p) scaling!"
            )

    # Each GPU saves its own results (no all_gather!)
    output = {
        "world_size": world_size,
        "rank": rank,
        "results": results,
    }

    filename = f"hybrid_long_seq_rank{rank}_{world_size}gpu.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[GPU {rank}] Results saved to {filename}")

    # Summary from rank 0
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"GPUs: {world_size}")
        print("\nSuccessful sequence lengths:")
        for r in results:
            if r["status"] == "success":
                print(
                    f"  {r['seq_len']:>9,} total ({r['seq_per_gpu']:>7,}/GPU): "
                    f"{r['forward_time_ms']:>7.1f}ms, "
                    f"{r['peak_memory_mb']:>7.1f}MB/GPU"
                )

        max_successful = max(
            [r["seq_len"] for r in results if r["status"] == "success"], default=0
        )
        if max_successful > 0:
            print(f"\nMaximum successful sequence: {max_successful:,} tokens")
            print(f"Maximum per GPU: {max_successful // world_size:,} tokens")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
