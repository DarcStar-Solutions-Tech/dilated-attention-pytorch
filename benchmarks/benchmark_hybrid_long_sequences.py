#!/usr/bin/env python3
"""
Benchmark hybrid ring dilated attention with very long sequences.
Tests scaling behavior up to 128K+ tokens.
"""

import torch
import torch.distributed as dist
import time
import os
import gc
from typing import Dict, List
import json
import psutil
import GPUtil

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def get_memory_info():
    """Get detailed memory information."""
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        return {
            "gpu_used_mb": gpu.memoryUsed,
            "gpu_total_mb": gpu.memoryTotal,
            "gpu_util_percent": gpu.memoryUtil * 100,
            "cpu_ram_gb": psutil.virtual_memory().used / 1024**3,
            "cpu_ram_percent": psutil.virtual_memory().percent,
        }
    return {
        "cpu_ram_gb": psutil.virtual_memory().used / 1024**3,
        "cpu_ram_percent": psutil.virtual_memory().percent,
    }


def setup_distributed():
    """Initialize distributed training."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(rank)

    return rank, world_size


def test_sequence_length(
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict:
    """Test a specific sequence length."""

    # Adaptive segment lengths based on sequence length
    if segment_lengths is None:
        if seq_len <= 8192:
            segment_lengths = [min(2048, seq_len)]
            dilation_rates = [1]
        elif seq_len <= 32768:
            segment_lengths = [2048, 4096]
            dilation_rates = [1, 2]
        elif seq_len <= 131072:
            segment_lengths = [2048, 4096, 8192]
            dilation_rates = [1, 2, 4]
        else:
            segment_lengths = [2048, 4096, 8192, 16384]
            dilation_rates = [1, 2, 4, 8]

    if dilation_rates is None:
        dilation_rates = [1] * len(segment_lengths)

    # Get initial memory state
    initial_memory = get_memory_info()

    try:
        # Create model
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        model = RingDilatedAttentionHybrid(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,  # Use float32 for stability
            use_flash_attention=True,
            enable_memory_pool=True,
            use_pattern_cache=True,
        )

        # Adjust sequence length to be divisible by world size
        seq_len_per_gpu = seq_len // world_size
        if seq_len % world_size != 0:
            seq_len = seq_len_per_gpu * world_size
            if rank == 0:
                print(f"  Adjusted seq_len to {seq_len} (divisible by {world_size})")

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Get memory after allocation
        post_alloc_memory = get_memory_info()

        # Warmup
        if rank == 0:
            print("  Running warmup...")
        for _ in range(2):
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            if world_size > 1:
                dist.barrier()

        # Time the forward pass
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()

        forward_time = time.time() - start_time

        # Get peak memory
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        current_memory_mb = torch.cuda.memory_allocated(device) / 1024**2

        # Get final memory state
        final_memory = get_memory_info()

        # Calculate metrics
        total_tokens = batch_size * seq_len * num_heads
        throughput = total_tokens / forward_time / 1e6  # M tokens/s
        memory_per_token = (
            peak_memory_mb * 1024 / (seq_len * batch_size)
        )  # KB per token
        expected_memory_scaling = seq_len / world_size  # O(n/p) expectation

        result = {
            "status": "success",
            "seq_len": seq_len,
            "seq_len_per_gpu": seq_len // world_size,
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
            "expected_scaling": expected_memory_scaling,
            "memory_info": {
                "initial": initial_memory,
                "post_allocation": post_alloc_memory,
                "final": final_memory,
            },
        }

        # Clean up
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except Exception as e:
        # Get error memory state
        error_memory = get_memory_info()

        return {
            "status": "error",
            "seq_len": seq_len,
            "error": str(e),
            "error_type": type(e).__name__,
            "memory_at_error": error_memory,
        }


def find_max_sequence_length(
    rank: int,
    world_size: int,
    start_seq_len: int = 16384,
    max_seq_len: int = 1048576,  # 1M tokens
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
) -> int:
    """Find the maximum sequence length that fits in memory."""

    if rank == 0:
        print(f"\nFinding maximum sequence length for {world_size} GPU(s)...")

    current = start_seq_len
    max_working = 0

    while current <= max_seq_len:
        if rank == 0:
            print(f"\nTrying seq_len={current:,}...")

        result = test_sequence_length(
            seq_len=current,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            rank=rank,
            world_size=world_size,
        )

        if result["status"] == "success":
            max_working = current
            if rank == 0:
                print(
                    f"  ✓ Success: {result['forward_time_ms']:.1f}ms, "
                    f"{result['peak_memory_mb']:.1f}MB, "
                    f"{result['memory_per_token_kb']:.2f}KB/token"
                )

            # Try doubling
            current *= 2
        else:
            if rank == 0:
                print(f"  ✗ Failed: {result['error_type']} - {result['error']}")

            # Binary search between max_working and current
            if max_working > 0:
                mid = (max_working + current) // 2
                if mid == max_working:
                    break
                current = mid
            else:
                # First failure, try half
                current //= 2
                if current < 1024:
                    break

    return max_working


def main():
    """Run long sequence benchmarks."""
    rank, world_size = setup_distributed()

    if rank == 0:
        print("=== Hybrid Ring Attention - Long Sequence Benchmark ===")
        print(f"World size: {world_size}")
        print("Testing with batch_size=1, num_heads=8, head_dim=64")

        # Get system info
        mem_info = get_memory_info()
        print("\nSystem Memory:")
        if "gpu_total_mb" in mem_info:
            print(
                f"  GPU: {mem_info['gpu_used_mb']:.1f}/{mem_info['gpu_total_mb']:.1f} MB "
                f"({mem_info['gpu_util_percent']:.1f}%)"
            )
        print(
            f"  CPU RAM: {mem_info['cpu_ram_gb']:.1f} GB ({mem_info['cpu_ram_percent']:.1f}%)"
        )

    # Test specific sequence lengths
    test_lengths = [
        8192,  # 8K
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        131072,  # 128K
        # 262144,   # 256K - skip for now
        # 524288,   # 512K - skip for now
        # 1048576,  # 1M - skip for now
    ]

    results = []

    for seq_len in test_lengths:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Testing sequence length: {seq_len:,} tokens")

        result = test_sequence_length(
            seq_len=seq_len,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            rank=rank,
            world_size=world_size,
        )

        results.append(result)

        if rank == 0:
            if result["status"] == "success":
                print("\n✓ Success!")
                print(f"  Time: {result['forward_time_ms']:.1f} ms")
                print(f"  Throughput: {result['throughput_mtokens']:.2f} M tokens/s")
                print(f"  Memory/GPU: {result['peak_memory_mb']:.1f} MB")
                print(f"  Memory/token: {result['memory_per_token_kb']:.2f} KB")
                print(f"  Segments: {result['segment_lengths']}")
                print(f"  Dilation: {result['dilation_rates']}")
            else:
                print(f"\n✗ Failed: {result['error_type']}")
                print(f"  Error: {result['error']}")

        # Stop if we hit OOM
        if result["status"] == "error" and "out of memory" in result["error"].lower():
            if rank == 0:
                print("\nStopping due to OOM error")
            break

    # Find maximum sequence length
    if rank == 0:
        print(f"\n{'=' * 60}")

    max_seq_len = find_max_sequence_length(
        rank=rank,
        world_size=world_size,
        start_seq_len=results[-1]["seq_len"]
        if results[-1]["status"] == "success"
        else 16384,
    )

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Maximum working sequence length: {max_seq_len:,} tokens")
        print(f"Per GPU: {max_seq_len // world_size:,} tokens")

    # Save results
    if rank == 0:
        output = {
            "world_size": world_size,
            "system_memory": get_memory_info(),
            "test_results": [r for r in results if r["rank"] == 0],
            "max_sequence_length": max_seq_len,
            "max_seq_per_gpu": max_seq_len // world_size,
        }

        filename = f"hybrid_long_seq_results_{world_size}gpu.json"
        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {filename}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"GPUs: {world_size}")
        print(f"Max sequence length: {max_seq_len:,} tokens")
        print(f"Max per GPU: {max_seq_len // world_size:,} tokens")
        print("\nSuccessful runs:")
        for r in results:
            if r["status"] == "success" and r["rank"] == 0:
                print(
                    f"  {r['seq_len']:>9,}: {r['forward_time_ms']:>7.1f}ms, "
                    f"{r['peak_memory_mb']:>7.1f}MB, "
                    f"{r['memory_per_token_kb']:>6.2f}KB/token"
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
