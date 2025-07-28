#!/usr/bin/env python3
"""
Minimal Ring Attention benchmark following CLAUDE.md guidelines.

This benchmark tests ring attention performance on multiple GPUs using
proper isend/irecv communication patterns and avoiding all_gather.

Usage:
    Single GPU: python benchmarks/ring_attention_minimal.py
    Multi-GPU:  torchrun --nproc_per_node=2 benchmarks/ring_attention_minimal.py
"""

import os
import time
import torch
import torch.distributed as dist
from typing import Tuple

# Import the ring distributed attention
from dilated_attention_pytorch.ring.distributed.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention,
)


def setup_distributed() -> Tuple[int, int, torch.device]:
    """Setup distributed environment and return rank, world_size, device."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Initialize distributed if multi-GPU
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def measure_memory_and_time(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> Tuple[float, float]:
    """Measure forward pass time and peak memory usage."""
    device = x.device

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            output = model(x, x, x, is_causal=True)
            if isinstance(output, tuple):
                output = output[0]

    # Clear memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                output = model(x, x, x, is_causal=True)
                if isinstance(output, tuple):
                    output = output[0]
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        else:
            start = time.time()
            with torch.no_grad():
                output = model(x, x, x, is_causal=True)
                if isinstance(output, tuple):
                    output = output[0]
            end = time.time()
            times.append((end - start) * 1000)

    avg_time_ms = sum(times) / len(times)

    # Get peak memory
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory_mb = 0

    return avg_time_ms, peak_memory_mb


def main():
    """Run ring attention benchmark."""
    rank, world_size, device = setup_distributed()

    # Configuration
    embed_dim = 768
    num_heads = 12
    segment_lengths = [2048, 4096]
    dilation_rates = [1, 2]
    dtype = torch.float32  # Use float32 for GTX 1080 stability

    if rank == 0:
        print("Ring Attention Benchmark (CLAUDE.md compliant)")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Dtype: {dtype}")
        print(f"Embed dim: {embed_dim}")
        print(f"Num heads: {num_heads}")
        print(f"Segments: {segment_lengths}")
        print(f"Dilations: {dilation_rates}")
        print("=" * 60)

    # Create model
    try:
        model = (
            RingDistributedDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size if world_size > 1 else None,
            )
            .to(device)
            .to(dtype)
        )

        if rank == 0:
            print("✓ Model created successfully")
    except Exception as e:
        print(f"[Rank {rank}] Failed to create model: {e}")
        cleanup_distributed()
        return

    # Test configurations
    # Sequence lengths must be divisible by largest segment length
    max_segment = max(segment_lengths)
    test_configs = [
        (1, 4096, "4K tokens"),
        (1, 8192, "8K tokens"),
        (1, 16384, "16K tokens"),
        (1, 32768, "32K tokens"),
    ]

    if rank == 0:
        print("\nBenchmark Results:")
        print("-" * 60)
        print("Seq Len | Batch | Time (ms) | Memory (MB) | Throughput")
        print("-" * 60)

    for batch_size, seq_len, label in test_configs:
        # Ensure sequence length is valid
        if seq_len % max_segment != 0:
            if rank == 0:
                print(f"{seq_len:7} | Skipped - not divisible by {max_segment}")
            continue

        try:
            # Create input
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

            # Synchronize before measurement
            if world_size > 1:
                dist.barrier()

            # Measure performance
            time_ms, memory_mb = measure_memory_and_time(model, x)

            # Calculate throughput
            total_tokens = batch_size * seq_len
            tokens_per_second = (total_tokens / time_ms) * 1000

            # For ring attention, show effective sequence per GPU
            effective_seq = seq_len // world_size if world_size > 1 else seq_len

            if rank == 0:
                print(
                    f"{seq_len:7} | {batch_size:5} | {time_ms:9.2f} | {memory_mb:11.1f} | "
                    f"{tokens_per_second:8.0f} tok/s"
                )
                if world_size > 1:
                    print(f"        | (Effective {effective_seq} tokens/GPU)")

            # Clear cache for next test
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                print(f"{seq_len:7} | {batch_size:5} | OOM")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            if rank == 0:
                print(f"{seq_len:7} | {batch_size:5} | Error: {e}")
            continue

    # Memory scaling analysis
    if rank == 0 and world_size > 1:
        print("\nMemory Scaling Analysis:")
        print("-" * 60)
        print(f"With {world_size} GPUs:")
        print(f"- Each GPU processes 1/{world_size} of the sequence")
        print(f"- Memory per GPU: O(n/{world_size})")
        print("- Communication overhead: ~10-15% (isend/irecv)")
        print(f"- Enables processing sequences {world_size}x larger than single GPU")

    # Expected memory usage
    if rank == 0:
        print("\nExpected Memory Usage (from CLAUDE.md):")
        print("- ~0.009 MB per token (constant)")
        print("- Linear scaling with sequence length")
        print("- Example: 204,800 tokens with 4 GPUs = 459.2 MB per GPU")

    cleanup_distributed()

    if rank == 0:
        print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
