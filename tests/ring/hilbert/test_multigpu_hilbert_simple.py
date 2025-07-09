#!/usr/bin/env python3
"""
Simple multi-GPU test for Hilbert ring attention focusing on key metrics.

Run with: torchrun --nproc_per_node=2 tests/test_multigpu_hilbert_simple.py
"""

import os
import time
import torch
import torch.distributed as dist
from datetime import datetime

from dilated_attention_pytorch.ring.hilbert.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def setup():
    """Setup distributed environment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, device


def test_single_vs_multi_gpu():
    """Compare single GPU vs multi-GPU performance."""
    rank, world_size, device = setup()

    if rank == 0:
        print(f"\nComparing Single GPU vs {world_size}-GPU Performance")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"GPUs: {world_size}x {torch.cuda.get_device_name(0)}")

    # Test configuration
    batch_size = 1
    seq_len = 16384  # 16K tokens
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim
    num_iterations = 20

    # For multi-GPU, each GPU handles part of the sequence
    local_seq_len = seq_len // world_size

    # Segment configuration for local sequence
    segment_lengths = [2048, 4096, local_seq_len]
    dilation_rates = [1, 2, 4]

    # Create attention module
    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        use_hilbert=True,
        ring_size=world_size,
        dropout=0.0,
    ).to(device)

    # Create local inputs
    q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(5):
        _ = attention(q, k, v)

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    times = []
    memory_before = torch.cuda.memory_allocated(device) / 1e9

    for i in range(num_iterations):
        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()
        output = attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

        if rank == 0 and i == 0:
            print("\nFirst iteration details:")
            print(f"  Input shape per GPU: {q.shape}")
            print(f"  Output shape per GPU: {output.shape}")
            print(f"  Segment config: {segment_lengths}")

    memory_after = torch.cuda.max_memory_allocated(device) / 1e9
    memory_used = memory_after - memory_before

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Gather results
    metrics = torch.tensor([avg_time, memory_used], device=device)
    all_metrics = [torch.zeros_like(metrics) for _ in range(world_size)]
    dist.all_gather(all_metrics, metrics)

    if rank == 0:
        all_metrics_np = torch.stack(all_metrics).cpu().numpy()
        avg_times = all_metrics_np[:, 0]
        memory_usages = all_metrics_np[:, 1]

        print(f"\nPerformance Results ({seq_len} tokens total):")
        print(f"  Average time: {avg_times.mean() * 1000:.2f} ms")
        print(f"  Min/Max time: {min_time * 1000:.2f} / {max_time * 1000:.2f} ms")
        print(
            f"  Throughput: {(batch_size * seq_len) / avg_times.mean() / 1e6:.2f} M tokens/sec"
        )
        print(f"  Memory per GPU: {memory_usages.max():.2f} GB")
        print(f"  Total memory: {memory_usages.sum():.2f} GB")
        print(f"  Tokens per GPU: {local_seq_len}")

        # Estimate single GPU performance (linear scaling assumption)
        single_gpu_estimate = avg_times.mean() * world_size
        print("\nScaling Analysis:")
        print(f"  Estimated single GPU time: {single_gpu_estimate * 1000:.2f} ms")
        print(f"  Speedup vs single GPU: {single_gpu_estimate / avg_times.mean():.2f}x")
        print(f"  Scaling efficiency: {100 / world_size:.1f}%")

    dist.barrier()


def test_weak_scaling():
    """Test weak scaling - keep tokens per GPU constant."""
    rank, world_size, device = setup()

    if rank == 0:
        print("\n" + "=" * 80)
        print("Weak Scaling Test (constant tokens per GPU)")
        print("=" * 80)

    tokens_per_gpu = 8192
    batch_size = 1
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim

    # Total sequence length scales with GPUs
    total_seq_len = tokens_per_gpu * world_size

    # Create attention
    segment_lengths = [2048, 4096, tokens_per_gpu]
    dilation_rates = [1, 2, 4]

    attention = RingDilatedAttentionHilbertOptimizedFixed(
        dim=embed_dim,
        heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        use_hilbert=True,
        ring_size=world_size,
        dropout=0.0,
    ).to(device)

    # Create inputs
    q = torch.randn(batch_size, tokens_per_gpu, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, tokens_per_gpu, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, tokens_per_gpu, num_heads, head_dim, device=device)

    # Benchmark
    num_iterations = 20
    times = []

    for _ in range(5):  # warmup
        _ = attention(q, k, v)

    torch.cuda.synchronize()
    dist.barrier()

    for _ in range(num_iterations):
        dist.barrier()
        start = time.perf_counter()
        _ = attention(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    if rank == 0:
        print("Configuration:")
        print(f"  GPUs: {world_size}")
        print(f"  Tokens per GPU: {tokens_per_gpu}")
        print(f"  Total tokens: {total_seq_len}")
        print("\nResults:")
        print(f"  Time per iteration: {avg_time * 1000:.2f} ms")
        print(f"  Throughput: {total_seq_len / avg_time / 1e6:.2f} M tokens/sec")
        print(
            f"  Throughput per GPU: {tokens_per_gpu / avg_time / 1e6:.2f} M tokens/sec/GPU"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    test_single_vs_multi_gpu()
    test_weak_scaling()
