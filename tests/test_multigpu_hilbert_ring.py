#!/usr/bin/env python3
"""
Test multi-GPU performance of Hilbert ring dilated attention.

This script MUST be run with torchrun:
torchrun --nproc_per_node=2 tests/test_multigpu_hilbert_ring.py
"""

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Dict

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)


def setup_distributed():
    """Setup distributed environment."""
    if "RANK" not in os.environ:
        raise RuntimeError("This script must be run with torchrun")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Cleanup distributed environment."""
    dist.destroy_process_group()


def benchmark_attention(
    attention_module: nn.Module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    num_iterations: int = 10,
    warmup: int = 3,
) -> Dict[str, float]:
    """Benchmark attention module on multi-GPU."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Calculate local sequence length
    local_seq_len = seq_len // world_size

    # Create local inputs
    q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(warmup):
        _ = attention_module(q, k, v)

    torch.cuda.synchronize()
    dist.barrier()

    # Timing
    times = []
    for _ in range(num_iterations):
        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()
        _ = attention_module(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    # Get memory usage
    memory_allocated = torch.cuda.max_memory_allocated(device) / 1e9  # GB

    # Gather results on rank 0
    times_tensor = torch.tensor(times, device=device)
    memory_tensor = torch.tensor([memory_allocated], device=device)

    if rank == 0:
        all_times = [torch.zeros_like(times_tensor) for _ in range(world_size)]
        all_memory = [torch.zeros_like(memory_tensor) for _ in range(world_size)]
    else:
        all_times = None
        all_memory = None

    dist.gather(times_tensor, all_times, dst=0)
    dist.gather(memory_tensor, all_memory, dst=0)

    if rank == 0:
        # Calculate statistics
        all_times_np = torch.stack(all_times).cpu().numpy()
        mean_time = all_times_np.mean()
        std_time = all_times_np.std()

        all_memory_np = torch.stack(all_memory).cpu().numpy()
        max_memory = all_memory_np.max()
        total_memory = all_memory_np.sum()

        return {
            "mean_time": mean_time,
            "std_time": std_time,
            "max_memory_per_gpu": max_memory,
            "total_memory": total_memory,
            "throughput": (batch_size * seq_len) / mean_time / 1e6,  # M tokens/sec
        }

    return {}


def test_scaling():
    """Test scaling with different sequence lengths."""
    rank, world_size, local_rank, device = setup_distributed()

    if rank == 0:
        print("Testing Multi-GPU Hilbert Ring Dilated Attention")
        print(f"World size: {world_size}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print("=" * 80)

    # Test configurations
    configs = [
        # (batch_size, seq_len, description)
        (1, 8192, "8K tokens"),
        (1, 16384, "16K tokens"),
        (1, 32768, "32K tokens"),
        (1, 65536, "64K tokens"),
        (1, 131072, "128K tokens"),
    ]

    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim

    results = []

    for batch_size, seq_len, desc in configs:
        if rank == 0:
            print(f"\nTesting {desc} (batch={batch_size}, seq_len={seq_len})")
            print("-" * 60)

        # Skip if not divisible by world size
        if seq_len % world_size != 0:
            if rank == 0:
                print(f"Skipping - sequence length must be divisible by {world_size}")
            continue

        # Adjust segment lengths based on local sequence
        local_seq_len = seq_len // world_size
        segment_lengths = []
        base = 2048
        while base <= local_seq_len:
            segment_lengths.append(base)
            base *= 2
        if not segment_lengths or segment_lengths[-1] != local_seq_len:
            segment_lengths.append(local_seq_len)

        # Create dilation rates
        dilation_rates = [2**i for i in range(len(segment_lengths))]

        # Skip if incompatible
        if local_seq_len % segment_lengths[-1] != 0:
            if rank == 0:
                print("Skipping - incompatible segment configuration")
            continue

        try:
            # Test with Hilbert
            attention_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
                dim=embed_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=True,
                ring_size=world_size,
                dropout=0.0,
            ).to(device)

            result_hilbert = benchmark_attention(
                attention_hilbert,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device,
            )

            # Test without Hilbert
            attention_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
                dim=embed_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=False,
                ring_size=world_size,
                dropout=0.0,
            ).to(device)

            result_no_hilbert = benchmark_attention(
                attention_no_hilbert,
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device,
            )

            if rank == 0:
                print("\nWith Hilbert:")
                print(
                    f"  Time: {result_hilbert['mean_time'] * 1000:.2f}±{result_hilbert['std_time'] * 1000:.2f} ms"
                )
                print(
                    f"  Memory per GPU: {result_hilbert['max_memory_per_gpu']:.2f} GB"
                )
                print(f"  Total memory: {result_hilbert['total_memory']:.2f} GB")
                print(f"  Throughput: {result_hilbert['throughput']:.2f} M tokens/sec")

                print("\nWithout Hilbert:")
                print(
                    f"  Time: {result_no_hilbert['mean_time'] * 1000:.2f}±{result_no_hilbert['std_time'] * 1000:.2f} ms"
                )
                print(
                    f"  Memory per GPU: {result_no_hilbert['max_memory_per_gpu']:.2f} GB"
                )
                print(f"  Total memory: {result_no_hilbert['total_memory']:.2f} GB")
                print(
                    f"  Throughput: {result_no_hilbert['throughput']:.2f} M tokens/sec"
                )

                speedup = result_no_hilbert["mean_time"] / result_hilbert["mean_time"]
                print(f"\nHilbert speedup: {speedup:.2f}x")

                results.append(
                    {
                        "seq_len": seq_len,
                        "desc": desc,
                        "hilbert_time": result_hilbert["mean_time"],
                        "no_hilbert_time": result_no_hilbert["mean_time"],
                        "speedup": speedup,
                        "memory_per_gpu": result_hilbert["max_memory_per_gpu"],
                    }
                )

        except Exception as e:
            if rank == 0:
                print(f"Error: {e}")

        # Clean up
        torch.cuda.empty_cache()
        dist.barrier()

    # Summary
    if rank == 0 and results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\n{'Seq Length':>12} | {'Speedup':>8} | {'Memory/GPU':>11}")
        print("-" * 40)
        for r in results:
            print(
                f"{r['seq_len']:>12} | {r['speedup']:>8.2f}x | {r['memory_per_gpu']:>9.2f} GB"
            )

        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        print(f"\nAverage Hilbert speedup: {avg_speedup:.2f}x")

    cleanup_distributed()


def test_communication_overhead():
    """Test communication overhead in ring attention."""
    rank, world_size, local_rank, device = setup_distributed()

    if rank == 0:
        print("\n" + "=" * 80)
        print("Testing Communication Overhead")
        print("=" * 80)

    seq_len = 16384
    batch_size = 1
    num_heads = 8
    head_dim = 64
    _ = num_heads * head_dim
    local_seq_len = seq_len // world_size

    # Create inputs
    q = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    _ = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)
    _ = torch.randn(batch_size, local_seq_len, num_heads, head_dim, device=device)

    # Test raw communication time
    tensor_size = q.numel() * q.element_size() / 1e6  # MB

    # Time ring communication
    num_iterations = 100
    dist.barrier()

    start = time.perf_counter()
    for _ in range(num_iterations):
        # Simulate ring pass
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1) % world_size

        send_req = dist.isend(q.contiguous(), send_rank)
        recv_buffer = torch.empty_like(q)
        recv_req = dist.irecv(recv_buffer, recv_rank)

        send_req.wait()
        recv_req.wait()

    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()

    comm_time = (end - start) / num_iterations * 1000  # ms
    bandwidth = tensor_size / (comm_time / 1000)  # MB/s

    if rank == 0:
        print(f"Tensor size: {tensor_size:.2f} MB")
        print(f"Ring communication time: {comm_time:.2f} ms")
        print(f"Effective bandwidth: {bandwidth:.2f} MB/s")
        print(
            f"Theoretical time for {world_size} passes: {comm_time * world_size:.2f} ms"
        )


if __name__ == "__main__":
    test_scaling()
    test_communication_overhead()
