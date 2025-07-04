#!/usr/bin/env python3
"""
Simple benchmark for Hybrid Hilbert achieving 262K tokens.
"""

import os
import torch
import torch.distributed as dist
import time
import argparse


def init_distributed():
    """Initialize distributed if needed."""
    if not dist.is_initialized() and "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def benchmark_sequence(seq_len, use_hilbert=True):
    """Benchmark a specific sequence length."""
    rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(
            f"Testing {seq_len:,} tokens with {'Hilbert' if use_hilbert else 'Standard'}"
        )
        print(f"World size: {world_size} GPUs")
        print(f"{'=' * 60}")

    try:
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
            RingDilatedAttentionHybridOptimizedV2,
        )
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        # Parameters
        batch_size = 1
        num_heads = 8
        hidden_dim = 512
        head_dim = hidden_dim // num_heads

        # Check memory before
        if rank == 0:
            allocated = torch.cuda.memory_allocated(rank) / 1024**3
            free = (
                torch.cuda.get_device_properties(rank).total_memory
                - torch.cuda.memory_reserved(rank)
            ) / 1024**3
            print(f"Memory before: {allocated:.1f} GB allocated, {free:.1f} GB free")

        # Create model
        if use_hilbert:
            model = RingDilatedAttentionHybridHilbert(
                segment_lengths=[4096],
                dilation_rates=[1],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                enable_memory_pool=True,
                use_pattern_cache=True,
                use_hilbert=True,
                hilbert_chunk_size=min(8192, seq_len // world_size),
            )
        else:
            model = RingDilatedAttentionHybridOptimizedV2(
                segment_lengths=[4096],
                dilation_rates=[1],
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=torch.float32,
                enable_memory_pool=True,
                use_pattern_cache=True,
            )

        # Create tensors
        if rank == 0:
            print("Creating tensors...")

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Check memory after allocation
        if rank == 0:
            allocated = torch.cuda.memory_allocated(rank) / 1024**3
            print(f"Memory after tensors: {allocated:.1f} GB allocated")

        # Warmup
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            print("Warming up...")

        for i in range(2):
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            if world_size > 1:
                dist.barrier()

        # Benchmark
        if rank == 0:
            print("Benchmarking...")

        times = []
        for i in range(5):
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

            if rank == 0:
                print(f"  Iteration {i + 1}: {elapsed:.1f} ms")

        # Results
        if rank == 0:
            avg_time = sum(times) / len(times)
            tokens_per_sec = (seq_len * batch_size) / (avg_time / 1000)

            print("\nResults:")
            print(f"  Average time: {avg_time:.1f} ms")
            print(f"  Tokens/second: {tokens_per_sec:,.0f}")
            print(f"  Output shape: {output.shape}")

            # Final memory
            allocated = torch.cuda.memory_allocated(rank) / 1024**3
            print(f"  Peak memory: {allocated:.1f} GB")

        return avg_time if rank == 0 else None

    except Exception as e:
        if rank == 0:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()
        return None
    finally:
        # Cleanup
        if "q" in locals():
            del q, k, v
        if "output" in locals():
            del output
        if "model" in locals():
            del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    rank, world_size = init_distributed()

    if rank == 0:
        print("\nHYBRID RING ATTENTION BENCHMARK")
        print("Target: 262K tokens like the original implementation")

    if args.compare:
        # Test standard
        time_standard = benchmark_sequence(args.seq_len, use_hilbert=False)

        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

        # Test Hilbert
        time_hilbert = benchmark_sequence(args.seq_len, use_hilbert=True)

        if rank == 0 and time_standard and time_hilbert:
            print(f"\n{'=' * 60}")
            print("COMPARISON:")
            print(f"  Standard: {time_standard:.1f} ms")
            print(f"  Hilbert:  {time_hilbert:.1f} ms")
            print(f"  Speedup:  {time_standard / time_hilbert:.2f}x")
            print(f"{'=' * 60}")
    else:
        benchmark_sequence(args.seq_len, use_hilbert=True)

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
