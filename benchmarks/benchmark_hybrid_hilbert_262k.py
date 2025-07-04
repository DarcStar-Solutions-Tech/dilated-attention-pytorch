#!/usr/bin/env python3
"""
Benchmark Hybrid Hilbert Ring Attention for 262K+ tokens.

This tests the Hilbert-enhanced version of the ring attention
that successfully achieved 262K tokens on the current hardware.
"""

import os
import torch
import torch.distributed as dist
import time
import argparse
from datetime import datetime
import json
import psutil
import GPUtil


def init_distributed():
    """Initialize distributed environment."""
    if dist.is_initialized():
        # Already initialized
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return rank, local_rank, world_size

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")

        return rank, local_rank, int(os.environ["WORLD_SIZE"])
    else:
        return 0, 0, 1


def get_memory_info(device_id=0):
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3

        gpus = GPUtil.getGPUs()
        if device_id < len(gpus):
            gpu = gpus[device_id]
            total = gpu.memoryTotal / 1024  # Convert to GB
            used = gpu.memoryUsed / 1024
            free = gpu.memoryFree / 1024

            return {
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "used_gb": used,
                "free_gb": free,
                "gpu_name": gpu.name,
            }
    return None


def benchmark_sequence_length(
    seq_len,
    hidden_dim=512,
    num_heads=8,
    segment_lengths=[4096],
    dilation_rates=[1],
    batch_size=1,
    dtype=torch.float32,
    use_hilbert=True,
    warmup=2,
    iterations=5,
):
    """Benchmark a specific sequence length."""
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Testing sequence length: {seq_len:,} tokens")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size: {batch_size}")
        print(f"Hidden dim: {hidden_dim}, Heads: {num_heads}")
        print(f"Segment lengths: {segment_lengths}")
        print(f"Dilation rates: {dilation_rates}")
        print(f"Using Hilbert: {use_hilbert}")
        print(f"{'=' * 60}")

    try:
        # Import the implementations
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized_v2 import (
            RingDilatedAttentionHybridOptimizedV2,
        )
        from dilated_attention_pytorch.ring_dilated_attention_hybrid_hilbert import (
            RingDilatedAttentionHybridHilbert,
        )

        # Memory check before allocation
        if rank == 0:
            mem_info = get_memory_info(local_rank)
            if mem_info:
                print("\nMemory before allocation:")
                print(f"  GPU: {mem_info['gpu_name']}")
                print(f"  Total: {mem_info['total_gb']:.1f} GB")
                print(f"  Free: {mem_info['free_gb']:.1f} GB")
                print(f"  Used: {mem_info['used_gb']:.1f} GB")

        # Create model
        if use_hilbert:
            model = RingDilatedAttentionHybridHilbert(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=dtype,
                enable_memory_pool=True,
                use_pattern_cache=True,
                precompute_patterns=True,
                use_hilbert=True,
                hilbert_chunk_size=min(8192, seq_len // world_size),
            )
            model_name = "Hybrid Hilbert"
        else:
            model = RingDilatedAttentionHybridOptimizedV2(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                ring_size=world_size,
                device=device,
                dtype=dtype,
                enable_memory_pool=True,
                use_pattern_cache=True,
                precompute_patterns=True,
            )
            model_name = "Hybrid Standard"

        # Create input tensors
        head_dim = hidden_dim // num_heads
        chunk_size = seq_len // world_size

        if rank == 0:
            print("\nAllocating tensors...")
            print(f"  Full shape: ({batch_size}, {seq_len}, {num_heads}, {head_dim})")
            print(f"  Chunk per GPU: {chunk_size:,} tokens")
            print(
                f"  Memory per tensor: {batch_size * seq_len * num_heads * head_dim * 2 / 1024**3:.2f} GB"
            )

        # Allocate tensors
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Synchronize
        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        # Memory after allocation
        if rank == 0:
            mem_info = get_memory_info(local_rank)
            if mem_info:
                print("\nMemory after allocation:")
                print(f"  Free: {mem_info['free_gb']:.1f} GB")
                print(f"  Allocated: {mem_info['allocated_gb']:.1f} GB")

        # Warmup
        if rank == 0:
            print("\nWarming up...")

        for i in range(warmup):
            with torch.no_grad():
                _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()

            if rank == 0:
                print(f"  Warmup {i + 1}/{warmup} complete")

        # Benchmark
        if rank == 0:
            print(f"\nBenchmarking {iterations} iterations...")

        torch.cuda.synchronize()
        if world_size > 1:
            dist.barrier()

        start_time = time.perf_counter()

        for i in range(iterations):
            with torch.no_grad():
                output = model(q, k, v, is_causal=False)

            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()

            if rank == 0 and (i + 1) % max(1, iterations // 5) == 0:
                print(f"  Iteration {i + 1}/{iterations} complete")

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time = total_time / iterations * 1000  # ms

        # Verify output shape
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

        # Collect results
        if rank == 0:
            mem_info = get_memory_info(local_rank)

            results = {
                "model": model_name,
                "seq_len": seq_len,
                "world_size": world_size,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "avg_time_ms": avg_time,
                "total_time_s": total_time,
                "iterations": iterations,
                "tokens_per_second": (seq_len * batch_size * iterations) / total_time,
                "memory": mem_info,
                "success": True,
            }

            print(f"\n{'=' * 60}")
            print(f"RESULTS for {seq_len:,} tokens:")
            print(f"  Average time: {avg_time:.1f} ms")
            print(f"  Tokens/second: {results['tokens_per_second']:,.0f}")
            print(f"  Peak memory: {mem_info['allocated_gb']:.1f} GB")
            print(f"{'=' * 60}")

            return results

        return None

    except Exception as e:
        if rank == 0:
            print(f"\nERROR: {e}")
            import traceback

            traceback.print_exc()

            return {
                "model": model_name if "model_name" in locals() else "Unknown",
                "seq_len": seq_len,
                "world_size": world_size,
                "error": str(e),
                "success": False,
            }
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
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=262144,
        help="Maximum sequence length to test",
    )
    parser.add_argument(
        "--start-seq-len", type=int, default=32768, help="Starting sequence length"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Benchmark iterations"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with and without Hilbert"
    )
    args = parser.parse_args()

    rank, local_rank, world_size = init_distributed()

    if rank == 0:
        print("\n" + "=" * 80)
        print("HYBRID HILBERT RING ATTENTION - 262K TOKEN BENCHMARK")
        print("=" * 80)
        print(f"Running on {world_size} GPU(s)")
        print("Target: Achieve 262K+ tokens with Hilbert ordering")

        # System info
        print("\nSystem Info:")
        print(f"  CPU: {psutil.cpu_count()} cores")
        print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")

        # Get GPU info
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
                print(f"    Memory: {gpu.memoryTotal / 1024:.1f} GB")
                print(f"    Driver: {gpu.driver}")
        except:
            pass

    # Test sequence lengths
    seq_lengths = []
    current = args.start_seq_len
    while current <= args.max_seq_len:
        # Ensure divisible by world size
        if current % world_size == 0:
            seq_lengths.append(current)
        current *= 2

    all_results = []

    # Test each sequence length
    for seq_len in seq_lengths:
        if args.compare:
            # Test without Hilbert
            result_standard = benchmark_sequence_length(
                seq_len=seq_len,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                batch_size=args.batch_size,
                use_hilbert=False,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            if rank == 0 and result_standard:
                all_results.append(result_standard)

            # Clean up between tests
            torch.cuda.empty_cache()
            if world_size > 1:
                dist.barrier()

        # Test with Hilbert
        result_hilbert = benchmark_sequence_length(
            seq_len=seq_len,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            batch_size=args.batch_size,
            use_hilbert=True,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        if rank == 0 and result_hilbert:
            all_results.append(result_hilbert)

        # Clean up
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

    # Save results
    if rank == 0 and all_results:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
        filename = f"hybrid_hilbert_262k_{world_size}gpu_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "world_size": world_size,
                    "args": vars(args),
                    "results": all_results,
                },
                f,
                indent=2,
            )

        print(f"\nResults saved to {filename}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        successful = [r for r in all_results if r.get("success", False)]
        if successful:
            max_seq = max(r["seq_len"] for r in successful)
            print(f"\nMaximum successful sequence length: {max_seq:,} tokens")

            if args.compare:
                # Compare Hilbert vs Standard
                hilbert_results = [r for r in successful if "Hilbert" in r["model"]]
                standard_results = [r for r in successful if "Standard" in r["model"]]

                if hilbert_results and standard_results:
                    print("\nHilbert vs Standard speedups:")
                    for h in hilbert_results:
                        s = next(
                            (
                                s
                                for s in standard_results
                                if s["seq_len"] == h["seq_len"]
                            ),
                            None,
                        )
                        if s:
                            speedup = s["avg_time_ms"] / h["avg_time_ms"]
                            print(f"  {h['seq_len']:,} tokens: {speedup:.2f}x speedup")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
