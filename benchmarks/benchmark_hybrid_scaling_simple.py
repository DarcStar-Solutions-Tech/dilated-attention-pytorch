#!/usr/bin/env python3
"""
Simple scaling benchmark for Hybrid Ring Attention without all_gather.
Each rank reports its own metrics independently.

Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_scaling_simple.py
"""

import os
import gc
import time
import json
import torch
import torch.distributed as dist
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)


def test_sequence(seq_len: int, rank: int, world_size: int, device: torch.device):
    """Test a single sequence length and return metrics."""

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        # Create model
        model = RingDilatedAttentionHybrid(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            dropout=0.0,
            ring_size=world_size,
            device=device,
            # Auto dtype selection
            enable_memory_pool=True,
            use_pattern_cache=True,
            use_flash_attention=False,
        )

        dtype = model.dtype
        _ = 4 if dtype == torch.float32 else 2

        # Memory after model
        mem_model_mb = torch.cuda.memory_allocated(device) / (1024**2)

        # Create inputs
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * 0.1
        )
        k = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * 0.1
        )
        v = (
            torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            * 0.1
        )

        # Memory after inputs
        mem_inputs_mb = torch.cuda.memory_allocated(device) / (1024**2)
        input_memory_mb = mem_inputs_mb - mem_model_mb

        # Warmup
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)
        torch.cuda.synchronize()

        # Reset peak memory after warmup
        torch.cuda.reset_peak_memory_stats()

        # Timed run
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v, is_causal=False)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Memory stats
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        # Memory per token
        total_tokens = batch_size * seq_len
        memory_per_token_kb = (peak_memory_mb * 1024) / total_tokens

        # Clean up
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "success": True,
            "seq_len": seq_len,
            "rank": rank,
            "time_ms": elapsed * 1000,
            "model_memory_mb": mem_model_mb,
            "input_memory_mb": input_memory_mb,
            "peak_memory_mb": peak_memory_mb,
            "memory_per_token_kb": memory_per_token_kb,
            "throughput_tokens_sec": total_tokens / elapsed,
            "dtype": str(dtype),
        }

    except torch.cuda.OutOfMemoryError:
        return {
            "success": False,
            "seq_len": seq_len,
            "rank": rank,
            "error": "OOM",
        }
    except Exception as e:
        return {
            "success": False,
            "seq_len": seq_len,
            "rank": rank,
            "error": str(type(e).__name__),
            "detail": str(e),
        }


def main():
    # Initialize distributed
    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 benchmarks/benchmark_hybrid_scaling_simple.py"
        )
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Hybrid Ring Attention - Scaling Benchmark (Simple)")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
        print()

    # Test sequences - start small and grow
    test_sequences = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    results = []

    for seq_len in test_sequences:
        # Synchronize before each test
        dist.barrier()

        # Each rank tests independently
        result = test_sequence(seq_len, rank, world_size, device)
        results.append(result)

        # Only rank 0 prints progress
        if rank == 0:
            if result["success"]:
                print(
                    f"Seq {seq_len:6d}: {result['time_ms']:6.1f}ms, "
                    f"{result['peak_memory_mb']:7.1f}MB peak, "
                    f"{result['memory_per_token_kb']:5.2f}KB/token"
                )
            else:
                print(f"Seq {seq_len:6d}: Failed - {result['error']}")

        # Stop if failed
        if not result["success"]:
            break

        # Small delay between tests
        time.sleep(0.5)

    # Save results independently for each rank
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"benchmarks/hybrid_scaling_rank{rank}_{world_size}gpu_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "rank": rank,
                "world_size": world_size,
                "gpu_name": torch.cuda.get_device_name(device),
                "compute_capability": str(torch.cuda.get_device_capability(device)),
                "results": results,
            },
            f,
            indent=2,
        )

    # Summary from rank 0
    if rank == 0:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        successful = [r for r in results if r["success"]]
        if successful:
            max_seq = max(r["seq_len"] for r in successful)
            print(f"\nMaximum successful sequence: {max_seq:,} tokens")

            print("\nScaling Analysis:")
            print(
                f"{'Seq Len':<10} {'Time (ms)':<10} {'Memory (MB)':<12} {'KB/token':<10} {'Speedup':<10}"
            )
            print("-" * 60)

            base_time = successful[0]["time_ms"]
            for r in successful:
                speedup = (
                    base_time * successful[0]["seq_len"] / (r["time_ms"] * r["seq_len"])
                )
                print(
                    f"{r['seq_len']:<10} {r['time_ms']:<10.1f} {r['peak_memory_mb']:<12.1f} "
                    f"{r['memory_per_token_kb']:<10.2f} {speedup:<10.2f}"
                )

            # Memory scaling
            if len(successful) > 1:
                first = successful[0]
                last = successful[-1]
                seq_growth = last["seq_len"] / first["seq_len"]
                mem_growth = last["peak_memory_mb"] / first["peak_memory_mb"]

                print(
                    f"\nMemory Scaling ({first['seq_len']} → {last['seq_len']} tokens):"
                )
                print(f"  Sequence growth: {seq_growth:.1f}x")
                print(f"  Memory growth: {mem_growth:.1f}x")
                print(f"  Efficiency: {seq_growth / mem_growth:.2f} (ideal: 1.0)")

                # Average memory per token
                avg_kb_per_token = sum(
                    r["memory_per_token_kb"] for r in successful
                ) / len(successful)
                print(f"\nAverage memory per token: {avg_kb_per_token:.2f} KB")
                print(
                    f"With {world_size} GPUs, each GPU handles ~{100 / world_size:.0f}% of K,V tensors"
                )

        print(f"\nResults saved to: {filename} (and rank1 file)")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
