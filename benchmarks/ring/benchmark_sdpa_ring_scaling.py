#!/usr/bin/env python3
"""Benchmark RingDilatedAttentionSDPA with multi-GPU scaling."""

import torch
import torch.distributed as dist
import os
import sys
import gc
import time
import json
from pathlib import Path
from typing import Dict

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch import RingDilatedAttentionSDPA  # noqa: E402
from dilated_attention_pytorch.utils import get_optimal_dtype  # noqa: E402

# Apply communication fixes for SDPA variant
from dilated_attention_pytorch.ring.utils import ring_communication_fix  # noqa: E402
from dilated_attention_pytorch.ring.base import ring_dilated_attention_sdpa  # noqa: E402

ring_dilated_attention_sdpa.ring_pass_kv_safe = (
    ring_communication_fix.ring_pass_kv_fixed
)


def benchmark_sequence_length(
    model: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_iters: int = 5,
) -> Dict:
    """Benchmark a specific sequence length."""
    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(x)

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**2)

    # Time multiple iterations
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iters):
        with torch.no_grad():
            output = model(x)

    torch.cuda.synchronize()
    end_time = time.time()

    # Get stats
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    mem_used = peak_mem - mem_before
    avg_time = (end_time - start_time) / num_iters
    throughput = (batch_size * seq_len) / avg_time

    # Calculate effective sequence per GPU
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    effective_seq = seq_len // world_size

    return {
        "time_per_iter": avg_time,
        "throughput": throughput,
        "peak_memory_mb": peak_mem,
        "memory_used_mb": mem_used,
        "memory_per_token_kb": mem_used / (batch_size * seq_len) * 1024,
        "effective_seq_per_gpu": effective_seq,
        "output_shape": list(output.shape),
    }


def main():
    # Configuration
    embed_dim = 512
    num_heads = 8
    batch_size = 2

    # Test configurations - including larger sequences
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates)
        (2048, [512, 1024], [1, 2]),
        (4096, [512, 1024, 2048], [1, 2, 4]),
        (8192, [1024, 2048, 4096], [1, 2, 4]),
        (16384, [2048, 4096, 8192], [1, 2, 4]),
        (32768, [4096, 8192, 16384], [1, 2, 4]),
        (65536, [8192, 16384, 32768], [1, 2, 4]),
    ]

    # Initialize distributed if available
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dtype = get_optimal_dtype(device)

    if rank == 0:
        print(f"\n{'=' * 80}")
        print("RingDilatedAttentionSDPA Multi-GPU Scaling Benchmark")
        print(f"{'=' * 80}")
        print(f"World size: {world_size}")
        print(f"Device: {device}, Dtype: {dtype}")
        print(f"Batch size: {batch_size}, Embed dim: {embed_dim}, Heads: {num_heads}")

    results = []

    for seq_len, segment_lengths, dilation_rates in test_configs:
        # Skip if not divisible
        if seq_len % max(segment_lengths) != 0:
            continue

        # Skip very large sequences on single GPU
        if world_size == 1 and seq_len > 16384:
            if rank == 0:
                print(f"\nSkipping seq_len={seq_len} on single GPU (too large)")
            continue

        if rank == 0:
            print(f"\n{'-' * 60}")
            print(f"Testing sequence length: {seq_len}")
            print(f"Segments: {segment_lengths}, Dilations: {dilation_rates}")
            if world_size > 1:
                print(f"Effective sequence per GPU: {seq_len // world_size}")

        try:
            # Create model
            model = RingDilatedAttentionSDPA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=dtype,
            )
            model.eval()

            # Benchmark
            metrics = benchmark_sequence_length(
                model, seq_len, batch_size, embed_dim, device, dtype
            )

            # Add metadata
            metrics.update(
                {
                    "sequence_length": seq_len,
                    "world_size": world_size,
                    "rank": rank,
                    "batch_size": batch_size,
                    "segment_lengths": segment_lengths,
                    "dilation_rates": dilation_rates,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                }
            )

            # Calculate sparsity
            total_positions = 0
            for seg_len, dilation in zip(segment_lengths, dilation_rates):
                if seg_len <= seq_len:
                    actual_positions = min(seg_len, seq_len // dilation)
                    total_positions += actual_positions
            sparsity = 1.0 - (total_positions / seq_len)
            metrics["sparsity"] = sparsity

            results.append(metrics)

            if rank == 0:
                print("\nResults:")
                print(f"  Time per iteration: {metrics['time_per_iter']:.3f}s")
                print(f"  Throughput: {metrics['throughput']:,.0f} tokens/sec")
                print(f"  Memory used: {metrics['memory_used_mb']:.2f} MB")
                print(f"  Memory per token: {metrics['memory_per_token_kb']:.3f} KB")
                print(f"  Sparsity: {sparsity:.1%}")

                if world_size > 1:
                    print(f"  Memory scaling: O(n/{world_size})")
                    theoretical_mem_reduction = world_size
                    _ = metrics["memory_used_mb"]
                    print(f"  Theoretical reduction: {theoretical_mem_reduction}x")

            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            if rank == 0:
                print(f"Error with seq_len={seq_len}: {e}")
            import traceback

            if rank == 0:
                traceback.print_exc()

    # Synchronize before saving results
    if dist.is_initialized():
        dist.barrier()

    # Save results
    if rank == 0:
        output_dir = Path("benchmarks/results/ring/sdpa_scaling")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"sdpa_scaling_w{world_size}_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"Results saved to: {output_file}")

        # Print summary
        print(f"\nSummary (World Size: {world_size}):")
        print(
            f"{'Seq Len':<10} {'Time(s)':<10} {'Memory(MB)':<12} {'Throughput':<15} {'KB/Token':<10}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"{r['sequence_length']:<10} "
                f"{r['time_per_iter']:<10.3f} "
                f"{r['memory_used_mb']:<12.2f} "
                f"{r['throughput']:<15,.0f} "
                f"{r['memory_per_token_kb']:<10.3f}"
            )

        # Calculate scaling efficiency
        if world_size > 1 and results:
            print("\nMemory Scaling Efficiency:")
            avg_kb_per_token = sum(r["memory_per_token_kb"] for r in results) / len(
                results
            )
            print(f"  Average memory per token: {avg_kb_per_token:.3f} KB")
            print(f"  Expected with perfect O(n/{world_size}) scaling")

            # Show max sequence achieved
            max_seq = max(r["sequence_length"] for r in results)
            print(f"\nMaximum sequence length tested: {max_seq:,} tokens")
            if max_seq >= 1_000_000:
                print("  ✓ Successfully scaled to 1M+ tokens!")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
