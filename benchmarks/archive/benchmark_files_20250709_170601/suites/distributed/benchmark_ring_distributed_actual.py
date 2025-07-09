#!/usr/bin/env python3
"""
Actual distributed multi-GPU benchmark for ring attention.

This benchmark runs on multiple GPUs using PyTorch distributed.
Launch with: torchrun --nproc_per_node=<num_gpus> benchmark_ring_distributed_actual.py
"""

import torch
import torch.distributed as dist
import gc
import os
import sys
import time
import json
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
    RingDilatedAttentionHilbertCore,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info, get_optimal_dtype


@dataclass
class DistributedBenchmarkResult:
    """Store distributed benchmark results."""

    implementation: str
    world_size: int
    rank: int
    total_seq_len: int
    local_seq_len: int
    batch_size: int
    memory_mb: float
    forward_time_ms: float
    tokens_per_second_local: float
    tokens_per_second_global: float
    success: bool
    error: Optional[str] = None


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def benchmark_distributed_config(
    model_class: type,
    model_name: str,
    total_seq_len: int,
    batch_size: int,
    embed_dim: int = 768,
    num_heads: int = 12,
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    warmup_steps: int = 3,
    measure_steps: int = 10,
) -> DistributedBenchmarkResult:
    """Benchmark a configuration in distributed setting."""
    if segment_lengths is None:
        segment_lengths = [4096, 8192, 16384]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Ensure all processes use same dtype
    _ = get_gpu_info(device)
    optimal_dtype = get_optimal_dtype(device)

    local_seq_len = total_seq_len // world_size

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated() / 1024 / 1024

    try:
        # Create local input
        x_local = torch.randn(
            batch_size, local_seq_len, embed_dim, device=device, dtype=optimal_dtype
        )

        # Synchronize before model creation
        if dist.is_initialized():
            dist.barrier()

        # Create model
        if model_name == "HilbertCore":
            model = model_class(
                dim=embed_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=world_size,
                use_hilbert=True,
                use_custom_backward=True,
            )
        else:
            model = model_class(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                use_hilbert=True,
                device=device,
                dtype=optimal_dtype,
                memory_efficient=True,
            )

        model = model.to(device)
        model.eval()

        # Synchronize before warmup
        if dist.is_initialized():
            dist.barrier()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(x_local, total_seq_len=total_seq_len, already_split=True)

        # Synchronize before measurement
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Measure
        forward_times = []
        with torch.no_grad():
            for _ in range(measure_steps):
                start_time = time.perf_counter()

                output = model(x_local, total_seq_len=total_seq_len, already_split=True)

                torch.cuda.synchronize()
                forward_times.append(time.perf_counter() - start_time)

        # Calculate metrics
        avg_forward_time = sum(forward_times) / len(forward_times)
        forward_time_ms = avg_forward_time * 1000
        tokens_per_second_local = (batch_size * local_seq_len) / avg_forward_time
        tokens_per_second_global = (batch_size * total_seq_len) / avg_forward_time

        # Memory usage
        peak_mem = torch.cuda.memory_allocated() / 1024 / 1024
        memory_mb = peak_mem - start_mem

        # Cleanup
        del x_local, model, output

        result = DistributedBenchmarkResult(
            implementation=model_name,
            world_size=world_size,
            rank=rank,
            total_seq_len=total_seq_len,
            local_seq_len=local_seq_len,
            batch_size=batch_size,
            memory_mb=memory_mb,
            forward_time_ms=forward_time_ms,
            tokens_per_second_local=tokens_per_second_local,
            tokens_per_second_global=tokens_per_second_global,
            success=True,
        )

    except torch.cuda.OutOfMemoryError:
        result = DistributedBenchmarkResult(
            implementation=model_name,
            world_size=world_size,
            rank=rank,
            total_seq_len=total_seq_len,
            local_seq_len=local_seq_len,
            batch_size=batch_size,
            memory_mb=float("inf"),
            forward_time_ms=float("inf"),
            tokens_per_second_local=0,
            tokens_per_second_global=0,
            success=False,
            error="OOM",
        )
    except Exception as e:
        result = DistributedBenchmarkResult(
            implementation=model_name,
            world_size=world_size,
            rank=rank,
            total_seq_len=total_seq_len,
            local_seq_len=local_seq_len,
            batch_size=batch_size,
            memory_mb=0,
            forward_time_ms=0,
            tokens_per_second_local=0,
            tokens_per_second_global=0,
            success=False,
            error=str(e),
        )
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def run_distributed_benchmarks():
    """Run distributed benchmarks."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Only rank 0 prints headers
    if rank == 0:
        gpu_info = get_gpu_info(device)
        print("=" * 80)
        print("Distributed Ring Attention Benchmark")
        print("=" * 80)
        print(f"World size: {world_size}")
        print(f"GPU: {gpu_info.name} ({gpu_info.architecture})")
        print(f"Total memory per GPU: {gpu_info.total_memory_gb:.1f} GB")
        print()

    # Test configurations
    test_configs = [
        # (implementation, name)
        (RingDilatedAttentionHilbertOptimizedCorrect, "HilbertOptimizedCorrect"),
        (RingDilatedAttentionHilbertCore, "HilbertCore"),
    ]

    # Sequence lengths to test
    sequence_lengths = [
        8192,  # 8K
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        131072,  # 128K
        262144,  # 256K
        524288,  # 512K
    ]

    batch_sizes = [1, 2]

    results = []

    for impl_class, impl_name in test_configs:
        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                # Skip if local sequence would be too small
                if seq_len // world_size < 1024:
                    continue

                # Only rank 0 prints progress
                if rank == 0:
                    print(
                        f"Testing {impl_name} B={batch_size} L={seq_len:,}...",
                        end="",
                        flush=True,
                    )

                # All ranks must participate
                result = benchmark_distributed_config(
                    impl_class, impl_name, seq_len, batch_size
                )
                results.append(result)

                # Only rank 0 prints results
                if rank == 0:
                    if result.success:
                        print(
                            f" ✓ {result.memory_mb:.1f}MB, "
                            f"{result.forward_time_ms:.1f}ms, "
                            f"{result.tokens_per_second_global / 1e6:.2f}M tok/s"
                        )
                    else:
                        print(f" ✗ {result.error}")

                # Stop testing larger sequences if OOM
                if not result.success and result.error == "OOM":
                    break

    # Gather results from all ranks
    if dist.is_initialized():
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if rank == 0:
            # Flatten results
            all_results_flat = []
            for rank_results in all_results:
                all_results_flat.extend(rank_results)
            results = all_results_flat

    # Analysis on rank 0
    if rank == 0:
        analyze_distributed_results(results, world_size)
        save_distributed_results(results, world_size)

    cleanup_distributed()


def analyze_distributed_results(
    results: List[DistributedBenchmarkResult], world_size: int
):
    """Analyze distributed benchmark results."""
    print()
    print("=" * 80)
    print("Distributed Benchmark Results Summary")
    print("=" * 80)

    # Group by implementation
    implementations = list(set(r.implementation for r in results))

    for impl in implementations:
        impl_results = [r for r in results if r.implementation == impl and r.rank == 0]
        successful = [r for r in impl_results if r.success]

        print(f"\n{impl}:")
        print("-" * 40)

        if not successful:
            print("  No successful runs")
            continue

        # Max sequence length
        max_seq = max(r.total_seq_len for r in successful)
        print(f"  Max sequence length: {max_seq:,} tokens")
        print(f"  Per GPU: {max_seq // world_size:,} tokens")

        # Best performance
        best_perf = max(successful, key=lambda r: r.tokens_per_second_global)
        print(
            f"  Best throughput: {best_perf.tokens_per_second_global / 1e6:.2f}M tokens/sec"
        )
        print(f"    Config: B={best_perf.batch_size}, L={best_perf.total_seq_len:,}")

        # Memory efficiency
        avg_memory_per_token = sum(
            r.memory_mb / (r.batch_size * r.local_seq_len) for r in successful
        ) / len(successful)
        print(f"  Avg memory per token: {avg_memory_per_token:.4f} MB")

    # Check for 200K+ capability
    print()
    print("=" * 80)
    print("200K+ Token Capability")
    print("=" * 80)

    large_seq_results = [
        r for r in results if r.total_seq_len >= 200000 and r.success and r.rank == 0
    ]

    if large_seq_results:
        print("✓ Successfully processed sequences with 200K+ tokens!")
        print(f"  World size: {world_size}")
        print("  Configurations that succeeded:")

        for r in sorted(large_seq_results, key=lambda x: x.total_seq_len):
            print(
                f"    {r.implementation}: {r.total_seq_len:,} tokens "
                f"({r.local_seq_len:,} per GPU, {r.memory_mb:.1f}MB)"
            )
    else:
        print(f"✗ Could not process 200K+ tokens with {world_size} GPUs")

        # Find max achieved
        if successful:
            max_achieved = max(r.total_seq_len for r in successful)
            print(f"  Maximum achieved: {max_achieved:,} tokens")

            # Estimate GPUs needed
            if max_achieved > 0:
                gpus_needed = int(200000 * world_size / max_achieved) + 1
                print(f"  Estimated GPUs needed for 200K: {gpus_needed}")


def save_distributed_results(
    results: List[DistributedBenchmarkResult], world_size: int
):
    """Save distributed benchmark results."""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"benchmark-ring-distributed-w{world_size}-{timestamp}.json"

    # Convert results to dict
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "world_size": world_size,
        "results": [
            {
                "implementation": r.implementation,
                "world_size": r.world_size,
                "rank": r.rank,
                "total_seq_len": r.total_seq_len,
                "local_seq_len": r.local_seq_len,
                "batch_size": r.batch_size,
                "memory_mb": r.memory_mb,
                "forward_time_ms": r.forward_time_ms,
                "tokens_per_second_global": r.tokens_per_second_global,
                "success": r.success,
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(filename, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    """Main entry point."""
    try:
        run_distributed_benchmarks()
    except Exception as e:
        print(f"Error in distributed benchmark: {e}")
        import traceback

        traceback.print_exc()
        cleanup_distributed()


if __name__ == "__main__":
    main()
