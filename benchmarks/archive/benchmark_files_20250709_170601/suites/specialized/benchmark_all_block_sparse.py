#!/usr/bin/env python3
"""
Comprehensive benchmark of all block-sparse implementations.
Tests single-GPU and multi-GPU performance.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    create_multihead_block_sparse,
    SparsePatternConfig,
    DistributedSparseConfig,
)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    implementation: str
    sequence_length: int
    batch_size: int
    forward_time_ms: float
    backward_time_ms: Optional[float]
    memory_mb: float
    max_sequence: Optional[int]
    multi_gpu: bool
    num_gpus: int
    error: Optional[str] = None


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_single_gpu(
    impl_name: str,
    model_fn,
    seq_lengths: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup: int = 2,
    iterations: int = 5,
) -> List[BenchmarkResult]:
    """Benchmark implementation on single GPU."""
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{impl_name} - Single GPU:")
    print("-" * 60)

    max_sequence = 0

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model
            model = model_fn(seq_len)
            if hasattr(model, "to"):
                model = model.to(device)

            # Create inputs
            if impl_name == "Multihead":
                # Multihead expects (batch, seq, embed_dim)
                embed_dim = num_heads * head_dim
                q = torch.randn(
                    batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
            else:
                # Others expect (batch, seq, heads, dim) including Adaptive with wrapper
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=device,
                    dtype=torch.float16,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)

            # Measure memory after allocation
            mem_before = get_gpu_memory()

            # Warmup
            for _ in range(warmup):
                with torch.amp.autocast("cuda"):
                    _ = model(q, k, v)
                torch.cuda.synchronize()

            # Time forward pass
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                with torch.amp.autocast("cuda"):
                    output = model(q, k, v)
                torch.cuda.synchronize()
            forward_time = (time.perf_counter() - start) / iterations * 1000

            # Skip backward pass for now due to gradient issues
            backward_time = None

            # Memory usage
            mem_after = get_gpu_memory()
            memory_used = mem_after - mem_before

            print(
                f"  {seq_len:6d} tokens: {forward_time:6.1f}ms forward, {memory_used:6.1f}MB"
            )

            results.append(
                BenchmarkResult(
                    implementation=impl_name,
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    forward_time_ms=forward_time,
                    backward_time_ms=backward_time,
                    memory_mb=memory_used,
                    max_sequence=None,
                    multi_gpu=False,
                    num_gpus=1,
                )
            )

            max_sequence = seq_len

            # Cleanup
            del model, q, k, v, output
            clear_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  {seq_len:6d} tokens: OOM")
                break
            else:
                print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")
                results.append(
                    BenchmarkResult(
                        implementation=impl_name,
                        sequence_length=seq_len,
                        batch_size=batch_size,
                        forward_time_ms=0,
                        backward_time_ms=0,
                        memory_mb=0,
                        max_sequence=max_sequence,
                        multi_gpu=False,
                        num_gpus=1,
                        error=str(e),
                    )
                )
                break

    # Add max sequence info
    if results and max_sequence > 0:
        results[-1].max_sequence = max_sequence

    return results


def benchmark_data_parallel(
    impl_name: str,
    model_fn,
    seq_lengths: List[int],
    batch_size: int = 2,
    num_heads: int = 8,
    head_dim: int = 64,
) -> List[BenchmarkResult]:
    """Benchmark implementation with DataParallel."""
    if torch.cuda.device_count() < 2:
        print(f"\n{impl_name} - DataParallel: Skipped (need 2+ GPUs)")
        return []

    results = []
    print(f"\n{impl_name} - DataParallel ({torch.cuda.device_count()} GPUs):")
    print("-" * 60)

    max_sequence = 0

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model and wrap in DataParallel
            model = model_fn(seq_len)
            model = nn.DataParallel(model)
            model = model.cuda()

            # Create inputs
            if impl_name == "Multihead":
                embed_dim = num_heads * head_dim
                q = torch.randn(
                    batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
            else:
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.float16,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)

            # Time forward pass
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(3):
                with torch.amp.autocast("cuda"):
                    output = model(q, k, v)
                torch.cuda.synchronize()
            forward_time = (time.perf_counter() - start) / 3 * 1000

            # Memory per GPU
            mem_gpus = []
            for i in range(torch.cuda.device_count()):
                mem_gpus.append(torch.cuda.memory_allocated(i) / 1024**2)

            print(
                f"  {seq_len:6d} tokens: {forward_time:6.1f}ms, "
                f"Memory: {' / '.join(f'GPU{i}:{m:.0f}MB' for i, m in enumerate(mem_gpus))}"
            )

            results.append(
                BenchmarkResult(
                    implementation=impl_name,
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    forward_time_ms=forward_time,
                    backward_time_ms=None,
                    memory_mb=sum(mem_gpus),
                    max_sequence=None,
                    multi_gpu=True,
                    num_gpus=torch.cuda.device_count(),
                )
            )

            max_sequence = seq_len

            # Cleanup
            del model, q, k, v, output
            clear_gpu_memory()

        except Exception as e:
            if "out of memory" in str(e).lower():
                print(f"  {seq_len:6d} tokens: OOM")
                break
            else:
                print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")
                break

    if results and max_sequence > 0:
        results[-1].max_sequence = max_sequence

    return results


def benchmark_distributed_impl(rank, world_size, seq_lengths, results_queue):
    """Benchmark distributed implementation."""
    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    if rank == 0:
        print("\nDistributed Implementation:")
        print("-" * 60)

    batch_size = 1
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create distributed model
            config = DistributedSparseConfig(
                sparsity_ratio=0.05,
                pattern_type="dilated_sparse",
                enable_memory_optimization=True,
            )

            model = create_block_sparse_attention(
                variant="distributed",
                segment_lengths=[2048] if seq_len <= 8192 else [4096],
                dilation_rates=[1],
                embed_dim=embed_dim,
                num_heads=num_heads,
                distributed_config=config,
            ).cuda()

            # Create inputs
            q = torch.randn(
                batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Warmup and sync
            for _ in range(2):
                _ = model(q, k, v)
            torch.cuda.synchronize()
            dist.barrier()

            # Time forward pass
            start = time.perf_counter()
            for _ in range(3):
                output = model(q, k, v)
                torch.cuda.synchronize()
            dist.barrier()
            forward_time = (time.perf_counter() - start) / 3 * 1000

            # Memory usage
            mem_used = torch.cuda.memory_allocated() / 1024**2

            if rank == 0:
                print(
                    f"  {seq_len:6d} tokens: {forward_time:6.1f}ms, {mem_used:.1f}MB per GPU"
                )
                results_queue.put(
                    {
                        "seq_len": seq_len,
                        "forward_time": forward_time,
                        "memory": mem_used,
                        "success": True,
                    }
                )

            # Cleanup
            del model, q, k, v, output
            clear_gpu_memory()

        except Exception as e:
            if rank == 0:
                print(f"  {seq_len:6d} tokens: Error - {str(e)[:50]}")
                results_queue.put(
                    {
                        "seq_len": seq_len,
                        "error": str(e),
                        "success": False,
                    }
                )
            break

    dist.destroy_process_group()


def create_model_functions():
    """Create model factory functions for each implementation."""

    def base_model(seq_len):
        # Adaptive segment lengths for better memory usage
        if seq_len <= 8192:
            segment_lengths = [2048]
        elif seq_len <= 16384:
            segment_lengths = [4096]
        elif seq_len <= 32768:
            segment_lengths = [8192]
        else:
            segment_lengths = [16384]

        return create_block_sparse_attention(
            variant="base",
            segment_lengths=segment_lengths,
            dilation_rates=[1],
            sparse_config=SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.05,  # 95% sparse
                block_size=64,
            ),
        )

    def multihead_model(seq_len):
        return create_multihead_block_sparse(
            embed_dim=512,
            num_heads=8,
            sparsity_ratio=0.05,
            segment_lengths=[2048] if seq_len <= 8192 else [4096],
            dilation_rates=[1],
            dtype=torch.float16,  # Specify dtype
        )

    def adaptive_model(seq_len):
        # Use the fixed wrapper version
        from dilated_attention_pytorch.block_sparse_adaptive_fixed import (
            BlockSparseAdaptive,
        )

        return BlockSparseAdaptive(
            segment_lengths=[2048] if seq_len <= 8192 else [4096],
            dilation_rates=[1],
            num_heads=8,
            head_dim=64,
        )

    return {
        "Base": base_model,
        "Multihead": multihead_model,
        "Adaptive": adaptive_model,
    }


def main():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("Block-Sparse Implementation Benchmarks")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test configurations
    seq_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # Get model functions
    model_fns = create_model_functions()

    all_results = []

    # 1. Single GPU benchmarks
    print("\n" + "=" * 80)
    print("SINGLE GPU BENCHMARKS")
    print("=" * 80)

    for impl_name, model_fn in model_fns.items():
        results = benchmark_single_gpu(impl_name, model_fn, seq_lengths)
        all_results.extend(results)

    # 2. DataParallel benchmarks
    if torch.cuda.device_count() >= 2:
        print("\n" + "=" * 80)
        print("DATAPARALLEL BENCHMARKS")
        print("=" * 80)

        for impl_name, model_fn in model_fns.items():
            results = benchmark_data_parallel(
                impl_name, model_fn, seq_lengths, batch_size=2
            )
            all_results.extend(results)

    # 3. Distributed implementation
    if torch.cuda.device_count() >= 2:
        print("\n" + "=" * 80)
        print("DISTRIBUTED IMPLEMENTATION")
        print("=" * 80)

        world_size = min(torch.cuda.device_count(), 2)
        results_queue = mp.Queue()

        mp.spawn(
            benchmark_distributed_impl,
            args=(
                world_size,
                seq_lengths[:5],
                results_queue,
            ),  # Limit sequences for distributed
            nprocs=world_size,
            join=True,
        )

        # Collect distributed results
        dist_results = []
        while not results_queue.empty():
            result = results_queue.get()
            if result["success"]:
                dist_results.append(
                    BenchmarkResult(
                        implementation="Distributed",
                        sequence_length=result["seq_len"],
                        batch_size=1,
                        forward_time_ms=result["forward_time"],
                        backward_time_ms=None,
                        memory_mb=result["memory"],
                        max_sequence=None,
                        multi_gpu=True,
                        num_gpus=world_size,
                    )
                )
        all_results.extend(dist_results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find max sequences
    max_sequences = {}
    for result in all_results:
        key = f"{result.implementation}-{'Multi' if result.multi_gpu else 'Single'}"
        if result.error is None:
            max_sequences[key] = result.sequence_length

    print("\nMaximum Sequence Lengths:")
    for key, max_seq in sorted(max_sequences.items()):
        print(f"  {key:25s}: {max_seq:,} tokens")

    # Performance at 8K tokens
    print("\nPerformance at 8,192 tokens:")
    perf_8k = [r for r in all_results if r.sequence_length == 8192 and r.error is None]
    for result in sorted(perf_8k, key=lambda x: x.forward_time_ms):
        mode = "Multi-GPU" if result.multi_gpu else "Single-GPU"
        print(
            f"  {result.implementation:15s} ({mode:10s}): {result.forward_time_ms:6.1f}ms"
        )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"block_sparse_benchmark_results_{timestamp}.json"

    results_dict = []
    for r in all_results:
        results_dict.append(
            {
                "implementation": r.implementation,
                "sequence_length": r.sequence_length,
                "batch_size": r.batch_size,
                "forward_time_ms": r.forward_time_ms,
                "backward_time_ms": r.backward_time_ms,
                "memory_mb": r.memory_mb,
                "max_sequence": r.max_sequence,
                "multi_gpu": r.multi_gpu,
                "num_gpus": r.num_gpus,
                "error": r.error,
            }
        )

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_names": [
                        torch.cuda.get_device_name(i)
                        for i in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else [],
                },
                "results": results_dict,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
