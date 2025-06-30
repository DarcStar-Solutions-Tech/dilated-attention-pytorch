"""
Distributed benchmarking for Ring Dilated Attention across multiple GPUs.

This script properly initializes distributed PyTorch to benchmark
Ring Attention implementations with actual multi-GPU communication.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse
from typing import List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3

# Try to import production version
try:
    from dilated_attention_pytorch import RingDilatedAttentionProduction

    RING_PRODUCTION_AVAILABLE = True
except ImportError:
    RING_PRODUCTION_AVAILABLE = False


@dataclass
class DistributedBenchmarkResult:
    """Result for distributed benchmark."""

    implementation: str
    sequence_length: int
    batch_size: int
    num_heads: int
    head_dim: int
    world_size: int
    mode: str  # single, simulated, distributed
    time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    comm_time_ms: float = 0.0
    pattern_cache_enabled: bool = False
    memory_pool_enabled: bool = False


def init_process(rank, world_size, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set CUDA device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def benchmark_single_config(
    rank: int,
    world_size: int,
    implementation: str,
    seq_len: int,
    batch_size: int,
    num_heads: int = 8,
    head_dim: int = 64,
    enable_pattern_cache: bool = False,
    enable_memory_pool: bool = False,
    warmup_steps: int = 3,
    benchmark_steps: int = 10,
) -> Optional[DistributedBenchmarkResult]:
    """Benchmark a single configuration on one GPU."""

    device = torch.device(f"cuda:{rank}")

    # Get appropriate segment configuration
    if seq_len <= 8192:
        segment_lengths = [1024, 2048, 4096]
        dilation_rates = [1, 2, 4]
    elif seq_len <= 32768:
        segment_lengths = [2048, 4096, 8192]
        dilation_rates = [1, 2, 4]
    else:
        segment_lengths = [4096, 8192, 16384]
        dilation_rates = [1, 2, 4]

    # Ensure divisibility
    max_segment = max(segment_lengths)
    seq_len = (seq_len // max_segment) * max_segment

    try:
        # Create model
        kwargs = {
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
            "ring_size": world_size,
            "device": device,
            "dtype": torch.float16,
            "enable_memory_pool": enable_memory_pool,
            "use_pattern_cache": enable_pattern_cache,
        }

        if implementation == "ring_v2":
            model = RingDilatedAttentionV2(**kwargs)
        elif implementation == "ring_v3":
            model = RingDilatedAttentionV3(**kwargs)
        elif implementation == "ring_production" and RING_PRODUCTION_AVAILABLE:
            model = RingDilatedAttentionProduction(**kwargs)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Get mode for reporting
        mode = model.mode

        # Create input tensors (each GPU gets its portion)
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(warmup_steps):
            _ = model(q, k, v)

        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device) / 1024 / 1024

        # Time computation
        start_time = time.time()
        comm_time = 0.0

        for _ in range(benchmark_steps):
            if world_size > 1:
                comm_start = time.time()
                dist.barrier()
                comm_time += time.time() - comm_start

            _ = model(q, k, v)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        total_time = end_time - start_time
        time_ms = (total_time * 1000) / benchmark_steps
        comm_time_ms = (comm_time * 1000) / benchmark_steps

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        memory_mb = peak_memory - start_memory

        total_tokens = batch_size * seq_len * world_size  # Total across all GPUs
        throughput = total_tokens / (time_ms / 1000)

        result = DistributedBenchmarkResult(
            implementation=implementation,
            sequence_length=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            world_size=world_size,
            mode=mode,
            time_ms=time_ms,
            memory_mb=memory_mb,
            throughput_tokens_per_sec=throughput,
            comm_time_ms=comm_time_ms,
            pattern_cache_enabled=enable_pattern_cache,
            memory_pool_enabled=enable_memory_pool,
        )

        # Only rank 0 returns the result
        if rank == 0:
            return result
        else:
            return None

    except Exception as e:
        if rank == 0:
            print(f"Error benchmarking {implementation}: {e}")
        return None
    finally:
        # Cleanup
        if "model" in locals():
            del model
        if "q" in locals():
            del q, k, v
        torch.cuda.empty_cache()


def run_distributed_benchmark(
    rank: int,
    world_size: int,
    implementations: List[str],
    sequence_lengths: List[int],
    batch_sizes: List[int],
    enable_pattern_cache: bool = False,
    enable_memory_pool: bool = False,
    output_file: Optional[str] = None,
):
    """Run distributed benchmark on a single process."""

    # Initialize distributed
    init_process(rank, world_size)

    results = []

    try:
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                for impl in implementations:
                    if rank == 0:
                        print(
                            f"\nBenchmarking {impl} - seq_len={seq_len}, batch={batch_size}"
                        )

                    # Benchmark without optimizations
                    result = benchmark_single_config(
                        rank,
                        world_size,
                        impl,
                        seq_len,
                        batch_size,
                        enable_pattern_cache=False,
                        enable_memory_pool=False,
                    )
                    if result:
                        results.append(result)
                        print(
                            f"  Baseline: {result.time_ms:.1f}ms, {result.memory_mb:.1f}MB, mode={result.mode}"
                        )

                    # Benchmark with pattern cache
                    result = benchmark_single_config(
                        rank,
                        world_size,
                        impl,
                        seq_len,
                        batch_size,
                        enable_pattern_cache=True,
                        enable_memory_pool=False,
                    )
                    if result:
                        results.append(result)
                        print(
                            f"  With cache: {result.time_ms:.1f}ms, {result.memory_mb:.1f}MB"
                        )

                    # Benchmark with both optimizations
                    result = benchmark_single_config(
                        rank,
                        world_size,
                        impl,
                        seq_len,
                        batch_size,
                        enable_pattern_cache=True,
                        enable_memory_pool=True,
                    )
                    if result:
                        results.append(result)
                        print(
                            f"  With both: {result.time_ms:.1f}ms, {result.memory_mb:.1f}MB"
                        )

        # Save results from rank 0
        if rank == 0 and output_file and results:
            save_results(results, output_file)

    finally:
        cleanup()


def save_results(results: List[DistributedBenchmarkResult], output_file: str):
    """Save benchmark results to file."""

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        json_results.append(
            {
                "implementation": r.implementation,
                "sequence_length": r.sequence_length,
                "batch_size": r.batch_size,
                "num_heads": r.num_heads,
                "head_dim": r.head_dim,
                "world_size": r.world_size,
                "mode": r.mode,
                "time_ms": r.time_ms,
                "memory_mb": r.memory_mb,
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "comm_time_ms": r.comm_time_ms,
                "pattern_cache_enabled": r.pattern_cache_enabled,
                "memory_pool_enabled": r.memory_pool_enabled,
            }
        )

    # Save JSON
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "results": json_results,
            },
            f,
            indent=2,
        )

    # Generate summary report
    generate_summary_report(results, output_file.replace(".json", "_summary.md"))


def generate_summary_report(
    results: List[DistributedBenchmarkResult], output_file: str
):
    """Generate a markdown summary report."""

    lines = []
    lines.append("# Distributed Ring Attention Benchmark Report")
    lines.append(f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"\nWorld Size: {results[0].world_size if results else 'Unknown'}")

    # Group by implementation
    impl_results = {}
    for r in results:
        key = (r.implementation, r.pattern_cache_enabled, r.memory_pool_enabled)
        if key not in impl_results:
            impl_results[key] = []
        impl_results[key].append(r)

    # Compare implementations
    lines.append("\n## Performance Comparison")
    lines.append(
        "\n| Implementation | Optimizations | Seq Len | Batch | Time (ms) | Memory (MB) | Throughput (tok/s) | Mode |"
    )
    lines.append(
        "|----------------|---------------|---------|-------|-----------|-------------|-------------------|------|"
    )

    for (impl, cache, pool), results_list in sorted(impl_results.items()):
        opt_str = []
        if cache:
            opt_str.append("cache")
        if pool:
            opt_str.append("pool")
        opt_str = "+".join(opt_str) if opt_str else "none"

        for r in results_list:
            lines.append(
                f"| {r.implementation} | {opt_str} | {r.sequence_length:,} | {r.batch_size} | "
                f"{r.time_ms:.1f} | {r.memory_mb:.1f} | {r.throughput_tokens_per_sec:,.0f} | {r.mode} |"
            )

    # V2 vs V3 comparison
    v2_results = [r for r in results if r.implementation == "ring_v2"]
    v3_results = [r for r in results if r.implementation == "ring_v3"]

    if v2_results and v3_results:
        lines.append("\n## V2 vs V3 Performance")
        lines.append("\n### Relative Performance (V3 vs V2)")

        for v2 in v2_results:
            # Find matching v3 result
            v3_match = next(
                (
                    v3
                    for v3 in v3_results
                    if v3.sequence_length == v2.sequence_length
                    and v3.batch_size == v2.batch_size
                    and v3.pattern_cache_enabled == v2.pattern_cache_enabled
                    and v3.memory_pool_enabled == v2.memory_pool_enabled
                ),
                None,
            )

            if v3_match:
                speedup = v2.time_ms / v3_match.time_ms
                opt_str = []
                if v2.pattern_cache_enabled:
                    opt_str.append("cache")
                if v2.memory_pool_enabled:
                    opt_str.append("pool")
                opt_str = "+".join(opt_str) if opt_str else "baseline"

                lines.append(
                    f"- Seq {v2.sequence_length:,}, Batch {v2.batch_size}, {opt_str}: "
                    f"V3 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than V2"
                )

    # Communication overhead analysis
    lines.append("\n## Communication Overhead")
    for r in results:
        if r.comm_time_ms > 0:
            comm_percent = (r.comm_time_ms / r.time_ms) * 100
            lines.append(
                f"- {r.implementation} (seq={r.sequence_length:,}): "
                f"{r.comm_time_ms:.1f}ms ({comm_percent:.1f}% of total time)"
            )

    # Save report
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Distributed Ring Attention Benchmark")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["ring_v2", "ring_v3"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1, 2], help="Batch sizes to test"
    )
    parser.add_argument(
        "--world-size", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--output-file",
        default="benchmark_results/distributed_ring_benchmark.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Spawn processes for distributed training
    mp.spawn(
        run_distributed_benchmark,
        args=(
            args.world_size,
            args.implementations,
            args.sequence_lengths,
            args.batch_sizes,
            True,  # enable_pattern_cache
            True,  # enable_memory_pool
            args.output_file,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
