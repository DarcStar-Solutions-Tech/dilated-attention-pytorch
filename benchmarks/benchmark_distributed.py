#!/usr/bin/env python3
"""
Multi-GPU benchmark for distributed dilated attention implementations.

This script tests the scalability and performance of distributed implementations
across multiple GPUs using PyTorch distributed.
"""

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dilated_attention_pytorch.improved_distributed_dilated_attention import (
        DistributedImprovedDilatedAttention,
        DistributedImprovedMultiheadDilatedAttention,
    )

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    print("Warning: Distributed implementations not available")
    DISTRIBUTED_AVAILABLE = False

try:
    from dilated_attention_pytorch.ring_distributed_dilated_attention import (
        RingDistributedDilatedAttention,
    )

    RING_DISTRIBUTED_AVAILABLE = True
except ImportError:
    print("Warning: Ring distributed implementation not available")
    RING_DISTRIBUTED_AVAILABLE = False

try:
    from dilated_attention_pytorch.block_sparse_ring_distributed_dilated_attention import (
        BlockSparseRingDistributedDilatedAttention,
    )

    BLOCK_SPARSE_DISTRIBUTED_AVAILABLE = True
except ImportError:
    print("Warning: Block sparse distributed implementation not available")
    BLOCK_SPARSE_DISTRIBUTED_AVAILABLE = False


@dataclass
class DistributedBenchmarkResult:
    """Container for distributed benchmark results."""

    implementation: str
    world_size: int
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    mean_time_ms: float
    std_time_ms: float
    peak_memory_mb_per_gpu: float
    total_memory_mb: float
    throughput_tokens_per_sec: float
    scaling_efficiency: float  # Compared to single GPU
    communication_overhead_ms: float
    success: bool
    error: str | None = None

    def __str__(self):
        if not self.success:
            return (
                f"{self.implementation} @ {self.world_size} GPUs: FAILED - {self.error}"
            )

        return (
            f"{self.implementation} @ {self.world_size} GPUs: "
            f"{self.mean_time_ms:.1f}ms, "
            f"{self.peak_memory_mb_per_gpu:.0f}MB/GPU, "
            f"scaling={self.scaling_efficiency:.1%}, "
            f"comm_overhead={self.communication_overhead_ms:.1f}ms"
        )


def setup_distributed(rank: int, world_size: int):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize process group
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def create_distributed_module(
    impl_name: str,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    rank: int,
    world_size: int,
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module | None:
    """Create distributed attention module."""
    embed_dim = num_heads * head_dim
    device = torch.device(f"cuda:{rank}")

    # Configure segment lengths based on sequence length
    if seq_len >= 32768:
        segment_lengths = [2048, 4096, 8192, seq_len]
        dilation_rates = [1, 2, 4, 8]
    else:
        segment_lengths = [1024, 2048, 4096]
        dilation_rates = [1, 2, 4]

    segment_lengths = [min(seg, seq_len) for seg in segment_lengths]

    try:
        if impl_name == "DistributedImprovedDilatedAttention" and DISTRIBUTED_AVAILABLE:
            return (
                DistributedImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    world_size=world_size,
                    rank=rank,
                )
                .to(device)
                .to(dtype)
            )

        elif (
            impl_name == "DistributedImprovedMultiheadDilatedAttention"
            and DISTRIBUTED_AVAILABLE
        ):
            return (
                DistributedImprovedMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    world_size=world_size,
                    rank=rank,
                )
                .to(device)
                .to(dtype)
            )

        elif (
            impl_name == "RingDistributedDilatedAttention"
            and RING_DISTRIBUTED_AVAILABLE
        ):
            # Ring size should match world size for distributed
            return (
                RingDistributedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=world_size,
                    enable_distributed=True,
                )
                .to(device)
                .to(dtype)
            )

        elif (
            impl_name == "BlockSparseRingDistributedDilatedAttention"
            and BLOCK_SPARSE_DISTRIBUTED_AVAILABLE
        ):
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
                SparsePatternConfig,
            )

            sparse_config = SparsePatternConfig(
                sparsity_ratio=0.9,
                block_size=64,
                local_window_size=256,
                pattern_type="dilated_sparse",
            )

            return (
                BlockSparseRingDistributedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparse_config=sparse_config,
                    world_size=world_size,
                    rank=rank,
                )
                .to(device)
                .to(dtype)
            )

    except Exception as e:
        print(f"Rank {rank}: Failed to create {impl_name}: {e}")
        return None

    return None


def benchmark_distributed_worker(
    rank: int,
    world_size: int,
    impl_name: str,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    num_runs: int,
    results_queue: mp.Queue,
):
    """Worker function for distributed benchmarking."""
    try:
        # Setup distributed
        setup_distributed(rank, world_size)

        # Create module
        module = create_distributed_module(
            impl_name, seq_len, num_heads, head_dim, rank, world_size
        )

        if module is None:
            results_queue.put(
                {"rank": rank, "success": False, "error": "Failed to create module"}
            )
            return

        embed_dim = num_heads * head_dim
        device = torch.device(f"cuda:{rank}")
        dtype = torch.float16

        # Create inputs
        if "Multihead" in impl_name:
            q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        else:
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            v = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = module(q, k, v)

        dist.barrier()
        torch.cuda.synchronize()

        # Measure computation time
        comp_times = []
        comm_times = []

        for _ in range(num_runs):
            torch.cuda.synchronize()

            # Total time
            total_start = time.perf_counter()

            # Mock communication time measurement (simplified)
            comm_start = time.perf_counter()
            dist.barrier()
            comm_end = time.perf_counter()

            # Computation time
            with torch.no_grad():
                _ = module(q, k, v)

            torch.cuda.synchronize()
            total_end = time.perf_counter()

            total_time = (total_end - total_start) * 1000
            comm_time = (comm_end - comm_start) * 1000
            comp_time = total_time - comm_time

            comp_times.append(comp_time)
            comm_times.append(comm_time)

        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Send results
        results_queue.put(
            {
                "rank": rank,
                "success": True,
                "comp_times": comp_times,
                "comm_times": comm_times,
                "peak_memory_mb": peak_memory,
            }
        )

    except Exception as e:
        results_queue.put({"rank": rank, "success": False, "error": str(e)})
    finally:
        cleanup_distributed()


def run_distributed_benchmark(
    impl_name: str,
    world_size: int,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    num_runs: int = 5,
    single_gpu_baseline: float | None = None,
) -> DistributedBenchmarkResult:
    """Run distributed benchmark across multiple GPUs."""

    # Check available GPUs
    if torch.cuda.device_count() < world_size:
        return DistributedBenchmarkResult(
            implementation=impl_name,
            world_size=world_size,
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mean_time_ms=0,
            std_time_ms=0,
            peak_memory_mb_per_gpu=0,
            total_memory_mb=0,
            throughput_tokens_per_sec=0,
            scaling_efficiency=0,
            communication_overhead_ms=0,
            success=False,
            error=f"Not enough GPUs available (need {world_size}, have {torch.cuda.device_count()})",
        )

    # Create results queue
    results_queue = mp.Queue()

    # Spawn processes
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=benchmark_distributed_worker,
            args=(
                rank,
                world_size,
                impl_name,
                seq_len,
                batch_size,
                num_heads,
                head_dim,
                num_runs,
                results_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(world_size):
        results.append(results_queue.get())

    # Wait for all processes
    for p in processes:
        p.join()

    # Check if all succeeded
    if not all(r["success"] for r in results):
        errors = [r.get("error", "Unknown error") for r in results if not r["success"]]
        return DistributedBenchmarkResult(
            implementation=impl_name,
            world_size=world_size,
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mean_time_ms=0,
            std_time_ms=0,
            peak_memory_mb_per_gpu=0,
            total_memory_mb=0,
            throughput_tokens_per_sec=0,
            scaling_efficiency=0,
            communication_overhead_ms=0,
            success=False,
            error=f"Distributed execution failed: {'; '.join(errors)}",
        )

    # Aggregate results
    all_comp_times = []
    all_comm_times = []
    peak_memories = []

    for r in results:
        all_comp_times.extend(r["comp_times"])
        all_comm_times.extend(r["comm_times"])
        peak_memories.append(r["peak_memory_mb"])

    mean_comp_time = sum(all_comp_times) / len(all_comp_times)
    mean_comm_time = sum(all_comm_times) / len(all_comm_times)
    mean_total_time = mean_comp_time + mean_comm_time
    std_time = (
        sum((t - mean_total_time) ** 2 for t in all_comp_times) / len(all_comp_times)
    ) ** 0.5

    peak_memory_per_gpu = max(peak_memories)
    total_memory = sum(peak_memories)

    # Calculate throughput
    total_tokens = batch_size * seq_len * world_size
    throughput = (total_tokens / mean_total_time) * 1000

    # Calculate scaling efficiency
    if single_gpu_baseline:
        ideal_time = single_gpu_baseline / world_size
        scaling_efficiency = ideal_time / mean_total_time
    else:
        scaling_efficiency = 1.0

    return DistributedBenchmarkResult(
        implementation=impl_name,
        world_size=world_size,
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        mean_time_ms=mean_total_time,
        std_time_ms=std_time,
        peak_memory_mb_per_gpu=peak_memory_per_gpu,
        total_memory_mb=total_memory,
        throughput_tokens_per_sec=throughput,
        scaling_efficiency=scaling_efficiency,
        communication_overhead_ms=mean_comm_time,
        success=True,
    )


def main():  # noqa: PLR0912
    parser = argparse.ArgumentParser(
        description="Multi-GPU distributed attention benchmark"
    )
    parser.add_argument(
        "--world-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/benchmarks",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Check for multiple GPUs
    if torch.cuda.device_count() < 2:
        print("ERROR: Multi-GPU benchmarking requires at least 2 GPUs")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        return

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Determine implementations
    implementations = []
    if DISTRIBUTED_AVAILABLE:
        implementations.extend(
            [
                "DistributedImprovedDilatedAttention",
                "DistributedImprovedMultiheadDilatedAttention",
            ]
        )
    if RING_DISTRIBUTED_AVAILABLE:
        implementations.append("RingDistributedDilatedAttention")
    if BLOCK_SPARSE_DISTRIBUTED_AVAILABLE:
        implementations.append("BlockSparseRingDistributedDilatedAttention")

    if not implementations:
        print("ERROR: No distributed implementations available")
        return

    print(f"\nImplementations to benchmark: {implementations}")
    print(f"World sizes: {args.world_sizes}")
    print(f"Sequence lengths: {args.sequence_lengths}")

    # Run benchmarks
    results = {impl: [] for impl in implementations}

    for seq_len in args.sequence_lengths:
        print(f"\n{'=' * 60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'=' * 60}")

        for impl_name in implementations:
            print(f"\n{impl_name}:")

            # Get single GPU baseline
            single_gpu_result = None
            if 1 in args.world_sizes:
                single_gpu_result = run_distributed_benchmark(
                    impl_name,
                    1,
                    seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    args.num_runs,
                )
                results[impl_name].append(single_gpu_result)
                print(f"  {single_gpu_result}")

            # Multi-GPU benchmarks
            for world_size in args.world_sizes:
                if world_size == 1:
                    continue

                baseline = (
                    single_gpu_result.mean_time_ms
                    if single_gpu_result and single_gpu_result.success
                    else None
                )

                result = run_distributed_benchmark(
                    impl_name,
                    world_size,
                    seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    args.num_runs,
                    single_gpu_baseline=baseline,
                )

                results[impl_name].append(result)
                print(f"  {result}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-UTC")

    results_dict = {
        "metadata": {
            "timestamp": timestamp,
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "batch_size_per_gpu": args.batch_size,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
        "results": {
            impl: [asdict(r) for r in impl_results]
            for impl, impl_results in results.items()
        },
    }

    json_path = os.path.join(args.output_dir, f"benchmark-distributed-{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DISTRIBUTED BENCHMARK SUMMARY")
    print("=" * 60)

    for seq_len in args.sequence_lengths:
        print(f"\nSequence Length: {seq_len}")

        for world_size in args.world_sizes:
            print(f"\n  World Size: {world_size} GPU(s)")
            print("  " + "-" * 40)

            for impl in implementations:
                result = next(
                    (
                        r
                        for r in results[impl]
                        if r.seq_len == seq_len and r.world_size == world_size
                    ),
                    None,
                )

                if result:
                    if result.success:
                        print(
                            f"  {impl:40s}: {result.mean_time_ms:8.1f}ms, "
                            f"Scaling: {result.scaling_efficiency:.1%}, "
                            f"Comm: {result.communication_overhead_ms:.1f}ms"
                        )
                    else:
                        print(f"  {impl:40s}: FAILED - {result.error}")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    main()
