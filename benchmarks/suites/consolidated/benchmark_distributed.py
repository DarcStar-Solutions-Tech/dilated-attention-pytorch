#!/usr/bin/env python3
"""
Distributed/Multi-GPU benchmark suite.

This consolidates functionality from:
- benchmark_multi_gpu.py
- benchmark_ring_distributed_actual.py
- test_distributed_suite.py
- benchmark_ring_attention_correct_multi_gpu.py
- test_multi_gpu_actual.py
- And many other distributed test files
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.core.config import BenchmarkConfig  # noqa: E402
from benchmarks.core.unified_runner import UnifiedBenchmarkRunner  # noqa: E402
from benchmarks.core.utils.distributed import cleanup_distributed, setup_distributed  # noqa: E402


class DistributedBenchmark:
    """Benchmark for distributed/multi-GPU configurations."""

    def __init__(self, rank: int, world_size: int, config: BenchmarkConfig):
        """Initialize distributed benchmark.

        Args:
            rank: Process rank
            world_size: Total number of processes
            config: Benchmark configuration
        """
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.config.device = f"cuda:{rank}"
        self.runner = UnifiedBenchmarkRunner(config)

    def benchmark_weak_scaling(self):
        """Benchmark weak scaling (constant load per GPU)."""
        if self.rank == 0:
            print("\nWeak Scaling Test")
            print("=" * 80)
            print("Each GPU processes the same amount of data")
            print("-" * 80)

        # Each GPU gets same batch size
        batch_size_per_gpu = self.config.batch_sizes[0]
        results = []

        for seq_len in self.config.sequence_lengths:
            # Find segment config
            segment_lengths = None
            dilation_rates = None

            for segs, dils in zip(
                self.config.segment_lengths, self.config.dilation_rates
            ):
                if seq_len % max(segs) == 0:
                    segment_lengths = segs
                    dilation_rates = dils
                    break

            if segment_lengths is None:
                continue

            # Synchronize before timing
            if dist.is_initialized():
                dist.barrier()

            # Run benchmark
            result = self.runner.benchmark_single_configuration(
                implementation="ring",
                batch_size=batch_size_per_gpu,
                seq_len=seq_len,
                num_heads=self.config.num_heads[0],
                embed_dim=self.config.embed_dims[0],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )

            # Gather results to rank 0
            if dist.is_initialized():
                # Collect timing from all ranks
                all_times = [None] * self.world_size
                dist.all_gather_object(all_times, result.forward_time_ms)

                if self.rank == 0 and not result.error:
                    avg_time = sum(all_times) / len(all_times)
                    total_batch = batch_size_per_gpu * self.world_size
                    print(
                        f"  L={seq_len:5d}: {avg_time:6.2f}ms (avg), "
                        f"Total batch={total_batch}, "
                        f"Throughput={total_batch * seq_len / avg_time * 1000:.0f} tok/s"
                    )

            results.append(result)

        return results

    def benchmark_strong_scaling(self):
        """Benchmark strong scaling (fixed total problem size)."""
        if self.rank == 0:
            print("\nStrong Scaling Test")
            print("=" * 80)
            print("Total workload is divided among GPUs")
            print("-" * 80)

        # Total batch size is fixed, divided among GPUs
        total_batch_size = self.config.batch_sizes[0] * self.world_size
        batch_size_per_gpu = total_batch_size // self.world_size

        results = []

        for seq_len in self.config.sequence_lengths:
            # For ring attention, sequence is also divided
            seq_len_per_gpu = seq_len // self.world_size

            # Find segment config
            segment_lengths = None
            dilation_rates = None

            for segs, dils in zip(
                self.config.segment_lengths, self.config.dilation_rates
            ):
                if seq_len % max(segs) == 0:
                    segment_lengths = segs
                    dilation_rates = dils
                    break

            if segment_lengths is None:
                continue

            # Synchronize
            if dist.is_initialized():
                dist.barrier()

            # Run benchmark
            result = self.runner.benchmark_single_configuration(
                implementation="ring",
                batch_size=batch_size_per_gpu,
                seq_len=seq_len,  # Full sequence length
                num_heads=self.config.num_heads[0],
                embed_dim=self.config.embed_dims[0],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )

            # Report results
            if dist.is_initialized():
                all_times = [None] * self.world_size
                dist.all_gather_object(all_times, result.forward_time_ms)

                if self.rank == 0 and not result.error:
                    avg_time = sum(all_times) / len(all_times)
                    print(
                        f"  L={seq_len:5d} ({seq_len_per_gpu}/GPU): "
                        f"{avg_time:6.2f}ms, "
                        f"Speedup vs 1 GPU: {seq_len / seq_len_per_gpu:.1f}x theoretical"
                    )

            results.append(result)

        return results

    def benchmark_communication_overhead(self):
        """Benchmark communication overhead in ring attention."""
        if self.rank == 0:
            print("\nCommunication Overhead Analysis")
            print("=" * 80)

        # Test different message sizes
        message_sizes = [
            1024 * 1024,
            10 * 1024 * 1024,
            100 * 1024 * 1024,
        ]  # 1MB, 10MB, 100MB

        for size in message_sizes:
            # Create tensors
            send_tensor = torch.randn(size // 4, device=f"cuda:{self.rank}")
            recv_tensor = torch.empty_like(send_tensor)

            # Time communication
            if dist.is_initialized():
                dist.barrier()

                start = time.perf_counter()

                # Ring communication pattern
                src = (self.rank - 1) % self.world_size
                dst = (self.rank + 1) % self.world_size

                # Non-blocking send/recv
                send_op = dist.isend(send_tensor, dst)
                recv_op = dist.irecv(recv_tensor, src)

                send_op.wait()
                recv_op.wait()

                torch.cuda.synchronize()
                end = time.perf_counter()

                comm_time = (end - start) * 1000  # ms
                bandwidth = size / (1024**3) / (comm_time / 1000)  # GB/s

                if self.rank == 0:
                    print(
                        f"  {size / 1024**2:6.1f}MB: {comm_time:6.2f}ms, "
                        f"{bandwidth:5.1f} GB/s"
                    )

    def run_distributed_benchmarks(self):
        """Run all distributed benchmarks."""
        if self.rank == 0:
            print(f"\nRunning on {self.world_size} GPUs")
            print("=" * 80)

        # 1. Weak scaling
        weak_results = self.benchmark_weak_scaling()

        # 2. Strong scaling
        strong_results = self.benchmark_strong_scaling()

        # 3. Communication overhead
        if self.world_size > 1:
            self.benchmark_communication_overhead()

        return weak_results, strong_results


def run_rank(rank: int, world_size: int, config: BenchmarkConfig):
    """Run benchmarks on a single rank.

    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Benchmark configuration
    """
    # Setup distributed
    setup_distributed(rank, world_size, backend=config.backend)

    # Run benchmarks
    benchmark = DistributedBenchmark(rank, world_size, config)
    benchmark.run_distributed_benchmarks()

    # Cleanup
    cleanup_distributed()


def main():
    """Run distributed benchmarks."""
    parser = argparse.ArgumentParser(description="Distributed/Multi-GPU benchmarks")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[8192, 16384, 32768],
        help="Sequence lengths to test",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend",
    )

    args = parser.parse_args()

    # Determine world size
    if args.world_size is None:
        args.world_size = torch.cuda.device_count()

    if args.world_size < 2:
        print("ERROR: Distributed benchmarks require at least 2 GPUs")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        return

    # Create configuration
    config = BenchmarkConfig(
        implementations=["ring"],
        sequence_lengths=args.seq_lengths,
        batch_sizes=[args.batch_size],
        num_heads=[12],
        embed_dims=[768],
        segment_lengths=[[2048, 4096, 8192], [4096, 8192, 16384]],
        dilation_rates=[[1, 2, 4], [1, 2, 4]],
        backend=args.backend,
        distributed=True,
        world_size=args.world_size,
        warmup_iterations=2,
        benchmark_iterations=5,
    )

    print("=" * 80)
    print("Distributed Attention Benchmarks")
    print("=" * 80)
    print(f"World size: {args.world_size} GPUs")
    print(f"Backend: {args.backend}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch size per GPU: {args.batch_size}")
    print("=" * 80)

    # Launch processes
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    mp.spawn(
        run_rank, args=(args.world_size, config), nprocs=args.world_size, join=True
    )

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
