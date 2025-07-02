#!/usr/bin/env python3
"""
Multi-GPU Benchmark for RingDilatedAttentionV2Collective

This benchmark validates the distributed functionality of RingDilatedAttentionV2Collective
by testing performance scaling across different GPU configurations.

Usage:
    # Single GPU (baseline)
    python benchmark_ring_v2_collective_distributed.py

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 benchmark_ring_v2_collective_distributed.py
    torchrun --nproc_per_node=4 benchmark_ring_v2_collective_distributed.py
    torchrun --nproc_per_node=8 benchmark_ring_v2_collective_distributed.py
"""

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

# Import the module to benchmark
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    batch_size: int = 2
    seq_lengths: List[int] = None
    embed_dim: int = 768
    num_heads: int = 12
    segment_lengths: List[int] = None
    dilation_rates: List[int] = None
    ring_sizes: List[int] = None
    warmup_runs: int = 3
    benchmark_runs: int = 10
    use_flash_attention: bool = True
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    profile_enabled: bool = False
    save_results: bool = True
    output_dir: str = "benchmark_results"

    def __post_init__(self):
        if self.seq_lengths is None:
            self.seq_lengths = [16384, 32768, 65536, 131072]
        if self.segment_lengths is None:
            self.segment_lengths = [2048, 4096, 8192]
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4]
        if self.ring_sizes is None:
            self.ring_sizes = [1, 2, 4, 8]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    seq_length: int
    ring_size: int
    world_size: int
    rank: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    throughput_tokens_per_sec: float
    effective_batch_size: int
    distributed_overhead_ms: Optional[float] = None
    communication_time_ms: Optional[float] = None


class DistributedBenchmark:
    """Handles distributed setup and benchmarking."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.device = None
        self.is_distributed = False

    def setup_distributed(self):
        """Initialize distributed training if running with torchrun."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
            self.is_distributed = True

            # Initialize process group
            dist.init_process_group(backend="nccl")

            # Set device
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            # Single GPU mode
            self.device = torch.device(
                self.config.device if torch.cuda.is_available() else "cpu"
            )

        return self.device

    def cleanup_distributed(self):
        """Clean up distributed process group."""
        if self.is_distributed:
            dist.destroy_process_group()

    def log(self, message: str):
        """Log message only from rank 0."""
        if self.rank == 0:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

    def create_model(
        self, seq_length: int, ring_size: int
    ) -> RingDilatedAttentionV2Collective:
        """Create the attention model with given configuration."""
        # Adjust ring size based on world size
        effective_ring_size = min(ring_size, self.world_size)

        model = RingDilatedAttentionV2Collective(
            segment_lengths=self.config.segment_lengths,
            dilation_rates=self.config.dilation_rates,
            ring_size=effective_ring_size,
            use_flash_attention=self.config.use_flash_attention,
            device=self.device,
            dtype=self.config.dtype,
        ).to(self.device)

        return model

    def generate_input(
        self, seq_length: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random input tensors."""
        # For distributed, each rank processes a subset
        effective_batch_size = batch_size
        if self.is_distributed:
            effective_batch_size = batch_size // self.world_size
            if effective_batch_size == 0:
                effective_batch_size = 1

        query = torch.randn(
            effective_batch_size,
            seq_length,
            self.config.num_heads,
            self.config.embed_dim // self.config.num_heads,
            device=self.device,
            dtype=self.config.dtype,
            requires_grad=True,
        )

        # For self-attention
        key = _ = query

        return query, key

    def measure_communication_overhead(
        self, model: nn.Module, seq_length: int
    ) -> float:
        """Measure the communication overhead in distributed setting."""
        if not self.is_distributed:
            return 0.0

        # Create dummy tensor of appropriate size
        chunk_size = seq_length // model.ring_size
        dummy_tensor = torch.randn(
            self.config.batch_size,
            chunk_size,
            self.config.num_heads,
            self.config.embed_dim // self.config.num_heads,
            device=self.device,
            dtype=self.config.dtype,
        )

        # Measure all_gather time
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        gathered = [torch.empty_like(dummy_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, dummy_tensor)

        torch.cuda.synchronize()
        comm_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        return comm_time

    def run_single_benchmark(self, seq_length: int, ring_size: int) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        self.log(
            f"Running benchmark: seq_length={seq_length}, ring_size={ring_size}, "
            f"world_size={self.world_size}"
        )

        # Create model
        model = self.create_model(seq_length, ring_size)
        model.train()  # Enable gradients

        # Generate input
        query, key = self.generate_input(seq_length, self.config.batch_size)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Warmup runs
        for _ in range(self.config.warmup_runs):
            output = model(query, key, key)
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()

        # Measure communication overhead
        comm_time = self.measure_communication_overhead(model, seq_length)

        # Benchmark runs
        forward_times = []
        backward_times = []
        total_times = []

        for _ in range(self.config.benchmark_runs):
            # Clear gradients
            model.zero_grad()
            if query.grad is not None:
                query.grad.zero_()

            torch.cuda.synchronize()
            start_total = time.perf_counter()

            # Forward pass
            start_forward = time.perf_counter()
            output = model(query, key, key)
            torch.cuda.synchronize()
            forward_time = time.perf_counter() - start_forward

            # Backward pass
            start_backward = time.perf_counter()
            loss = output.sum()
            loss.backward()
            torch.cuda.synchronize()
            backward_time = time.perf_counter() - start_backward

            total_time = time.perf_counter() - start_total

            forward_times.append(forward_time * 1000)  # Convert to ms
            backward_times.append(backward_time * 1000)
            total_times.append(total_time * 1000)

        # Calculate averages
        avg_forward = sum(forward_times) / len(forward_times)
        avg_backward = sum(backward_times) / len(backward_times)
        avg_total = sum(total_times) / len(total_times)

        # Memory stats
        memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9  # GB
        memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9  # GB

        # Calculate throughput
        effective_batch_size = self.config.batch_size
        if self.is_distributed:
            effective_batch_size = self.config.batch_size // self.world_size
            if effective_batch_size == 0:
                effective_batch_size = 1

        total_tokens = effective_batch_size * seq_length * self.world_size
        throughput = total_tokens / (avg_total / 1000)  # tokens per second

        # Calculate distributed overhead
        distributed_overhead = None
        if self.is_distributed and ring_size > 1:
            # Estimate overhead as difference from ideal linear scaling
            ideal_time = avg_total / self.world_size
            distributed_overhead = avg_total - ideal_time

        result = BenchmarkResult(
            seq_length=seq_length,
            ring_size=ring_size,
            world_size=self.world_size,
            rank=self.rank,
            forward_time_ms=avg_forward,
            backward_time_ms=avg_backward,
            total_time_ms=avg_total,
            memory_allocated_gb=memory_allocated,
            memory_reserved_gb=memory_reserved,
            throughput_tokens_per_sec=throughput,
            effective_batch_size=effective_batch_size,
            distributed_overhead_ms=distributed_overhead,
            communication_time_ms=comm_time,
        )

        # Clean up
        del model, query, key, output
        torch.cuda.empty_cache()
        gc.collect()

        return result

    def run_profiled_benchmark(
        self, seq_length: int, ring_size: int
    ) -> Tuple[BenchmarkResult, str]:
        """Run benchmark with profiling enabled."""
        self.log(f"Running profiled benchmark: seq_length={seq_length}")

        model = self.create_model(seq_length, ring_size)
        model.train()
        query, key = self.generate_input(seq_length, self.config.batch_size)

        # Profile filename
        profile_file = os.path.join(
            self.config.output_dir,
            f"profile_seq{seq_length}_ring{ring_size}_rank{self.rank}.json",
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("model_forward"):
                output = model(query, key, key)

            with record_function("model_backward"):
                loss = output.sum()
                loss.backward()

        # Export profile
        if self.rank == 0:
            prof.export_chrome_trace(profile_file)

        # Still collect timing data
        result = self.run_single_benchmark(seq_length, ring_size)

        return result, profile_file

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark configurations."""
        results = []

        for seq_length in self.config.seq_lengths:
            # Skip sequences that are too long for available memory
            if seq_length > 131072 and self.config.batch_size > 1:
                self.log(
                    f"Skipping seq_length={seq_length} with batch_size="
                    f"{self.config.batch_size} (too large)"
                )
                continue

            for ring_size in self.config.ring_sizes:
                # Skip ring sizes larger than world size
                if ring_size > self.world_size:
                    continue

                try:
                    if (
                        self.config.profile_enabled
                        and seq_length == self.config.seq_lengths[0]
                    ):
                        result, profile_file = self.run_profiled_benchmark(
                            seq_length, ring_size
                        )
                        self.log(f"Profile saved to: {profile_file}")
                    else:
                        result = self.run_single_benchmark(seq_length, ring_size)

                    results.append(result)

                    # Log result
                    self.log(
                        f"  Forward: {result.forward_time_ms:.2f}ms, "
                        f"Backward: {result.backward_time_ms:.2f}ms, "
                        f"Total: {result.total_time_ms:.2f}ms, "
                        f"Throughput: {result.throughput_tokens_per_sec:.0f} tokens/s"
                    )
                    if result.communication_time_ms:
                        self.log(
                            f"  Communication: {result.communication_time_ms:.2f}ms"
                        )

                except torch.cuda.OutOfMemoryError:
                    self.log(
                        f"  OOM for seq_length={seq_length}, ring_size={ring_size}"
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    self.log(f"  Error: {str(e)}")
                    torch.cuda.empty_cache()
                    gc.collect()

        return results

    def save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to JSON file."""
        if not self.config.save_results or self.rank != 0:
            return

        os.makedirs(self.config.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.config.output_dir,
            f"ring_v2_collective_benchmark_{timestamp}_world{self.world_size}.json",
        )

        # Convert results to dict
        results_dict = {
            "config": asdict(self.config),
            "world_size": self.world_size,
            "timestamp": timestamp,
            "results": [asdict(r) for r in results],
        }

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        self.log(f"Results saved to: {filename}")

    def print_summary(self, results: List[BenchmarkResult]):
        """Print summary of benchmark results."""
        if self.rank != 0 or not results:
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by sequence length
        seq_results = {}
        for r in results:
            if r.seq_length not in seq_results:
                seq_results[r.seq_length] = []
            seq_results[r.seq_length].append(r)

        for seq_length in sorted(seq_results.keys()):
            print(f"\nSequence Length: {seq_length}")
            print("-" * 60)
            print(
                f"{'Ring Size':<10} {'Forward (ms)':<15} {'Backward (ms)':<15} "
                f"{'Total (ms)':<15} {'Throughput':<15}"
            )
            print("-" * 60)

            for r in sorted(seq_results[seq_length], key=lambda x: x.ring_size):
                print(
                    f"{r.ring_size:<10} {r.forward_time_ms:<15.2f} "
                    f"{r.backward_time_ms:<15.2f} {r.total_time_ms:<15.2f} "
                    f"{r.throughput_tokens_per_sec:<15.0f}"
                )

        # Scaling efficiency analysis
        if self.world_size > 1:
            print("\n" + "=" * 80)
            print("SCALING EFFICIENCY ANALYSIS")
            print("=" * 80)

            for seq_length in sorted(seq_results.keys()):
                results_for_seq = seq_results[seq_length]

                # Find baseline (ring_size=1)
                baseline = next((r for r in results_for_seq if r.ring_size == 1), None)
                if not baseline:
                    continue

                print(f"\nSequence Length: {seq_length}")
                print("-" * 60)
                print(
                    f"{'Ring Size':<10} {'Speedup':<10} {'Efficiency':<12} "
                    f"{'Comm Time (ms)':<15}"
                )
                print("-" * 60)

                for r in sorted(results_for_seq, key=lambda x: x.ring_size):
                    speedup = baseline.total_time_ms / r.total_time_ms
                    efficiency = speedup / r.ring_size * 100
                    comm_time = r.communication_time_ms or 0

                    print(
                        f"{r.ring_size:<10} {speedup:<10.2f} {efficiency:<12.1f}% "
                        f"{comm_time:<15.2f}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU benchmark for RingDilatedAttentionV2Collective"
    )

    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for benchmarking"
    )
    parser.add_argument(
        "--seq_lengths",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads")
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        help="Segment lengths",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Dilation rates",
    )
    parser.add_argument(
        "--ring_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Ring sizes to test",
    )
    parser.add_argument(
        "--warmup_runs", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument(
        "--benchmark_runs", type=int, default=10, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable Flash Attention",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable profiling for first config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Don't save results to file"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create config
    config = BenchmarkConfig(
        batch_size=args.batch_size,
        seq_lengths=args.seq_lengths,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        ring_sizes=args.ring_sizes,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        use_flash_attention=not args.no_flash_attention,
        dtype=dtype,
        profile_enabled=args.profile,
        save_results=not args.no_save,
        output_dir=args.output_dir,
    )

    # Create and run benchmark
    benchmark = DistributedBenchmark(config)

    try:
        # Setup distributed
        device = benchmark.setup_distributed()
        benchmark.log(f"Initialized on {device}, world_size={benchmark.world_size}")

        # Run benchmarks
        results = benchmark.run_all_benchmarks()

        # Save and print results
        benchmark.save_results(results)
        benchmark.print_summary(results)

    finally:
        # Cleanup
        benchmark.cleanup_distributed()


if __name__ == "__main__":
    main()
