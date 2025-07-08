#!/usr/bin/env python3
"""
Comprehensive benchmark for corrected ring attention with O(n/k) memory.

This benchmark tests:
1. Memory usage scaling with world size
2. Performance at extreme sequence lengths (200K+ tokens)
3. Comparison between implementations
4. Multi-GPU simulation with proper memory accounting
"""

import torch
import gc
import os
import sys
import time
import json
from datetime import datetime
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_core import (
    RingDilatedAttentionHilbertCore,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info, get_optimal_dtype


@dataclass
class BenchmarkResult:
    """Store benchmark results."""

    implementation: str
    world_size: int
    total_seq_len: int
    local_seq_len: int
    batch_size: int
    memory_mb: float
    forward_time_ms: float
    tokens_per_second: float
    memory_per_token_mb: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    batch_sizes: List[int]
    world_sizes: List[int]
    sequence_lengths: List[int]
    implementations: List[Tuple[type, str]]
    embed_dim: int = 768
    num_heads: int = 12
    segment_lengths: List[int] = None
    dilation_rates: List[int] = None

    def __post_init__(self):
        if self.segment_lengths is None:
            self.segment_lengths = [4096, 8192, 16384]
        if self.dilation_rates is None:
            self.dilation_rates = [1, 2, 4]


class RingAttentionBenchmark:
    """Comprehensive ring attention benchmark."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_info = get_gpu_info(self.device)
        self.optimal_dtype = get_optimal_dtype(self.device)
        self.results: List[BenchmarkResult] = []

    def cleanup_memory(self):
        """Force memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0

    def benchmark_single_config(
        self,
        model_class: type,
        model_name: str,
        batch_size: int,
        world_size: int,
        total_seq_len: int,
        warmup_steps: int = 3,
        measure_steps: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        local_seq_len = total_seq_len // world_size

        # Skip if sequence is too short for world size
        if local_seq_len < 1024:
            return BenchmarkResult(
                implementation=model_name,
                world_size=world_size,
                total_seq_len=total_seq_len,
                local_seq_len=local_seq_len,
                batch_size=batch_size,
                memory_mb=0,
                forward_time_ms=0,
                tokens_per_second=0,
                memory_per_token_mb=0,
                success=False,
                error="Sequence too short for world size",
            )

        self.cleanup_memory()
        start_mem = self.get_memory_mb()

        try:
            # Create local input
            x_local = torch.randn(
                batch_size,
                local_seq_len,
                self.config.embed_dim,
                device=self.device,
                dtype=self.optimal_dtype,
            )

            # Create model
            if model_name == "HilbertCore":
                model = model_class(
                    dim=self.config.embed_dim,
                    heads=self.config.num_heads,
                    segment_lengths=self.config.segment_lengths,
                    dilation_rates=self.config.dilation_rates,
                    ring_size=world_size,
                    use_hilbert=True,
                    use_custom_backward=True,
                )
            else:
                model = model_class(
                    embed_dim=self.config.embed_dim,
                    num_heads=self.config.num_heads,
                    segment_lengths=self.config.segment_lengths,
                    dilation_rates=self.config.dilation_rates,
                    dropout=0.0,
                    use_hilbert=True,
                    device=self.device,
                    dtype=self.optimal_dtype,
                    memory_efficient=True,
                )

            # Move model to device
            model = model.to(self.device)
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(warmup_steps):
                    _ = model(x_local, total_seq_len=total_seq_len, already_split=True)

            # Measure
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            forward_times = []

            with torch.no_grad():
                for _ in range(measure_steps):
                    start_time = time.perf_counter()

                    output = model(
                        x_local, total_seq_len=total_seq_len, already_split=True
                    )

                    torch.cuda.synchronize() if self.device.type == "cuda" else None
                    forward_times.append(time.perf_counter() - start_time)

            # Calculate metrics
            avg_forward_time = sum(forward_times) / len(forward_times)
            forward_time_ms = avg_forward_time * 1000
            tokens_per_second = (batch_size * total_seq_len) / avg_forward_time

            # Memory usage
            peak_mem = self.get_memory_mb()
            memory_mb = peak_mem - start_mem
            memory_per_token_mb = memory_mb / (batch_size * local_seq_len)

            # Cleanup
            del x_local, model, output

            return BenchmarkResult(
                implementation=model_name,
                world_size=world_size,
                total_seq_len=total_seq_len,
                local_seq_len=local_seq_len,
                batch_size=batch_size,
                memory_mb=memory_mb,
                forward_time_ms=forward_time_ms,
                tokens_per_second=tokens_per_second,
                memory_per_token_mb=memory_per_token_mb,
                success=True,
            )

        except torch.cuda.OutOfMemoryError:
            return BenchmarkResult(
                implementation=model_name,
                world_size=world_size,
                total_seq_len=total_seq_len,
                local_seq_len=local_seq_len,
                batch_size=batch_size,
                memory_mb=float("inf"),
                forward_time_ms=float("inf"),
                tokens_per_second=0,
                memory_per_token_mb=float("inf"),
                success=False,
                error="OOM",
            )
        except Exception as e:
            return BenchmarkResult(
                implementation=model_name,
                world_size=world_size,
                total_seq_len=total_seq_len,
                local_seq_len=local_seq_len,
                batch_size=batch_size,
                memory_mb=0,
                forward_time_ms=0,
                tokens_per_second=0,
                memory_per_token_mb=0,
                success=False,
                error=str(e),
            )
        finally:
            self.cleanup_memory()

    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print("=" * 80)
        print("Ring Attention Correctness & Performance Benchmark")
        print("=" * 80)
        print(f"GPU: {self.gpu_info.name} ({self.gpu_info.architecture})")
        print(f"Compute capability: {self.gpu_info.compute_capability}")
        print(f"Total memory: {self.gpu_info.total_memory_gb:.1f} GB")
        print(f"Available memory: {self.gpu_info.available_memory_gb:.1f} GB")
        print(f"Optimal dtype: {self.optimal_dtype}")
        print()

        total_configs = (
            len(self.config.implementations)
            * len(self.config.batch_sizes)
            * len(self.config.world_sizes)
            * len(self.config.sequence_lengths)
        )

        print(f"Running {total_configs} benchmark configurations...")
        print("-" * 80)

        config_num = 0
        for model_class, model_name in self.config.implementations:
            for batch_size in self.config.batch_sizes:
                for world_size in self.config.world_sizes:
                    for seq_len in self.config.sequence_lengths:
                        config_num += 1

                        # Skip if we've already failed at a shorter sequence
                        if self.results:
                            prev_results = [
                                r
                                for r in self.results
                                if r.implementation == model_name
                                and r.batch_size == batch_size
                                and r.world_size == world_size
                                and r.total_seq_len < seq_len
                                and not r.success
                            ]
                            if prev_results:
                                print(
                                    f"[{config_num}/{total_configs}] Skipping {model_name} "
                                    f"B={batch_size} W={world_size} L={seq_len:,} (prev OOM)"
                                )
                                continue

                        print(
                            f"[{config_num}/{total_configs}] Testing {model_name} "
                            f"B={batch_size} W={world_size} L={seq_len:,}...",
                            end="",
                            flush=True,
                        )

                        result = self.benchmark_single_config(
                            model_class, model_name, batch_size, world_size, seq_len
                        )
                        self.results.append(result)

                        if result.success:
                            print(
                                f" ✓ {result.memory_mb:.1f}MB, "
                                f"{result.forward_time_ms:.1f}ms, "
                                f"{result.tokens_per_second / 1e6:.2f}M tok/s"
                            )
                        else:
                            print(f" ✗ {result.error}")

        return self.results

    def analyze_results(self):
        """Analyze and print benchmark results."""
        print()
        print("=" * 80)
        print("Benchmark Results Summary")
        print("=" * 80)

        # Group by implementation
        implementations = list(set(r.implementation for r in self.results))

        for impl in implementations:
            impl_results = [r for r in self.results if r.implementation == impl]
            successful = [r for r in impl_results if r.success]

            print(f"\n{impl}:")
            print("-" * 40)

            if not successful:
                print("  No successful runs")
                continue

            # Max sequence length by world size
            print("  Max sequence length achieved:")
            for world_size in sorted(set(r.world_size for r in successful)):
                ws_results = [r for r in successful if r.world_size == world_size]
                max_seq = max(r.total_seq_len for r in ws_results)
                print(f"    World size {world_size}: {max_seq:,} tokens")

            # Memory efficiency
            print("\n  Memory efficiency (MB per 1K tokens):")
            for world_size in sorted(set(r.world_size for r in successful)):
                ws_results = [r for r in successful if r.world_size == world_size]
                avg_mem_per_token = sum(
                    r.memory_per_token_mb for r in ws_results
                ) / len(ws_results)
                print(
                    f"    World size {world_size}: {avg_mem_per_token * 1000:.2f} MB/1K tokens"
                )

            # Performance
            print("\n  Best throughput:")
            best_perf = max(successful, key=lambda r: r.tokens_per_second)
            print(
                f"    {best_perf.tokens_per_second / 1e6:.2f}M tokens/sec "
                f"(B={best_perf.batch_size}, W={best_perf.world_size}, "
                f"L={best_perf.total_seq_len:,})"
            )

        # Verify O(n/k) scaling
        print()
        print("=" * 80)
        print("Memory Scaling Verification (O(n/k) property)")
        print("=" * 80)

        # Check if memory scales with local sequence length
        for impl in implementations:
            print(f"\n{impl}:")

            # Group by batch size
            batch_sizes = sorted(
                set(r.batch_size for r in self.results if r.implementation == impl)
            )

            for batch_size in batch_sizes:
                print(f"  Batch size {batch_size}:")

                # Collect data points
                data_points = []
                for r in self.results:
                    if (
                        r.implementation == impl
                        and r.batch_size == batch_size
                        and r.success
                    ):
                        data_points.append((r.local_seq_len, r.memory_mb))

                if len(data_points) > 1:
                    # Sort by local sequence length
                    data_points.sort()

                    # Calculate scaling factor
                    for i in range(1, len(data_points)):
                        seq_ratio = data_points[i][0] / data_points[0][0]
                        mem_ratio = data_points[i][1] / data_points[0][1]
                        print(
                            f"    {data_points[0][0]:,} → {data_points[i][0]:,} tokens: "
                            f"seq {seq_ratio:.1f}x, mem {mem_ratio:.1f}x "
                            f"{'✓' if abs(seq_ratio - mem_ratio) < 0.2 else '✗'}"
                        )

    def save_results(self, filename: Optional[str] = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
            filename = f"benchmark-ring-attention-correct-{timestamp}.json"

        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": {
                "name": self.gpu_info.name,
                "architecture": self.gpu_info.architecture,
                "compute_capability": self.gpu_info.compute_capability,
                "total_memory_gb": self.gpu_info.total_memory_gb,
                "optimal_dtype": str(self.optimal_dtype),
            },
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to: {filename}")


def test_extreme_sequences():
    """Test extreme sequence lengths to verify 200K+ token capability."""
    config = BenchmarkConfig(
        batch_sizes=[1],
        world_sizes=[1, 2, 4, 8],
        sequence_lengths=[
            8192,  # 8K
            16384,  # 16K
            32768,  # 32K
            65536,  # 64K
            131072,  # 128K
            262144,  # 256K
            524288,  # 512K
        ],
        implementations=[
            (RingDilatedAttentionHilbertOptimizedCorrect, "HilbertOptimizedCorrect"),
            (RingDilatedAttentionHilbertCore, "HilbertCore"),
        ],
    )

    benchmark = RingAttentionBenchmark(config)
    results = benchmark.run_benchmarks()
    benchmark.analyze_results()
    benchmark.save_results()

    # Special analysis for 200K+ tokens
    print()
    print("=" * 80)
    print("200K+ Token Sequence Analysis")
    print("=" * 80)

    large_seq_results = [r for r in results if r.total_seq_len >= 200000 and r.success]

    if large_seq_results:
        print(
            f"Successfully processed {len(large_seq_results)} configurations with 200K+ tokens!"
        )
        print()

        for r in sorted(large_seq_results, key=lambda x: x.total_seq_len):
            print(
                f"{r.implementation}: {r.total_seq_len:,} tokens "
                f"(world_size={r.world_size}, {r.memory_mb:.1f}MB, "
                f"{r.forward_time_ms:.1f}ms)"
            )
    else:
        print("Could not process 200K+ tokens with current GPU memory.")
        print()

        # Find the limiting factor
        max_by_world = {}
        for r in results:
            if r.success:
                if (
                    r.world_size not in max_by_world
                    or r.total_seq_len > max_by_world[r.world_size]
                ):
                    max_by_world[r.world_size] = r.total_seq_len

        print("Maximum sequence lengths achieved:")
        for ws in sorted(max_by_world.keys()):
            print(f"  World size {ws}: {max_by_world[ws]:,} tokens")

        # Estimate requirements for 200K
        if max_by_world:
            best_ws = max(max_by_world.keys())
            best_len = max_by_world[best_ws]
            if best_len > 0:
                required_ws = int(200000 * best_ws / best_len)
                print(f"\nEstimated world size needed for 200K tokens: {required_ws}")


def test_memory_scaling():
    """Test memory scaling to verify O(n/k) property."""
    config = BenchmarkConfig(
        batch_sizes=[1, 2],
        world_sizes=[1, 2, 4, 8],
        sequence_lengths=[8192, 16384, 32768],
        implementations=[
            (RingDilatedAttentionHilbertOptimizedCorrect, "HilbertOptimizedCorrect"),
        ],
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
    )

    benchmark = RingAttentionBenchmark(config)
    results = benchmark.run_benchmarks()
    benchmark.analyze_results()

    # Create visualization of memory scaling
    print()
    print("=" * 80)
    print("Memory Scaling Visualization")
    print("=" * 80)

    # ASCII bar chart
    for batch_size in config.batch_sizes:
        print(f"\nBatch size {batch_size}:")

        for world_size in config.world_sizes:
            ws_results = [
                r
                for r in results
                if r.batch_size == batch_size
                and r.world_size == world_size
                and r.success
            ]

            if ws_results:
                print(f"\n  World size {world_size}:")

                # Normalize to show memory usage
                max_mem = max(r.memory_mb for r in ws_results)

                for r in sorted(ws_results, key=lambda x: x.total_seq_len):
                    bar_len = int(50 * r.memory_mb / max_mem)
                    bar = "█" * bar_len + "░" * (50 - bar_len)
                    print(
                        f"    {r.total_seq_len:6,}: {bar} {r.memory_mb:6.1f}MB "
                        f"({r.local_seq_len:,}/GPU)"
                    )


def main():
    """Run comprehensive benchmarks."""
    print("Starting comprehensive ring attention benchmarks...")
    print("This will test the corrected O(n/k) memory implementation.")
    print()

    # Test 1: Extreme sequences
    print("Test 1: Extreme sequence lengths (200K+ tokens)")
    print("-" * 80)
    test_extreme_sequences()

    print("\n" + "=" * 80 + "\n")

    # Test 2: Memory scaling
    print("Test 2: Memory scaling verification")
    print("-" * 80)
    test_memory_scaling()


if __name__ == "__main__":
    main()
