"""Base classes for benchmark framework."""

import argparse
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tabulate import tabulate

from .config import BenchmarkConfig
from .utils import get_peak_memory_mb, reset_gpu_memory, setup_device_and_dtype


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    implementation: str
    batch_size: int
    seq_length: int
    num_heads: int
    head_dim: int
    time_ms: float
    memory_mb: float
    throughput: float
    success: bool = True
    error: Optional[str] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "implementation": self.implementation,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "time_ms": self.time_ms,
            "memory_mb": self.memory_mb,
            "throughput": self.throughput,
            "success": self.success,
            "error": self.error,
            **self.extra_metrics,
        }


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        output_dir: str = "benchmark_results",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize benchmark.

        Args:
            config: Benchmark configuration
            output_dir: Directory to save results
            device: Device to run on (cuda/cpu)
            dtype: Data type to use
        """
        self.config = config or BenchmarkConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup device and dtype
        device_str = device or self.config.device
        use_fp16 = dtype == torch.float16 if dtype else self.config.use_fp16
        self.device, self.dtype = setup_device_and_dtype(device_str, use_fp16)

        self.results: List[BenchmarkResult] = []
        self.models: Dict[str, nn.Module] = {}

    @abstractmethod
    def setup_models(self) -> Dict[str, nn.Module]:
        """Setup models to benchmark. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_model_inputs(
        self, batch_size: int, seq_length: int, num_heads: int, head_dim: int
    ) -> Tuple[torch.Tensor, ...]:
        """Get input tensors for models. Must be implemented by subclasses."""
        pass

    def warmup(
        self, model: nn.Module, inputs: Tuple[torch.Tensor, ...], steps: int = 3
    ):
        """Warmup model to ensure stable measurements."""
        with torch.no_grad():
            for _ in range(steps):
                try:
                    _ = model(*inputs)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                except Exception:
                    # Some models might fail on certain inputs
                    pass

    def measure_performance(
        self,
        model: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        num_runs: int = 10,
    ) -> Tuple[float, float]:
        """Measure model performance.

        Returns:
            Tuple of (time_ms, memory_mb)
        """
        # Reset memory stats
        reset_gpu_memory(self.device)

        # Warmup
        self.warmup(model, inputs, steps=self.config.warmup_steps)

        # Time measurements
        times = []
        for _ in range(num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(*inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Get average time in milliseconds
        time_ms = (sum(times) / len(times)) * 1000

        # Get peak memory
        memory_mb = get_peak_memory_mb(self.device)

        return time_ms, memory_mb

    def benchmark_configuration(
        self,
        implementation_name: str,
        model: nn.Module,
        batch_size: int,
        seq_length: int,
        num_heads: int,
        head_dim: int,
    ) -> BenchmarkResult:
        """Benchmark a single configuration."""
        try:
            # Get inputs
            inputs = self.get_model_inputs(batch_size, seq_length, num_heads, head_dim)

            # Measure performance
            time_ms, memory_mb = self.measure_performance(
                model, inputs, num_runs=self.config.benchmark_steps
            )

            # Calculate throughput (tokens/second)
            total_tokens = batch_size * seq_length
            throughput = (total_tokens / time_ms) * 1000

            return BenchmarkResult(
                implementation=implementation_name,
                batch_size=batch_size,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=time_ms,
                memory_mb=memory_mb,
                throughput=throughput,
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                implementation=implementation_name,
                batch_size=batch_size,
                seq_length=seq_length,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=0.0,
                memory_mb=0.0,
                throughput=0.0,
                success=False,
                error=str(e),
            )

    def run(self):
        """Run complete benchmark suite."""
        print(f"Setting up models on {self.device} with dtype {self.dtype}")
        self.models = self.setup_models()

        total_configs = (
            len(self.models)
            * len(self.config.batch_sizes)
            * len(self.config.seq_lengths)
            * len(self.config.num_heads_list)
        )

        print(f"Running {total_configs} benchmark configurations...")

        for impl_name, model in self.models.items():
            print(f"\nBenchmarking {impl_name}...")

            for batch_size in self.config.batch_sizes:
                for seq_length in self.config.seq_lengths:
                    for num_heads in self.config.num_heads_list:
                        print(
                            f"  Config: batch={batch_size}, seq={seq_length}, "
                            f"heads={num_heads}, head_dim={self.config.head_dim}"
                        )

                        result = self.benchmark_configuration(
                            impl_name,
                            model,
                            batch_size,
                            seq_length,
                            num_heads,
                            self.config.head_dim,
                        )

                        self.results.append(result)

                        if result.success:
                            print(
                                f"    ✓ Time: {result.time_ms:.2f}ms, "
                                f"Memory: {result.memory_mb:.2f}MB, "
                                f"Throughput: {result.throughput:.0f} tokens/s"
                            )
                        else:
                            print(f"    ✗ Failed: {result.error}")

    def format_results(self) -> str:
        """Format results as a table."""
        headers = [
            "Implementation",
            "Batch",
            "Seq Len",
            "Heads",
            "Time (ms)",
            "Memory (MB)",
            "Throughput (tok/s)",
            "Status",
        ]

        rows = []
        for r in self.results:
            rows.append(
                [
                    r.implementation,
                    r.batch_size,
                    r.seq_length,
                    r.num_heads,
                    f"{r.time_ms:.2f}" if r.success else "N/A",
                    f"{r.memory_mb:.2f}" if r.success else "N/A",
                    f"{r.throughput:.0f}" if r.success else "N/A",
                    "✓" if r.success else "✗",
                ]
            )

        return tabulate(rows, headers=headers, tablefmt="grid")

    def save_results(self, prefix: Optional[str] = None):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = prefix or self.__class__.__name__.lower()

        # Save JSON
        json_path = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "device": str(self.device),
                        "dtype": str(self.dtype),
                        "benchmark_config": self.config.__dict__,
                    },
                    "results": [r.to_dict() for r in self.results],
                    "timestamp": timestamp,
                },
                f,
                indent=2,
            )

        # Save formatted table
        txt_path = self.output_dir / f"{prefix}_{timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write(f"Benchmark Results - {timestamp}\n")
            f.write(f"Device: {self.device}, Dtype: {self.dtype}\n\n")
            f.write(self.format_results())

        print("\nResults saved to:")
        print(f"  - {json_path}")
        print(f"  - {txt_path}")

    def analyze(self):
        """Analyze and summarize results."""
        if not self.results:
            print("No results to analyze")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by implementation
        impl_results = {}
        for r in self.results:
            if r.implementation not in impl_results:
                impl_results[r.implementation] = []
            impl_results[r.implementation].append(r)

        # Summary statistics
        for impl, results in impl_results.items():
            successful = [r for r in results if r.success]
            if successful:
                avg_time = sum(r.time_ms for r in successful) / len(successful)
                avg_memory = sum(r.memory_mb for r in successful) / len(successful)
                avg_throughput = sum(r.throughput for r in successful) / len(successful)
                success_rate = len(successful) / len(results) * 100

                print(f"\n{impl}:")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  Avg Time: {avg_time:.2f}ms")
                print(f"  Avg Memory: {avg_memory:.2f}MB")
                print(f"  Avg Throughput: {avg_throughput:.0f} tokens/s")
            else:
                print(f"\n{impl}: All tests failed")

    @classmethod
    def from_args(cls, **kwargs):
        """Create benchmark from command line arguments."""
        parser = argparse.ArgumentParser(description=cls.__doc__)

        # Add common arguments
        parser.add_argument("--device", default="cuda", help="Device to use")
        parser.add_argument(
            "--dtype", default="float16", choices=["float16", "float32", "bfloat16"]
        )
        parser.add_argument(
            "--output-dir", default="benchmark_results", help="Output directory"
        )
        parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
        parser.add_argument(
            "--seq-lengths", nargs="+", type=int, default=[1024, 2048, 4096]
        )
        parser.add_argument("--num-heads", nargs="+", type=int, default=[8, 16])
        parser.add_argument("--head-dim", type=int, default=64)
        parser.add_argument("--warmup-steps", type=int, default=3)
        parser.add_argument("--benchmark-steps", type=int, default=10)

        # Allow subclasses to add custom arguments
        if hasattr(cls, "add_arguments"):
            cls.add_arguments(parser)

        args = parser.parse_args()

        # Convert dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[args.dtype]

        # Create config
        config = BenchmarkConfig(
            batch_sizes=args.batch_sizes,
            seq_lengths=args.seq_lengths,
            num_heads_list=args.num_heads,
            head_dim=args.head_dim,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
            device=args.device,
            use_fp16=dtype == torch.float16,
        )

        # Merge with any provided kwargs
        kwargs.update(
            {
                "config": config,
                "output_dir": args.output_dir,
                "device": args.device,
                "dtype": dtype,
            }
        )

        return cls(**kwargs)
