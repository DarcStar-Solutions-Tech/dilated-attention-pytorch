#!/usr/bin/env python3
"""
Comprehensive benchmarks for kernel implementations.

This script benchmarks:
1. HilbertAttentionCore (Triton implementation)
2. HilbertAttentionSimple (PyTorch fallback)
3. HilbertAttentionTritonWrapper (Q,K,V interface)

Measures:
- Forward pass performance
- Backward pass performance
- Memory usage
- Scaling with sequence length
- Effect of Hilbert reordering
"""

import torch
import torch.nn as nn
import time
import gc
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import kernel implementations
try:
    from dilated_attention_pytorch.kernels import (
        HilbertAttentionCore,
        HilbertAttentionSimple,
        HilbertAttentionTritonWrapper,
        TRITON_AVAILABLE,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from dilated_attention_pytorch.kernels import (
        HilbertAttentionCore,
        HilbertAttentionSimple,
        HilbertAttentionTritonWrapper,
        TRITON_AVAILABLE,
    )

# Import benchmark utilities
from benchmarks.core.base_benchmark import BaseBenchmark
from benchmarks.core.utils.timing import CUDATimer
from benchmarks.core.utils.memory import MemoryProfiler
from benchmarks.core.utils.data import generate_random_tensor


@dataclass
class KernelBenchmarkConfig:
    """Configuration for kernel benchmarks."""

    batch_size: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    segment_size: int
    dilation_rate: int
    use_hilbert: bool
    dtype: torch.dtype = torch.float32
    device: str = "cuda"


class KernelBenchmark(BaseBenchmark):
    """Benchmark suite for kernel implementations."""

    def __init__(
        self, device: str = "cuda", warmup_steps: int = 10, measure_steps: int = 100
    ):
        super().__init__(name="Kernel Benchmarks", device=device)
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.cuda_timer = CUDATimer() if device == "cuda" else None
        self.memory_profiler = MemoryProfiler(device)

    def create_model(self, model_type: str, config: KernelBenchmarkConfig) -> nn.Module:
        """Create model based on type and config."""
        if model_type == "core":
            if not TRITON_AVAILABLE:
                raise RuntimeError("Triton not available for HilbertAttentionCore")
            return (
                HilbertAttentionCore(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    segment_size=config.segment_size,
                    dilation_rate=config.dilation_rate,
                    use_custom_backward=True,
                )
                .to(config.device)
                .to(config.dtype)
            )
        elif model_type == "simple":
            return (
                HilbertAttentionSimple(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    segment_size=config.segment_size,
                    dilation_rate=config.dilation_rate,
                    use_hilbert=config.use_hilbert,
                )
                .to(config.device)
                .to(config.dtype)
            )
        elif model_type == "wrapper":
            return (
                HilbertAttentionTritonWrapper(
                    segment_lengths=[config.segment_size],
                    dilation_rates=[config.dilation_rate],
                    num_heads=config.num_heads,
                    head_dim=config.hidden_dim // config.num_heads,
                )
                .to(config.device)
                .to(config.dtype)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def benchmark_forward(
        self, model: nn.Module, config: KernelBenchmarkConfig
    ) -> Dict[str, float]:
        """Benchmark forward pass."""
        # Generate input
        if isinstance(model, HilbertAttentionTritonWrapper):
            # Q,K,V interface
            head_dim = config.hidden_dim // config.num_heads
            q = generate_random_tensor(
                (config.batch_size, config.seq_len, config.num_heads, head_dim),
                device=config.device,
                dtype=config.dtype,
            )
            k = q.clone()
            v = q.clone()
            inputs = (q, k, v)
        else:
            # Standard interface
            x = generate_random_tensor(
                (config.batch_size, config.seq_len, config.hidden_dim),
                device=config.device,
                dtype=config.dtype,
            )
            inputs = (x,)

        # Warmup
        for _ in range(self.warmup_steps):
            with torch.no_grad():
                if isinstance(model, HilbertAttentionCore):
                    _ = model(*inputs, use_hilbert=config.use_hilbert)
                else:
                    _ = model(*inputs)

        # Measure
        torch.cuda.synchronize() if config.device == "cuda" else None

        if self.cuda_timer and config.device == "cuda":
            times = []
            for _ in range(self.measure_steps):
                with torch.no_grad():
                    elapsed = self.cuda_timer.time_function(
                        lambda: model(*inputs, use_hilbert=config.use_hilbert)
                        if isinstance(model, HilbertAttentionCore)
                        else model(*inputs)
                    )
                times.append(elapsed)
        else:
            start_time = time.perf_counter()
            for _ in range(self.measure_steps):
                with torch.no_grad():
                    if isinstance(model, HilbertAttentionCore):
                        _ = model(*inputs, use_hilbert=config.use_hilbert)
                    else:
                        _ = model(*inputs)
            end_time = time.perf_counter()
            times = [
                (end_time - start_time) / self.measure_steps * 1000
            ]  # Convert to ms

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
        }

    def benchmark_backward(
        self, model: nn.Module, config: KernelBenchmarkConfig
    ) -> Dict[str, float]:
        """Benchmark backward pass."""
        # Generate input
        if isinstance(model, HilbertAttentionTritonWrapper):
            head_dim = config.hidden_dim // config.num_heads
            q = generate_random_tensor(
                (config.batch_size, config.seq_len, config.num_heads, head_dim),
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
            k = q.clone().detach().requires_grad_(True)
            v = q.clone().detach().requires_grad_(True)
            inputs = (q, k, v)
        else:
            x = generate_random_tensor(
                (config.batch_size, config.seq_len, config.hidden_dim),
                device=config.device,
                dtype=config.dtype,
                requires_grad=True,
            )
            inputs = (x,)

        def forward_backward():
            # Zero gradients
            for inp in inputs:
                if inp.grad is not None:
                    inp.grad.zero_()

            # Forward
            if isinstance(model, HilbertAttentionCore):
                output = model(*inputs, use_hilbert=config.use_hilbert)
            else:
                output = model(*inputs)

            # Backward
            loss = output.sum()
            loss.backward()

        # Warmup
        for _ in range(self.warmup_steps):
            forward_backward()

        # Measure
        torch.cuda.synchronize() if config.device == "cuda" else None

        if self.cuda_timer and config.device == "cuda":
            times = []
            for _ in range(self.measure_steps):
                elapsed = self.cuda_timer.time_function(forward_backward)
                times.append(elapsed)
        else:
            start_time = time.perf_counter()
            for _ in range(self.measure_steps):
                forward_backward()
            end_time = time.perf_counter()
            times = [(end_time - start_time) / self.measure_steps * 1000]

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
        }

    def benchmark_memory(
        self, model: nn.Module, config: KernelBenchmarkConfig
    ) -> Dict[str, float]:
        """Benchmark memory usage."""
        if config.device != "cuda":
            return {"peak_memory_mb": 0, "allocated_memory_mb": 0}

        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Generate input
        if isinstance(model, HilbertAttentionTritonWrapper):
            head_dim = config.hidden_dim // config.num_heads
            q = generate_random_tensor(
                (config.batch_size, config.seq_len, config.num_heads, head_dim),
                device=config.device,
                dtype=config.dtype,
            )
            k = q.clone()
            v = q.clone()
            inputs = (q, k, v)
        else:
            x = generate_random_tensor(
                (config.batch_size, config.seq_len, config.hidden_dim),
                device=config.device,
                dtype=config.dtype,
            )
            inputs = (x,)

        # Forward pass
        with torch.no_grad():
            if isinstance(model, HilbertAttentionCore):
                _ = model(*inputs, use_hilbert=config.use_hilbert)
            else:
                _ = model(*inputs)

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # MB

        return {
            "peak_memory_mb": peak_memory,
            "allocated_memory_mb": allocated_memory,
        }

    def run_comparison(self, configs: List[KernelBenchmarkConfig]) -> pd.DataFrame:
        """Run comparison across different implementations and configurations."""
        results = []

        for config in configs:
            print(
                f"\nBenchmarking config: seq_len={config.seq_len}, hidden_dim={config.hidden_dim}, "
                f"segment_size={config.segment_size}, use_hilbert={config.use_hilbert}"
            )

            # Test each implementation
            implementations = []
            if config.device == "cuda" and TRITON_AVAILABLE:
                implementations.append(("core", "HilbertAttentionCore"))
            implementations.extend(
                [
                    ("simple", "HilbertAttentionSimple"),
                    ("wrapper", "HilbertAttentionTritonWrapper"),
                ]
            )

            for impl_type, impl_name in implementations:
                try:
                    # Skip wrapper for certain configs
                    if impl_type == "wrapper" and not config.use_hilbert:
                        continue  # Wrapper doesn't support use_hilbert flag

                    print(f"  Testing {impl_name}...")
                    model = self.create_model(impl_type, config)

                    # Benchmark forward
                    forward_stats = self.benchmark_forward(model, config)

                    # Benchmark backward
                    backward_stats = self.benchmark_backward(model, config)

                    # Benchmark memory
                    memory_stats = self.benchmark_memory(model, config)

                    # Combine results
                    result = {
                        "implementation": impl_name,
                        "seq_len": config.seq_len,
                        "hidden_dim": config.hidden_dim,
                        "num_heads": config.num_heads,
                        "segment_size": config.segment_size,
                        "dilation_rate": config.dilation_rate,
                        "use_hilbert": config.use_hilbert,
                        "batch_size": config.batch_size,
                        "forward_mean_ms": forward_stats["mean_ms"],
                        "forward_std_ms": forward_stats["std_ms"],
                        "backward_mean_ms": backward_stats["mean_ms"],
                        "backward_std_ms": backward_stats["std_ms"],
                        "peak_memory_mb": memory_stats["peak_memory_mb"],
                        "throughput_seq_per_sec": (config.batch_size * config.seq_len)
                        / (forward_stats["mean_ms"] / 1000),
                    }
                    results.append(result)

                    # Clean up
                    del model
                    torch.cuda.empty_cache() if config.device == "cuda" else None
                    gc.collect()

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

        return pd.DataFrame(results)

    def plot_results(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Plot benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Kernel Implementation Benchmarks", fontsize=16)

        # 1. Forward pass time vs sequence length
        ax = axes[0, 0]
        for impl in df["implementation"].unique():
            data = df[(df["implementation"] == impl) & df["use_hilbert"]]
            ax.plot(data["seq_len"], data["forward_mean_ms"], marker="o", label=impl)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Forward Time (ms)")
        ax.set_title("Forward Pass Performance")
        ax.legend()
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 2. Backward pass time vs sequence length
        ax = axes[0, 1]
        for impl in df["implementation"].unique():
            data = df[(df["implementation"] == impl) & df["use_hilbert"]]
            ax.plot(data["seq_len"], data["backward_mean_ms"], marker="s", label=impl)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Backward Time (ms)")
        ax.set_title("Backward Pass Performance")
        ax.legend()
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 3. Memory usage
        ax = axes[1, 0]
        for impl in df["implementation"].unique():
            data = df[(df["implementation"] == impl) & df["use_hilbert"]]
            ax.plot(data["seq_len"], data["peak_memory_mb"], marker="^", label=impl)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Memory Usage")
        ax.legend()
        ax.grid(True)
        ax.set_xscale("log")

        # 4. Throughput comparison
        ax = axes[1, 1]
        for impl in df["implementation"].unique():
            data = df[(df["implementation"] == impl) & df["use_hilbert"]]
            ax.plot(
                data["seq_len"], data["throughput_seq_per_sec"], marker="d", label=impl
            )
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Throughput (seq/sec)")
        ax.set_title("Processing Throughput")
        ax.legend()
        ax.grid(True)
        ax.set_xscale("log")
        ax.set_yscale("log")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def run_hilbert_comparison(self) -> pd.DataFrame:
        """Compare performance with and without Hilbert reordering."""
        configs = []

        for seq_len in [128, 256, 512, 1024]:
            for use_hilbert in [True, False]:
                configs.append(
                    KernelBenchmarkConfig(
                        batch_size=4,
                        seq_len=seq_len,
                        hidden_dim=512,
                        num_heads=8,
                        segment_size=128,
                        dilation_rate=1,
                        use_hilbert=use_hilbert,
                        device=self.device,
                    )
                )

        return self.run_comparison(configs)

    def run_scaling_benchmark(self) -> pd.DataFrame:
        """Benchmark scaling with sequence length."""
        configs = []

        seq_lengths = [128, 256, 512, 1024, 2048]
        if (
            self.device == "cuda"
            and torch.cuda.get_device_properties(0).total_memory > 16e9
        ):
            seq_lengths.extend([4096, 8192])

        for seq_len in seq_lengths:
            # Adjust segment size based on sequence length
            segment_size = min(128, seq_len // 4)
            segment_size = max(64, segment_size)  # Minimum for Triton

            configs.append(
                KernelBenchmarkConfig(
                    batch_size=2,
                    seq_len=seq_len,
                    hidden_dim=512,
                    num_heads=8,
                    segment_size=segment_size,
                    dilation_rate=1,
                    use_hilbert=True,
                    device=self.device,
                )
            )

        return self.run_comparison(configs)

    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 80)
        print("Kernel Implementation Comprehensive Benchmarks")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Triton Available: {TRITON_AVAILABLE}")

        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        # Run scaling benchmark
        print("\n1. Running scaling benchmark...")
        scaling_df = self.run_scaling_benchmark()

        # Run Hilbert comparison (only for Simple implementation)
        print("\n2. Running Hilbert reordering comparison...")
        hilbert_df = self.run_hilbert_comparison()

        # Save results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        scaling_df.to_csv(f"kernel_scaling_benchmark_{timestamp}.csv", index=False)
        hilbert_df.to_csv(f"kernel_hilbert_comparison_{timestamp}.csv", index=False)

        # Plot results
        self.plot_results(scaling_df, f"kernel_scaling_benchmark_{timestamp}.png")

        # Plot Hilbert comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        simple_data = hilbert_df[
            hilbert_df["implementation"] == "HilbertAttentionSimple"
        ]

        hilbert_on = simple_data[simple_data["use_hilbert"]]
        hilbert_off = simple_data[~simple_data["use_hilbert"]]

        x = hilbert_on["seq_len"].values
        speedup = (
            hilbert_off["forward_mean_ms"].values / hilbert_on["forward_mean_ms"].values
        )

        ax.plot(x, speedup, "o-", linewidth=2, markersize=8)
        ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Speedup (Hilbert ON / OFF)")
        ax.set_title("Hilbert Reordering Performance Impact")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        plt.tight_layout()
        plt.savefig(f"kernel_hilbert_speedup_{timestamp}.png", dpi=150)

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print("\nScaling Results (seq_len=1024, use_hilbert=True):")
        summary = scaling_df[scaling_df["seq_len"] == 1024].copy()
        summary = summary.sort_values("forward_mean_ms")

        for _, row in summary.iterrows():
            print(f"\n{row['implementation']}:")
            print(
                f"  Forward:  {row['forward_mean_ms']:.2f} ms (±{row['forward_std_ms']:.2f})"
            )
            print(
                f"  Backward: {row['backward_mean_ms']:.2f} ms (±{row['backward_std_ms']:.2f})"
            )
            print(f"  Memory:   {row['peak_memory_mb']:.1f} MB")
            print(f"  Throughput: {row['throughput_seq_per_sec']:.0f} seq/sec")

        print("\nHilbert Reordering Impact (HilbertAttentionSimple):")
        for seq_len in [128, 256, 512, 1024]:
            on = hilbert_df[
                (hilbert_df["seq_len"] == seq_len)
                & (hilbert_df["use_hilbert"])
                & (hilbert_df["implementation"] == "HilbertAttentionSimple")
            ]
            off = hilbert_df[
                (hilbert_df["seq_len"] == seq_len)
                & (~hilbert_df["use_hilbert"])
                & (hilbert_df["implementation"] == "HilbertAttentionSimple")
            ]

            if len(on) > 0 and len(off) > 0:
                speedup = off.iloc[0]["forward_mean_ms"] / on.iloc[0]["forward_mean_ms"]
                print(f"  seq_len={seq_len}: {speedup:.2f}x speedup")


def main():
    """Run benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark kernel implementations")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer iterations"
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create benchmark
    warmup = 5 if args.quick else 10
    measure = 20 if args.quick else 100

    benchmark = KernelBenchmark(
        device=args.device, warmup_steps=warmup, measure_steps=measure
    )

    # Run benchmarks
    benchmark.run_full_benchmark()


if __name__ == "__main__":
    main()
