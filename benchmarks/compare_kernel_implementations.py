#!/usr/bin/env python3
"""
Direct performance comparison between kernel implementations.

This script provides a focused comparison of:
1. HilbertAttentionCore vs HilbertAttentionSimple
2. Effect of custom backward vs PyTorch autograd
3. Triton kernel vs PyTorch implementation
4. Performance across different hardware
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import platform

# Import implementations
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


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    implementation: str
    seq_len: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: float
    custom_backward: bool = False


class KernelComparison:
    """Compare kernel implementations head-to-head."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device)
        self.dtype = torch.float32  # Use float32 for fair comparison

        # Standard configuration
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.segment_size = 128
        self.dilation_rate = 1
        self.batch_size = 4

        # Timing parameters
        self.warmup_iters = 10
        self.benchmark_iters = 50

    def create_models(self) -> Dict[str, nn.Module]:
        """Create all model variants for testing."""
        models = {}

        # Simple implementation (always available)
        models["Simple"] = (
            HilbertAttentionSimple(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                segment_size=self.segment_size,
                dilation_rate=self.dilation_rate,
                use_hilbert=True,
            )
            .to(self.device)
            .to(self.dtype)
        )

        # Triton implementations (if available)
        if TRITON_AVAILABLE and self.device.type == "cuda":
            # Core with custom backward
            models["Core (Custom Backward)"] = (
                HilbertAttentionCore(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    segment_size=self.segment_size,
                    dilation_rate=self.dilation_rate,
                    use_custom_backward=True,
                )
                .to(self.device)
                .to(self.dtype)
            )

            # Core with PyTorch backward
            models["Core (PyTorch Backward)"] = (
                HilbertAttentionCore(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    segment_size=self.segment_size,
                    dilation_rate=self.dilation_rate,
                    use_custom_backward=False,
                )
                .to(self.device)
                .to(self.dtype)
            )

            # Wrapper implementation
            models["Wrapper"] = (
                HilbertAttentionTritonWrapper(
                    segment_lengths=[self.segment_size],
                    dilation_rates=[self.dilation_rate],
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                )
                .to(self.device)
                .to(self.dtype)
            )

        return models

    def benchmark_model(
        self, model: nn.Module, seq_len: int, model_name: str
    ) -> BenchmarkResult:
        """Benchmark a single model configuration."""
        # Prepare input
        if "Wrapper" in model_name:
            # Q,K,V interface
            q = torch.randn(
                self.batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            k = q.clone().detach().requires_grad_(True)
            v = q.clone().detach().requires_grad_(True)
            inputs = (q, k, v)
        else:
            # Standard interface
            x = torch.randn(
                self.batch_size,
                seq_len,
                self.hidden_dim,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            inputs = (x,)

        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Warmup
        for _ in range(self.warmup_iters):
            # Forward
            if "Core" in model_name:
                output = model(*inputs, use_hilbert=True)
            else:
                output = model(*inputs)

            # Backward
            loss = output.sum()
            loss.backward()

            # Zero gradients
            for inp in inputs:
                inp.grad.zero_()

        # Measure forward pass
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        forward_times = []
        for _ in range(self.benchmark_iters):
            start = time.perf_counter()

            with torch.no_grad():
                if "Core" in model_name:
                    output = model(*inputs, use_hilbert=True)
                else:
                    output = model(*inputs)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            forward_times.append((time.perf_counter() - start) * 1000)

        # Measure backward pass
        backward_times = []
        for _ in range(self.benchmark_iters):
            # Zero gradients
            for inp in inputs:
                if inp.grad is not None:
                    inp.grad.zero_()

            # Forward (no timing)
            if "Core" in model_name:
                output = model(*inputs, use_hilbert=True)
            else:
                output = model(*inputs)

            loss = output.sum()

            # Time backward
            start = time.perf_counter()
            loss.backward()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            backward_times.append((time.perf_counter() - start) * 1000)

        # Measure memory
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run once to measure memory
            if "Core" in model_name:
                output = model(*inputs, use_hilbert=True)
            else:
                output = model(*inputs)

            loss = output.sum()
            loss.backward()

            memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            memory_mb = 0

        # Clean up
        del inputs, output, loss
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return BenchmarkResult(
            implementation=model_name,
            seq_len=seq_len,
            forward_time_ms=np.median(forward_times),
            backward_time_ms=np.median(backward_times),
            total_time_ms=np.median(forward_times) + np.median(backward_times),
            memory_mb=memory_mb,
            custom_backward="Custom Backward" in model_name,
        )

    def run_comparison(self, seq_lengths: List[int]) -> List[BenchmarkResult]:
        """Run comparison across sequence lengths."""
        results = []
        models = self.create_models()

        print(f"\nRunning kernel comparison on {self.device}")
        print(
            f"Configurations: batch_size={self.batch_size}, hidden_dim={self.hidden_dim}, "
            f"num_heads={self.num_heads}, segment_size={self.segment_size}"
        )
        print("-" * 80)

        for seq_len in seq_lengths:
            print(f"\nSequence length: {seq_len}")

            for model_name, model in models.items():
                try:
                    result = self.benchmark_model(model, seq_len, model_name)
                    results.append(result)

                    print(
                        f"  {model_name:25s}: "
                        f"Forward: {result.forward_time_ms:6.2f} ms, "
                        f"Backward: {result.backward_time_ms:6.2f} ms, "
                        f"Memory: {result.memory_mb:6.1f} MB"
                    )

                except Exception as e:
                    print(f"  {model_name:25s}: Error - {e}")

        return results

    def plot_comparison(
        self, results: List[BenchmarkResult], save_path: Optional[str] = None
    ):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Kernel Implementation Comparison", fontsize=16)

        # Group results by implementation
        impl_results = {}
        for r in results:
            if r.implementation not in impl_results:
                impl_results[r.implementation] = []
            impl_results[r.implementation].append(r)

        # Sort results by seq_len
        for impl in impl_results:
            impl_results[impl].sort(key=lambda x: x.seq_len)

        # 1. Forward pass comparison
        ax = axes[0, 0]
        for impl, results_list in impl_results.items():
            seq_lens = [r.seq_len for r in results_list]
            forward_times = [r.forward_time_ms for r in results_list]
            ax.plot(
                seq_lens, forward_times, "o-", label=impl, linewidth=2, markersize=8
            )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Forward Pass Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 2. Backward pass comparison
        ax = axes[0, 1]
        for impl, results_list in impl_results.items():
            seq_lens = [r.seq_len for r in results_list]
            backward_times = [r.backward_time_ms for r in results_list]
            ax.plot(
                seq_lens, backward_times, "s-", label=impl, linewidth=2, markersize=8
            )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Backward Pass Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 3. Total time comparison
        ax = axes[1, 0]
        for impl, results_list in impl_results.items():
            seq_lens = [r.seq_len for r in results_list]
            total_times = [r.total_time_ms for r in results_list]
            ax.plot(seq_lens, total_times, "^-", label=impl, linewidth=2, markersize=8)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Total Time (Forward + Backward)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # 4. Memory usage
        ax = axes[1, 1]
        if self.device.type == "cuda":
            for impl, results_list in impl_results.items():
                seq_lens = [r.seq_len for r in results_list]
                memory = [r.memory_mb for r in results_list]
                ax.plot(seq_lens, memory, "d-", label=impl, linewidth=2, markersize=8)

            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Memory (MB)")
            ax.set_title("Peak Memory Usage")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log")
        else:
            ax.text(
                0.5,
                0.5,
                "Memory profiling\nnot available on CPU",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved plot to {save_path}")
        else:
            plt.show()

    def print_speedup_summary(self, results: List[BenchmarkResult]):
        """Print speedup summary comparing implementations."""
        print("\n" + "=" * 80)
        print("SPEEDUP SUMMARY")
        print("=" * 80)

        # Group by sequence length
        seq_len_results = {}
        for r in results:
            if r.seq_len not in seq_len_results:
                seq_len_results[r.seq_len] = {}
            seq_len_results[r.seq_len][r.implementation] = r

        # Find baseline (Simple implementation)
        for seq_len in sorted(seq_len_results.keys()):
            results_at_len = seq_len_results[seq_len]

            if "Simple" in results_at_len:
                baseline = results_at_len["Simple"]
                print(f"\nSequence Length: {seq_len}")
                print(f"Baseline: {baseline.implementation}")
                print("-" * 50)

                for impl_name, result in results_at_len.items():
                    if impl_name != "Simple":
                        forward_speedup = (
                            baseline.forward_time_ms / result.forward_time_ms
                        )
                        backward_speedup = (
                            baseline.backward_time_ms / result.backward_time_ms
                        )
                        total_speedup = baseline.total_time_ms / result.total_time_ms

                        print(f"{impl_name:30s}:")
                        print(f"  Forward:  {forward_speedup:5.2f}x faster")
                        print(f"  Backward: {backward_speedup:5.2f}x faster")
                        print(f"  Total:    {total_speedup:5.2f}x faster")

                        if self.device.type == "cuda":
                            memory_ratio = result.memory_mb / baseline.memory_mb
                            print(f"  Memory:   {memory_ratio:5.2f}x usage")

    def run_hardware_comparison(self):
        """Compare performance characteristics across hardware."""
        print("\n" + "=" * 80)
        print("HARDWARE INFORMATION")
        print("=" * 80)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Processor: {platform.processor()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")

        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA: {torch.version.cuda}")
            print(f"cuDNN: {torch.backends.cudnn.version()}")
            print(
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

            # Check compute capability
            major, minor = torch.cuda.get_device_capability()
            print(f"Compute Capability: {major}.{minor}")

            # Suggest optimal configuration
            if major >= 8:  # Ampere or newer
                print(
                    "\nOptimal configuration: Use Triton kernels with custom backward"
                )
            elif major >= 7:  # Volta/Turing
                print(
                    "\nOptimal configuration: Use Triton kernels, benchmark custom vs PyTorch backward"
                )
            else:
                print("\nOptimal configuration: Consider using PyTorch implementation")


def main():
    """Run kernel comparison benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare kernel implementations")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--save-plot", type=str, default=None, help="Path to save comparison plot"
    )

    args = parser.parse_args()

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    # Create comparison object
    comparison = KernelComparison(device=args.device)

    # Run hardware comparison
    comparison.run_hardware_comparison()

    # Run benchmarks
    results = comparison.run_comparison(args.seq_lengths)

    # Plot results
    if results:
        comparison.plot_comparison(results, args.save_plot)
        comparison.print_speedup_summary(results)

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if TRITON_AVAILABLE and args.device == "cuda":
        print("1. For training: Use HilbertAttentionCore with custom_backward=True")
        print("2. For inference: Use HilbertAttentionCore with custom_backward=False")
        print(
            "3. For compatibility: Use HilbertAttentionTritonWrapper for Q,K,V interface"
        )
    else:
        print("1. Triton not available - using PyTorch implementation")
        print("2. Consider installing Triton for better performance on CUDA")
        print("3. HilbertAttentionSimple provides good fallback performance")


if __name__ == "__main__":
    main()
