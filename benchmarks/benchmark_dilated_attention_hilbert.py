#!/usr/bin/env python3
"""
Comprehensive benchmark for dilated attention implementations using Hilbert Triton kernel.

This benchmark tests:
1. Full dilated attention implementation (not just kernel in isolation)
2. Different configurations (segment lengths, dilation rates)
3. Realistic model sizes (transformer-like dimensions)
4. Both forward and backward pass performance
5. Comparison with standard PyTorch implementation
6. Memory efficiency at different sequence lengths
7. Proper warmup and timing methodology
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import core benchmark utilities
from benchmarks.core.base_benchmark import BaseBenchmark
from benchmarks.core.utils.timing import CUDATimer

# Import dilated attention implementations
from dilated_attention_pytorch.base.dilated_attention import DilatedAttention
from dilated_attention_pytorch.base.multihead_dilated_attention import (
    MultiheadDilatedAttention,
)
from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    HilbertAttentionCore,
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    # Model dimensions (transformer-like)
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None
    hidden_dims: List[int] = None
    num_heads_list: List[int] = None

    # Dilated attention configs
    segment_configs: List[Tuple[List[int], List[int]]] = None

    # Benchmark settings
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    test_backward: bool = True
    use_fp16: bool = True
    save_results: bool = True
    plot_results: bool = True

    def __post_init__(self):
        """Set default values if not provided."""
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]

        if self.seq_lengths is None:
            self.seq_lengths = [
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
            ]

        if self.hidden_dims is None:
            # Common transformer dimensions
            self.hidden_dims = [512, 768, 1024, 1536]

        if self.num_heads_list is None:
            self.num_heads_list = [8, 12, 16, 24]

        if self.segment_configs is None:
            self.segment_configs = [
                # (segment_lengths, dilation_rates)
                ([512], [1]),  # No dilation
                ([512, 1024], [1, 2]),  # 2-level dilation
                ([512, 1024, 2048], [1, 2, 4]),  # 3-level dilation
                ([1024, 2048, 4096], [1, 2, 4]),  # Larger segments
                ([2048, 4096, 8192], [1, 2, 4]),  # LongNet default
            ]


class StandardAttention(nn.Module):
    """Standard PyTorch attention for comparison."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class DilatedAttentionBenchmark(BaseBenchmark):
    """Benchmark for dilated attention implementations."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark with config."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.float16
            if config.use_fp16 and device.type == "cuda"
            else torch.float32
        )

        super().__init__(
            device=device,
            dtype=dtype,
            warmup_iterations=config.warmup_iterations,
            benchmark_iterations=config.benchmark_iterations,
        )

        self.config = config
        self.results = []

    def create_attention_module(
        self,
        implementation: str,
        hidden_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
    ) -> nn.Module:
        """Create attention module based on implementation type."""
        if implementation == "standard":
            return StandardAttention(hidden_dim, num_heads).to(self.device, self.dtype)

        elif implementation == "dilated":
            return DilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                softmax_scale=1.0 / (hidden_dim // num_heads) ** 0.5,
                attention_dropout=0.0,
            ).to(self.device, self.dtype)

        elif implementation == "multihead_dilated":
            return MultiheadDilatedAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                bias=False,
            ).to(self.device, self.dtype)

        elif implementation == "hilbert_triton":
            # Use HilbertAttentionCore with dilated segments
            return HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_lengths[0],  # Use first segment length
                dilation_rate=dilation_rates[0] if dilation_rates else 1,
                dropout=0.0,
                use_custom_backward=True,
            ).to(self.device, self.dtype)

        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def benchmark_single_config(
        self,
        implementation: str,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
    ) -> Dict:
        """Benchmark a single configuration."""
        # Skip if sequence length is not compatible with segment lengths
        if seq_len % max(segment_lengths) != 0:
            return None

        # Create module
        try:
            module = self.create_attention_module(
                implementation, hidden_dim, num_heads, segment_lengths, dilation_rates
            )
        except Exception as e:
            print(f"Failed to create {implementation} module: {e}")
            return None

        # Create input tensor
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device=self.device, dtype=self.dtype
        )
        x.requires_grad_(self.config.test_backward)

        # Timing function for forward pass
        def forward_fn():
            if implementation == "dilated":
                # DilatedAttention expects (batch, seq, heads, head_dim)
                B, N, D = x.shape
                H = num_heads
                head_dim = D // H
                x_reshaped = x.view(B, N, H, head_dim)
                out = module(x_reshaped, x_reshaped, x_reshaped)
                return out.view(B, N, D)
            elif implementation == "hilbert_triton":
                # HilbertAttentionCore has its own forward method
                return module(x, use_hilbert=True)
            else:
                # Standard and MultiheadDilatedAttention
                return module(x)

        # Timing function for forward + backward
        def forward_backward_fn():
            out = forward_fn()
            if self.config.test_backward:
                loss = out.sum()
                loss.backward()
            return out

        # Warmup
        for _ in range(self.warmup_iterations):
            self.cleanup_memory()
            _ = forward_fn()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark forward pass
        forward_times = []
        forward_memory = 0

        with CUDATimer() as timer:
            for _ in range(self.benchmark_iterations):
                self.cleanup_memory()
                start_mem = (
                    torch.cuda.memory_allocated() / 1024**2
                    if self.device.type == "cuda"
                    else 0
                )

                timer.start()
                out = forward_fn()
                timer.end()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    end_mem = torch.cuda.memory_allocated() / 1024**2
                    forward_memory = max(forward_memory, end_mem - start_mem)

                forward_times.append(timer.elapsed)

        # Benchmark backward pass if requested
        backward_times = []
        total_memory = 0

        if self.config.test_backward:
            with CUDATimer() as timer:
                for _ in range(self.benchmark_iterations):
                    self.cleanup_memory()
                    x_grad = x.detach().clone().requires_grad_(True)

                    if implementation == "dilated":
                        B, N, D = x_grad.shape
                        H = num_heads
                        head_dim = D // H
                        x_grad_reshaped = x_grad.view(B, N, H, head_dim)

                    start_mem = (
                        torch.cuda.memory_allocated() / 1024**2
                        if self.device.type == "cuda"
                        else 0
                    )

                    # Forward
                    if implementation == "dilated":
                        out = module(x_grad_reshaped, x_grad_reshaped, x_grad_reshaped)
                        out = out.view(B, N, D)
                    elif implementation == "hilbert_triton":
                        out = module(x_grad, use_hilbert=True)
                    else:
                        out = module(x_grad)

                    # Backward
                    timer.start()
                    loss = out.sum()
                    loss.backward()
                    timer.end()

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                        end_mem = torch.cuda.memory_allocated() / 1024**2
                        total_memory = max(total_memory, end_mem - start_mem)

                    backward_times.append(timer.elapsed)

        # Calculate statistics
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)
        backward_mean = np.mean(backward_times) if backward_times else 0
        backward_std = np.std(backward_times) if backward_times else 0

        result = {
            "implementation": implementation,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
            "forward_time_ms": forward_mean * 1000,
            "forward_std_ms": forward_std * 1000,
            "backward_time_ms": backward_mean * 1000,
            "backward_std_ms": backward_std * 1000,
            "total_time_ms": (forward_mean + backward_mean) * 1000,
            "forward_memory_mb": forward_memory,
            "total_memory_mb": total_memory if total_memory > 0 else forward_memory,
            "throughput_tokens_per_sec": (batch_size * seq_len)
            / (forward_mean + backward_mean),
        }

        return result

    def run_benchmarks(self) -> List[Dict]:
        """Run all benchmarks based on config."""
        implementations = ["standard", "multihead_dilated", "hilbert_triton", "dilated"]

        total_configs = (
            len(self.config.batch_sizes)
            * len(self.config.seq_lengths)
            * len(self.config.hidden_dims)
            * len(self.config.segment_configs)
            * len(implementations)
        )

        print(f"Running {total_configs} benchmark configurations...")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.dtype}")
        print()

        config_idx = 0
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                for hidden_dim in self.config.hidden_dims:
                    # Find matching num_heads for hidden_dim
                    num_heads = None
                    for nh in self.config.num_heads_list:
                        if hidden_dim % nh == 0:
                            num_heads = nh
                            break

                    if num_heads is None:
                        continue

                    for segment_lengths, dilation_rates in self.config.segment_configs:
                        # Skip if segments don't fit in sequence
                        if seq_len < max(segment_lengths):
                            continue

                        for impl in implementations:
                            config_idx += 1
                            print(
                                f"[{config_idx}/{total_configs}] "
                                f"{impl} - B:{batch_size}, N:{seq_len}, "
                                f"D:{hidden_dim}, H:{num_heads}, "
                                f"Segs:{segment_lengths}"
                            )

                            result = self.benchmark_single_config(
                                impl,
                                batch_size,
                                seq_len,
                                hidden_dim,
                                num_heads,
                                segment_lengths,
                                dilation_rates,
                            )

                            if result:
                                self.results.append(result)
                                print(
                                    f"  Forward: {result['forward_time_ms']:.2f}ms, "
                                    f"Backward: {result['backward_time_ms']:.2f}ms, "
                                    f"Memory: {result['total_memory_mb']:.1f}MB"
                                )
                            else:
                                print("  Skipped (incompatible config)")

        return self.results

    def plot_results(self):
        """Generate plots from benchmark results."""
        if not self.results:
            print("No results to plot")
            return

        # Convert results to pandas-like structure for easier plotting
        import pandas as pd

        df = pd.DataFrame(self.results)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Dilated Attention Benchmark Results", fontsize=16)

        # 1. Performance vs sequence length
        ax = axes[0, 0]
        for impl in df["implementation"].unique():
            impl_df = df[df["implementation"] == impl]
            seq_lens = impl_df.groupby("seq_len")["total_time_ms"].mean()
            ax.plot(seq_lens.index, seq_lens.values, marker="o", label=impl)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Performance vs Sequence Length")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

        # 2. Memory usage vs sequence length
        ax = axes[0, 1]
        for impl in df["implementation"].unique():
            impl_df = df[df["implementation"] == impl]
            seq_lens = impl_df.groupby("seq_len")["total_memory_mb"].mean()
            ax.plot(seq_lens.index, seq_lens.values, marker="o", label=impl)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("Memory Usage vs Sequence Length")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True)

        # 3. Throughput comparison
        ax = axes[0, 2]
        impl_throughput = df.groupby("implementation")[
            "throughput_tokens_per_sec"
        ].mean()
        ax.bar(impl_throughput.index, impl_throughput.values)
        ax.set_xlabel("Implementation")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Average Throughput")
        ax.grid(True, axis="y")

        # 4. Forward vs backward time
        ax = axes[1, 0]
        impl_forward = df.groupby("implementation")["forward_time_ms"].mean()
        impl_backward = df.groupby("implementation")["backward_time_ms"].mean()
        x = np.arange(len(impl_forward))
        width = 0.35
        ax.bar(x - width / 2, impl_forward.values, width, label="Forward")
        ax.bar(x + width / 2, impl_backward.values, width, label="Backward")
        ax.set_xlabel("Implementation")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Forward vs Backward Pass Time")
        ax.set_xticks(x)
        ax.set_xticklabels(impl_forward.index, rotation=45)
        ax.legend()
        ax.grid(True, axis="y")

        # 5. Performance by hidden dimension
        ax = axes[1, 1]
        for impl in df["implementation"].unique():
            impl_df = df[df["implementation"] == impl]
            hidden_dims = impl_df.groupby("hidden_dim")["total_time_ms"].mean()
            ax.plot(hidden_dims.index, hidden_dims.values, marker="o", label=impl)
        ax.set_xlabel("Hidden Dimension")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Performance vs Hidden Dimension")
        ax.legend()
        ax.grid(True)

        # 6. Speedup over standard attention
        ax = axes[1, 2]
        standard_times = df[df["implementation"] == "standard"].set_index(
            ["batch_size", "seq_len", "hidden_dim"]
        )["total_time_ms"]

        for impl in ["multihead_dilated", "hilbert_triton", "dilated"]:
            impl_df = df[df["implementation"] == impl].set_index(
                ["batch_size", "seq_len", "hidden_dim"]
            )
            speedups = []
            seq_lens = []

            for idx in impl_df.index:
                if idx in standard_times.index:
                    speedup = standard_times[idx] / impl_df.loc[idx, "total_time_ms"]
                    speedups.append(speedup)
                    seq_lens.append(idx[1])  # seq_len is second element

            if speedups:
                # Group by sequence length and average
                speedup_by_seq = {}
                for sl, sp in zip(seq_lens, speedups):
                    if sl not in speedup_by_seq:
                        speedup_by_seq[sl] = []
                    speedup_by_seq[sl].append(sp)

                seq_lens_sorted = sorted(speedup_by_seq.keys())
                avg_speedups = [np.mean(speedup_by_seq[sl]) for sl in seq_lens_sorted]

                ax.plot(seq_lens_sorted, avg_speedups, marker="o", label=impl)

        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Speedup")
        ax.set_title("Speedup over Standard Attention")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"dilated_attention_benchmark_{timestamp}.png", dpi=150)
        plt.show()

    def save_results(self, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dilated_attention_benchmark_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(
                {
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "device": str(self.device),
                        "dtype": str(self.dtype),
                        "config": {
                            "batch_sizes": self.config.batch_sizes,
                            "seq_lengths": self.config.seq_lengths,
                            "hidden_dims": self.config.hidden_dims,
                            "num_heads_list": self.config.num_heads_list,
                            "segment_configs": self.config.segment_configs,
                            "warmup_iterations": self.config.warmup_iterations,
                            "benchmark_iterations": self.config.benchmark_iterations,
                            "test_backward": self.config.test_backward,
                        },
                    },
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"Results saved to {filename}")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(
        description="Benchmark dilated attention with Hilbert Triton kernel"
    )

    # Model dimensions
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 768, 1024],
        help="Hidden dimensions to test",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        help="Number of attention heads",
    )

    # Benchmark settings
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--no-backward", action="store_true", help="Skip backward pass benchmark"
    )
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save results to file"
    )
    parser.add_argument("--no-plot", action="store_true", help="Don't generate plots")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        hidden_dims=args.hidden_dims,
        num_heads_list=args.num_heads,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        test_backward=not args.no_backward,
        use_fp16=not args.fp32,
        save_results=not args.no_save,
        plot_results=not args.no_plot,
    )

    # Run benchmark
    benchmark = DilatedAttentionBenchmark(config)
    results = benchmark.run_benchmarks()

    # Save and plot results
    if config.save_results:
        benchmark.save_results()

    if config.plot_results:
        benchmark.plot_results()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    if results:
        # Group by implementation
        impl_stats = {}
        for r in results:
            impl = r["implementation"]
            if impl not in impl_stats:
                impl_stats[impl] = {
                    "forward_times": [],
                    "backward_times": [],
                    "total_times": [],
                    "memory": [],
                    "throughput": [],
                }

            impl_stats[impl]["forward_times"].append(r["forward_time_ms"])
            impl_stats[impl]["backward_times"].append(r["backward_time_ms"])
            impl_stats[impl]["total_times"].append(r["total_time_ms"])
            impl_stats[impl]["memory"].append(r["total_memory_mb"])
            impl_stats[impl]["throughput"].append(r["throughput_tokens_per_sec"])

        # Print statistics
        for impl, stats in impl_stats.items():
            print(f"\n{impl.upper()}:")
            print(
                f"  Forward time:  {np.mean(stats['forward_times']):.2f} ± {np.std(stats['forward_times']):.2f} ms"
            )
            print(
                f"  Backward time: {np.mean(stats['backward_times']):.2f} ± {np.std(stats['backward_times']):.2f} ms"
            )
            print(
                f"  Total time:    {np.mean(stats['total_times']):.2f} ± {np.std(stats['total_times']):.2f} ms"
            )
            print(
                f"  Memory usage:  {np.mean(stats['memory']):.1f} ± {np.std(stats['memory']):.1f} MB"
            )
            print(
                f"  Throughput:    {np.mean(stats['throughput']):.0f} ± {np.std(stats['throughput']):.0f} tokens/sec"
            )


if __name__ == "__main__":
    main()
