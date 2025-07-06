#!/usr/bin/env python3
"""
Detailed benchmark analysis for original DilatedAttention.
Includes memory profiling, scaling analysis, and performance metrics.
"""

import torch
import time
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

from dilated_attention_pytorch import DilatedAttention


class DilatedAttentionBenchmark:
    def __init__(self, device: torch.device):
        self.device = device
        self.results = []

    def profile_memory(self, func, *args, **kwargs):
        """Profile memory usage of a function."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Initial memory
            mem_before = torch.cuda.memory_allocated() / 1024**2

            # Run function
            result = func(*args, **kwargs)

            # Peak memory
            mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            mem_after = torch.cuda.memory_allocated() / 1024**2

            return result, {
                "before_mb": mem_before,
                "after_mb": mem_after,
                "peak_mb": mem_peak,
                "delta_mb": mem_peak - mem_before,
            }
        else:
            result = func(*args, **kwargs)
            return result, {"before_mb": 0, "after_mb": 0, "peak_mb": 0, "delta_mb": 0}

    def benchmark_single(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        num_runs: int = 10,
    ) -> Dict:
        """Benchmark a single configuration with detailed metrics."""

        # Create model
        model = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=0.0,
        ).to(self.device)

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Create inputs
        q = torch.randn(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Profile memory for model creation and first forward pass
        def first_forward():
            with torch.no_grad():
                return model(q, k, v)

        _, mem_stats = self.profile_memory(first_forward)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(q, k, v)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time multiple runs
        times = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(q, k, v)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Calculate statistics
        times_ms = np.array(times) * 1000

        # Theoretical FLOPs calculation (approximate)
        # Attention: 2 * seq_len * seq_len * embed_dim FLOPs
        embed_dim = num_heads * head_dim
        flops_per_attention = 2 * seq_len * seq_len * embed_dim
        total_flops = batch_size * flops_per_attention

        # Calculate effective bandwidth (approximate)
        # Memory accessed: Q, K, V, output, attention weights
        bytes_per_element = 2  # float16
        memory_accessed = batch_size * seq_len * embed_dim * 4 * bytes_per_element
        effective_bandwidth_gb = memory_accessed / (np.mean(times) * 1024**3)

        result = {
            "config": {
                "batch_size": batch_size,
                "seq_len": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "embed_dim": embed_dim,
                "segment_lengths": segment_lengths,
                "dilation_rates": dilation_rates,
                "num_segments": len(segment_lengths),
                "param_count": param_count,
            },
            "timing": {
                "mean_ms": float(np.mean(times_ms)),
                "std_ms": float(np.std(times_ms)),
                "min_ms": float(np.min(times_ms)),
                "max_ms": float(np.max(times_ms)),
                "median_ms": float(np.median(times_ms)),
                "p95_ms": float(np.percentile(times_ms, 95)),
            },
            "memory": mem_stats,
            "performance": {
                "sequences_per_second": float(1000 / np.mean(times_ms)),
                "tokens_per_second": float(
                    (batch_size * seq_len) * 1000 / np.mean(times_ms)
                ),
                "tflops": float(total_flops / (np.mean(times) * 1e12)),
                "effective_bandwidth_gb_s": float(effective_bandwidth_gb),
                "ms_per_token": float(np.mean(times_ms) / (batch_size * seq_len)),
            },
        }

        return result

    def run_scaling_analysis(self):
        """Analyze how performance scales with different parameters."""
        print("\n=== SCALING ANALYSIS ===\n")

        # Test 1: Sequence length scaling
        print("1. Sequence Length Scaling (fixed batch=1, heads=8, dim=64)")
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        seq_results = []

        for seq_len in seq_lengths:
            # Adjust segment lengths based on sequence length
            if seq_len <= 1024:
                segments = [seq_len // 4, seq_len // 2]
                dilations = [1, 2]
            else:
                segments = [seq_len // 4, seq_len // 2]
                dilations = [1, 2]

            try:
                result = self.benchmark_single(
                    1, seq_len, 8, 64, segments, dilations, num_runs=5
                )
                seq_results.append(result)
                print(
                    f"  seq_len={seq_len}: {result['timing']['mean_ms']:.1f}ms, "
                    f"{result['memory']['peak_mb']:.1f}MB"
                )
            except Exception as e:
                print(f"  seq_len={seq_len}: Failed - {str(e)}")

        # Test 2: Head count scaling
        print("\n2. Head Count Scaling (fixed batch=2, seq_len=2048, dim=64)")
        head_counts = [2, 4, 8, 16, 32]
        head_results = []

        for num_heads in head_counts:
            try:
                result = self.benchmark_single(
                    2, 2048, num_heads, 64, [512, 1024], [1, 2], num_runs=5
                )
                head_results.append(result)
                print(
                    f"  heads={num_heads}: {result['timing']['mean_ms']:.1f}ms, "
                    f"{result['memory']['peak_mb']:.1f}MB"
                )
            except Exception as e:
                print(f"  heads={num_heads}: Failed - {str(e)}")

        # Test 3: Batch size scaling
        print("\n3. Batch Size Scaling (fixed seq_len=2048, heads=8, dim=64)")
        batch_sizes = [1, 2, 4, 8]
        batch_results = []

        for batch_size in batch_sizes:
            try:
                result = self.benchmark_single(
                    batch_size, 2048, 8, 64, [512, 1024], [1, 2], num_runs=5
                )
                batch_results.append(result)
                print(
                    f"  batch={batch_size}: {result['timing']['mean_ms']:.1f}ms, "
                    f"{result['memory']['peak_mb']:.1f}MB"
                )
            except Exception as e:
                print(f"  batch={batch_size}: Failed - {str(e)}")

        return {
            "sequence_scaling": seq_results,
            "head_scaling": head_results,
            "batch_scaling": batch_results,
        }

    def create_visualizations(self, scaling_results: Dict, timestamp: str):
        """Create performance visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("DilatedAttention (Original) Performance Analysis", fontsize=16)

        # 1. Sequence length scaling
        ax = axes[0, 0]
        if scaling_results["sequence_scaling"]:
            seq_lens = [
                r["config"]["seq_len"] for r in scaling_results["sequence_scaling"]
            ]
            times = [
                r["timing"]["mean_ms"] for r in scaling_results["sequence_scaling"]
            ]
            ax.plot(seq_lens, times, "bo-", linewidth=2, markersize=8)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Sequence Length Scaling")
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")

        # 2. Memory usage
        ax = axes[0, 1]
        if scaling_results["sequence_scaling"]:
            seq_lens = [
                r["config"]["seq_len"] for r in scaling_results["sequence_scaling"]
            ]
            memory = [
                r["memory"]["peak_mb"] for r in scaling_results["sequence_scaling"]
            ]
            ax.plot(seq_lens, memory, "ro-", linewidth=2, markersize=8)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Peak Memory (MB)")
            ax.set_title("Memory Usage Scaling")
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)

        # 3. Head count scaling
        ax = axes[1, 0]
        if scaling_results["head_scaling"]:
            heads = [r["config"]["num_heads"] for r in scaling_results["head_scaling"]]
            times = [r["timing"]["mean_ms"] for r in scaling_results["head_scaling"]]
            ax.plot(heads, times, "go-", linewidth=2, markersize=8)
            ax.set_xlabel("Number of Heads")
            ax.set_ylabel("Time (ms)")
            ax.set_title("Head Count Scaling")
            ax.grid(True, alpha=0.3)

        # 4. Throughput
        ax = axes[1, 1]
        if scaling_results["sequence_scaling"]:
            seq_lens = [
                r["config"]["seq_len"] for r in scaling_results["sequence_scaling"]
            ]
            throughput = [
                r["performance"]["tokens_per_second"]
                for r in scaling_results["sequence_scaling"]
            ]
            ax.plot(seq_lens, throughput, "mo-", linewidth=2, markersize=8)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Tokens/Second")
            ax.set_title("Throughput vs Sequence Length")
            ax.grid(True, alpha=0.3)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")

        plt.tight_layout()
        filename = f"benchmarks/original_dilated_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        return filename


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    benchmark = DilatedAttentionBenchmark(device)

    # Run scaling analysis
    scaling_results = benchmark.run_scaling_analysis()

    # Create timestamp
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")

    # Save results
    results = {
        "metadata": {
            "implementation": "DilatedAttention (Original)",
            "timestamp": timestamp,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "N/A",
            "pytorch_version": torch.__version__,
        },
        "scaling_analysis": scaling_results,
    }

    filename = f"benchmarks/original_dilated_detailed_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {filename}")

    # Create visualizations
    plot_file = benchmark.create_visualizations(scaling_results, timestamp)
    print(f"Plots saved to {plot_file}")

    # Print summary
    print("\n=== PERFORMANCE SUMMARY ===")
    print("\nKey Findings:")

    # Analyze sequence scaling
    if scaling_results["sequence_scaling"]:
        seq_data = scaling_results["sequence_scaling"]
        seq_lens = [r["config"]["seq_len"] for r in seq_data]
        times = [r["timing"]["mean_ms"] for r in seq_data]

        # Calculate scaling factor
        if len(seq_lens) > 1:
            scaling_factor = np.log2(times[-1] / times[0]) / np.log2(
                seq_lens[-1] / seq_lens[0]
            )
            print(f"- Sequence length scaling: O(n^{scaling_factor:.2f})")

        # Best throughput
        best_throughput = max(r["performance"]["tokens_per_second"] for r in seq_data)
        best_seq = next(
            r["config"]["seq_len"]
            for r in seq_data
            if r["performance"]["tokens_per_second"] == best_throughput
        )
        print(
            f"- Best throughput: {best_throughput:,.0f} tokens/sec at seq_len={best_seq}"
        )

    print("\nMemory Efficiency:")
    if scaling_results["sequence_scaling"]:
        for r in scaling_results["sequence_scaling"][-3:]:  # Last 3 configs
            seq_len = r["config"]["seq_len"]
            mem_mb = r["memory"]["peak_mb"]
            mem_per_token = mem_mb * 1024 / seq_len  # KB per token
            print(f"- seq_len={seq_len}: {mem_mb:.1f}MB ({mem_per_token:.2f} KB/token)")


if __name__ == "__main__":
    main()
