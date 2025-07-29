#!/usr/bin/env python3
"""
Comprehensive benchmark for unified HilbertAttention implementation.

This benchmark tests the new consolidated HilbertAttention module which
automatically selects optimizations based on input parameters and hardware.

Tests:
- Standard vs Hilbert ordering performance
- Different sequence lengths and segment sizes
- Dilation rates impact
- Memory usage patterns
- Automatic optimization selection
- Forward and backward pass performance
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: Dict
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    memory_mb: float
    use_hilbert: bool
    actual_backend: str  # 'triton', 'pytorch', etc


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_attention(
    module: nn.Module,
    input_tensor: torch.Tensor,
    use_hilbert: bool,
    num_warmup: int = 3,
    num_iterations: int = 10,
    test_backward: bool = True,
) -> Tuple[float, float, float]:
    """Benchmark forward and backward passes."""

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(input_tensor, use_hilbert=use_hilbert)

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Time forward pass
    forward_times = []
    for _ in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            output = module(input_tensor, use_hilbert=use_hilbert)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_times.append((time.perf_counter() - start) * 1000)

    # Time backward pass if requested
    backward_times = []
    if test_backward:
        input_tensor.requires_grad = True

        for _ in range(num_iterations):
            output = module(input_tensor, use_hilbert=use_hilbert)
            loss = output.mean()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            loss.backward()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            backward_times.append((time.perf_counter() - start) * 1000)

            # Clear gradients
            input_tensor.grad = None
            module.zero_grad()

    # Remove outliers (top and bottom 10%)
    forward_times = sorted(forward_times)[
        len(forward_times) // 10 : -len(forward_times) // 10
    ]
    if backward_times:
        backward_times = sorted(backward_times)[
            len(backward_times) // 10 : -len(backward_times) // 10
        ]

    forward_ms = np.mean(forward_times)
    backward_ms = np.mean(backward_times) if backward_times else 0.0
    memory_mb = get_memory_usage()

    return forward_ms, backward_ms, memory_mb


def run_benchmark_suite(
    configs: List[Dict], output_dir: Optional[Path] = None
) -> List[BenchmarkResult]:
    """Run benchmarks for multiple configurations."""

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running benchmarks on {device}")
    print("=" * 80)

    for config in configs:
        print("\nTesting configuration:")
        print(
            f"  Batch: {config['batch_size']}, Seq: {config['seq_len']}, "
            f"Hidden: {config['hidden_dim']}, Heads: {config['num_heads']}"
        )
        print(
            f"  Segment: {config['segment_size']}, Dilation: {config['dilation_rate']}"
        )

        # Create module
        module = HilbertAttention(
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            segment_size=config["segment_size"],
            dilation_rate=config["dilation_rate"],
            dropout=0.0,
        )

        if device == "cuda":
            module = module.cuda()

        # Create input
        input_tensor = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["hidden_dim"],
            device=device,
            dtype=torch.float32,
        )

        # Benchmark with standard attention
        print("  Testing standard attention...", end="", flush=True)
        forward_std, backward_std, memory_std = benchmark_attention(
            module, input_tensor, use_hilbert=False
        )
        print(f" Forward: {forward_std:.2f}ms, Backward: {backward_std:.2f}ms")

        # Benchmark with Hilbert attention
        print("  Testing Hilbert attention...", end="", flush=True)
        forward_hilbert, backward_hilbert, memory_hilbert = benchmark_attention(
            module, input_tensor, use_hilbert=True
        )
        print(f" Forward: {forward_hilbert:.2f}ms, Backward: {backward_hilbert:.2f}ms")

        # Calculate speedup
        speedup_forward = forward_std / forward_hilbert
        speedup_backward = backward_std / backward_hilbert if backward_std > 0 else 0

        print(
            f"  Speedup: Forward {speedup_forward:.2f}x, Backward {speedup_backward:.2f}x"
        )

        # Store results
        results.append(
            BenchmarkResult(
                config=config,
                forward_time_ms=forward_std,
                backward_time_ms=backward_std,
                total_time_ms=forward_std + backward_std,
                memory_mb=memory_std,
                use_hilbert=False,
                actual_backend="pytorch",
            )
        )

        results.append(
            BenchmarkResult(
                config=config,
                forward_time_ms=forward_hilbert,
                backward_time_ms=backward_hilbert,
                total_time_ms=forward_hilbert + backward_hilbert,
                memory_mb=memory_hilbert,
                use_hilbert=True,
                actual_backend="triton" if module._triton_available else "pytorch",
            )
        )

    # Save results if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        results_data = []
        for r in results:
            results_data.append(
                {
                    "config": r.config,
                    "forward_time_ms": r.forward_time_ms,
                    "backward_time_ms": r.backward_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "memory_mb": r.memory_mb,
                    "use_hilbert": r.use_hilbert,
                    "actual_backend": r.actual_backend,
                }
            )

        with open(output_dir / f"hilbert_benchmark_{timestamp}.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Generate plots
        plot_results(results, output_dir / f"hilbert_benchmark_{timestamp}.png")

    return results


def plot_results(results: List[BenchmarkResult], output_path: Path):
    """Generate performance plots."""

    # Group results by configuration
    configs = {}
    for r in results:
        key = (r.config["seq_len"], r.config["segment_size"], r.config["dilation_rate"])
        if key not in configs:
            configs[key] = {"standard": None, "hilbert": None}

        if r.use_hilbert:
            configs[key]["hilbert"] = r
        else:
            configs[key]["standard"] = r

    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Forward pass speedup vs sequence length
    seq_lens = []
    speedups = []
    for key, data in sorted(configs.items()):
        if data["standard"] and data["hilbert"]:
            seq_lens.append(key[0])
            speedup = data["standard"].forward_time_ms / data["hilbert"].forward_time_ms
            speedups.append(speedup)

    ax1.plot(seq_lens, speedups, "o-", markersize=8)
    ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Forward Pass Speedup")
    ax1.set_title("Hilbert Attention Forward Pass Speedup")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory usage comparison
    standard_mem = []
    hilbert_mem = []
    labels = []
    for key, data in sorted(configs.items()):
        if data["standard"] and data["hilbert"]:
            standard_mem.append(data["standard"].memory_mb)
            hilbert_mem.append(data["hilbert"].memory_mb)
            labels.append(f"Seq={key[0]}")

    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width / 2, standard_mem, width, label="Standard")
    ax2.bar(x + width / 2, hilbert_mem, width, label="Hilbert")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Memory Usage (MB)")
    ax2.set_title("Memory Usage Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Impact of dilation rate
    dilation_impact = {}
    for key, data in configs.items():
        if data["standard"] and data["hilbert"]:
            dil_rate = key[2]
            if dil_rate not in dilation_impact:
                dilation_impact[dil_rate] = []
            speedup = data["standard"].forward_time_ms / data["hilbert"].forward_time_ms
            dilation_impact[dil_rate].append(speedup)

    dil_rates = sorted(dilation_impact.keys())
    avg_speedups = [np.mean(dilation_impact[d]) for d in dil_rates]

    ax3.bar(dil_rates, avg_speedups)
    ax3.axhline(y=1.0, color="r", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Dilation Rate")
    ax3.set_ylabel("Average Speedup")
    ax3.set_title("Impact of Dilation Rate on Performance")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Total time comparison
    for key, data in sorted(configs.items()):
        if data["standard"] and data["hilbert"]:
            seq_len = key[0]
            ax4.scatter(
                seq_len,
                data["standard"].total_time_ms,
                color="blue",
                s=100,
                alpha=0.6,
                label="Standard" if seq_len == min(seq_lens) else "",
            )
            ax4.scatter(
                seq_len,
                data["hilbert"].total_time_ms,
                color="orange",
                s=100,
                alpha=0.6,
                label="Hilbert" if seq_len == min(seq_lens) else "",
            )

    ax4.set_xlabel("Sequence Length")
    ax4.set_ylabel("Total Time (ms)")
    ax4.set_title("Total Execution Time (Forward + Backward)")
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Run the benchmark suite."""

    # Define test configurations
    configs = []

    # Test different sequence lengths
    for seq_len in [512, 1024, 2048]:
        configs.append(
            {
                "batch_size": 2,
                "seq_len": seq_len,
                "hidden_dim": 768,
                "num_heads": 12,
                "segment_size": 128,
                "dilation_rate": 1,
            }
        )

    # Test different dilation rates
    for dilation_rate in [2, 4]:
        configs.append(
            {
                "batch_size": 2,
                "seq_len": 1024,
                "hidden_dim": 768,
                "num_heads": 12,
                "segment_size": 128,
                "dilation_rate": dilation_rate,
            }
        )

    # Test different segment sizes
    for segment_size in [64, 256]:
        configs.append(
            {
                "batch_size": 2,
                "seq_len": 1024,
                "hidden_dim": 768,
                "num_heads": 12,
                "segment_size": segment_size,
                "dilation_rate": 2,
            }
        )

    # Run benchmarks
    output_dir = Path(__file__).parent / "results" / "hilbert"
    results = run_benchmark_suite(configs, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Calculate average speedups
    speedups = []
    for i in range(0, len(results), 2):
        if i + 1 < len(results):
            standard = results[i]
            hilbert = results[i + 1]
            if not standard.use_hilbert and hilbert.use_hilbert:
                speedup = standard.forward_time_ms / hilbert.forward_time_ms
                speedups.append(speedup)

    if speedups:
        print(f"Average forward pass speedup: {np.mean(speedups):.2f}x")
        print(f"Max forward pass speedup: {np.max(speedups):.2f}x")
        print(f"Min forward pass speedup: {np.min(speedups):.2f}x")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
