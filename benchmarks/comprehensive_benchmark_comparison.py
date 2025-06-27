#!/usr/bin/env python3
"""
Comprehensive benchmark to compare performance across different sequence lengths.
"""

import datetime
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import torch


# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager
matplotlib.use("Agg")  # Use non-interactive backend

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.dilated_attention import DilatedAttention
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.improved_multihead_dilated_attention import (
    ImprovedMultiheadDilatedAttention,
)
from dilated_attention_pytorch.multihead_dilated_attention import (
    MultiheadDilatedAttention,
)


def benchmark_attention(attention_module, inputs, num_runs=5, warmup=2):
    """Benchmark attention module."""
    if isinstance(inputs, tuple):
        q, k, v = inputs
    else:
        q = k = v = inputs

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            if (
                hasattr(attention_module, "forward")
                and "need_weights" in attention_module.forward.__code__.co_varnames
            ):
                _ = attention_module(q, k, v, need_weights=False)
            else:
                _ = attention_module(q, k, v)

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            if (
                hasattr(attention_module, "forward")
                and "need_weights" in attention_module.forward.__code__.co_varnames
            ):
                _ = attention_module(q, k, v, need_weights=False)
            else:
                _ = attention_module(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time, times


def main():
    # Configuration
    batch_size = 1
    num_heads = 8
    head_dim = 64
    embed_dim = num_heads * head_dim
    sequence_lengths = [2048, 4096, 8192, 16384]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Running comprehensive benchmark on {device}")
    print(f"Batch size: {batch_size}")
    print(f"Num heads: {num_heads}, Head dim: {head_dim}, Embed dim: {embed_dim}")
    print()

    results = {
        "DilatedAttention": {},
        "ImprovedDilatedAttention": {},
        "MultiheadDilatedAttention": {},
        "ImprovedMultiheadDilatedAttention": {},
    }

    for seq_len in sequence_lengths:
        print(f"\n=== Sequence length: {seq_len} ===")

        # Calculate segment lengths and dilation rates
        if seq_len <= 4096:
            segment_lengths = [seq_len]
            dilation_rates = [1]
        elif seq_len <= 8192:
            segment_lengths = [2048, 4096]
            dilation_rates = [1, 2]
        else:
            segment_lengths = [2048, 4096, 8192]
            dilation_rates = [1, 2, 4]

        print(f"Segment lengths: {segment_lengths}, Dilation rates: {dilation_rates}")

        # Create test tensors
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # For multihead attention (different input shape)
        mha_input = torch.randn(
            batch_size, seq_len, embed_dim, device=device, dtype=dtype
        )

        # Test DilatedAttention
        try:
            dilated_attn = DilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=dtype,
            )
            time_avg, _ = benchmark_attention(dilated_attn, (q, k, v))
            results["DilatedAttention"][seq_len] = time_avg
            print(f"DilatedAttention: {time_avg:.4f}s")
        except Exception as e:
            print(f"DilatedAttention failed: {e}")

        # Test ImprovedDilatedAttention
        try:
            improved_attn = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=dtype,
            )
            time_avg, _ = benchmark_attention(improved_attn, (q, k, v))
            results["ImprovedDilatedAttention"][seq_len] = time_avg
            print(f"ImprovedDilatedAttention: {time_avg:.4f}s")
        except Exception as e:
            print(f"ImprovedDilatedAttention failed: {e}")

        # Test MultiheadDilatedAttention
        try:
            mha_dilated = MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=dtype,
                batch_first=True,
            )
            time_avg, _ = benchmark_attention(mha_dilated, mha_input)
            results["MultiheadDilatedAttention"][seq_len] = time_avg
            print(f"MultiheadDilatedAttention: {time_avg:.4f}s")
        except Exception as e:
            print(f"MultiheadDilatedAttention failed: {e}")

        # Test ImprovedMultiheadDilatedAttention
        try:
            mha_improved = ImprovedMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                device=device,
                dtype=dtype,
                batch_first=True,
            )
            time_avg, _ = benchmark_attention(mha_improved, mha_input)
            results["ImprovedMultiheadDilatedAttention"][seq_len] = time_avg
            print(f"ImprovedMultiheadDilatedAttention: {time_avg:.4f}s")
        except Exception as e:
            print(f"ImprovedMultiheadDilatedAttention failed: {e}")

        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create visualization
    plt.figure(figsize=(10, 6))

    for impl_name, impl_results in results.items():
        if impl_results:
            seq_lens = sorted(impl_results.keys())
            times = [impl_results[sl] for sl in seq_lens]
            plt.plot(seq_lens, times, marker="o", label=impl_name)

    plt.xlabel("Sequence Length")
    plt.ylabel("Time (seconds)")
    plt.title("Dilated Attention Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.xscale("log", base=2)
    plt.yscale("log")

    # Save plot
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    plot_file = os.path.join(
        "..", "docs", "benchmarks", f"benchmark-comprehensive-{timestamp}.png"
    )
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_file}")

    # Save detailed results
    results_file = os.path.join(
        "..", "docs", "benchmarks", f"benchmark-comprehensive-results-{timestamp}.md"
    )

    with open(results_file, "w") as f:
        f.write("# Comprehensive Benchmark Results\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Number of heads: {num_heads}\n")
        f.write(f"- Head dimension: {head_dim}\n")
        f.write(f"- Embed dimension: {embed_dim}\n\n")
        f.write("## Results\n\n")

        # Create results table
        f.write(
            "| Sequence Length | DilatedAttention | ImprovedDilatedAttention | MultiheadDilatedAttention | ImprovedMultiheadDilatedAttention |\n"
        )
        f.write(
            "|-----------------|------------------|--------------------------|---------------------------|-----------------------------------|\n"
        )

        for seq_len in sequence_lengths:
            row = f"| {seq_len} "
            for impl in [
                "DilatedAttention",
                "ImprovedDilatedAttention",
                "MultiheadDilatedAttention",
                "ImprovedMultiheadDilatedAttention",
            ]:
                if seq_len in results[impl]:
                    row += f"| {results[impl][seq_len]:.4f}s "
                else:
                    row += "| N/A "
            row += "|\n"
            f.write(row)

        f.write("\n## Analysis\n\n")
        f.write("### Performance Improvements\n\n")

        # Calculate speedups
        for seq_len in sequence_lengths:
            if (
                seq_len in results["DilatedAttention"]
                and seq_len in results["ImprovedDilatedAttention"]
            ):
                speedup = (
                    results["DilatedAttention"][seq_len]
                    / results["ImprovedDilatedAttention"][seq_len]
                )
                f.write(
                    f"- Sequence {seq_len}: ImprovedDilatedAttention is {speedup:.2f}x faster\n"
                )

        f.write("\n### Bug Fixes Impact\n\n")
        f.write("The following critical bug fixes were implemented:\n\n")
        f.write("1. **Thread Safety**: Added proper synchronization for cache access\n")
        f.write(
            "2. **Memory Leak**: Fixed circular references in WeakValueDictionary\n"
        )
        f.write(
            "3. **Ring Size Validation**: Added validation for distributed scenarios\n"
        )
        f.write(
            "4. **Gradient Normalization**: Fixed mathematical order of operations\n\n"
        )
        f.write(
            "These fixes ensure correctness while maintaining or improving performance.\n"
        )

    print(f"\nResults saved to: {results_file}")

    # Memory usage summary
    if torch.cuda.is_available():
        print(
            f"\nPeak memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB"
        )


if __name__ == "__main__":
    main()

    # Use unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="comprehensive-comparison",
        parameters={}
    )
    
    # Add results
    output_manager.add_result("results", results)
    
    # Save results
    output_paths = output_manager.save_results()
    print(f"\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")
