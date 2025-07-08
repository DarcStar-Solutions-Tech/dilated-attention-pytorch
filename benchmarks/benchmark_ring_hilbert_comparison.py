#!/usr/bin/env python3
"""
Focused benchmark comparing Ring Attention with and without Hilbert ordering.

This script specifically tests the impact of Hilbert curve optimization
on Ring Dilated Attention performance.
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not installed. Using basic formatting.")

    def tabulate(data, headers, tablefmt="simple"):
        # Simple fallback
        result = " | ".join(headers) + "\n"
        result += "-" * (len(result) - 1) + "\n"
        for row in data:
            result += " | ".join(str(x) for x in row) + "\n"
        return result


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_proper import (
    RingDilatedAttentionHilbertProper,
)


def cleanup_memory():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_module(
    module: nn.Module,
    module_name: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    warmup: int = 3,
    iterations: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """Benchmark a single attention module."""

    print(f"\nBenchmarking {module_name}:")
    print(f"  Batch={batch_size}, Seq={seq_len}, Heads={num_heads}, HeadDim={head_dim}")

    # Create input tensors with shape [batch, seq_len, num_heads, head_dim]
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        cleanup_memory()
        with torch.amp.autocast(device_type="cuda", enabled=(dtype == torch.float16)):
            _ = module(q, k, v)
        torch.cuda.synchronize()

    # Measure forward pass
    print("  Measuring forward pass...")
    forward_times = []
    memory_peaks = []

    for i in range(iterations):
        cleanup_memory()

        # Record start
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.amp.autocast(device_type="cuda", enabled=(dtype == torch.float16)):
            _ = module(q, k, v)
        end_event.record()

        torch.cuda.synchronize()

        # Record time
        forward_times.append(start_event.elapsed_time(end_event))

        # Record memory
        memory_peaks.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB

    # Calculate statistics
    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    memory_mean = np.mean(memory_peaks)
    memory_std = np.std(memory_peaks)

    # Calculate throughput
    total_tokens = batch_size * seq_len
    throughput = total_tokens / (forward_mean / 1000)  # tokens/sec
    memory_per_token = (memory_mean * 1024) / total_tokens  # KB per token

    results = {
        "module_name": module_name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "forward_time_ms": forward_mean,
        "forward_std_ms": forward_std,
        "memory_mb": memory_mean,
        "memory_std_mb": memory_std,
        "throughput_tps": throughput,
        "memory_per_token_kb": memory_per_token,
    }

    print(f"  Forward: {forward_mean:.2f}±{forward_std:.2f} ms")
    print(f"  Memory: {memory_mean:.1f}±{memory_std:.1f} MB")
    print(f"  Throughput: {throughput:.0f} tokens/sec")

    return results


def run_comparison(
    seq_lengths: List[int],
    batch_sizes: List[int],
    num_heads_list: List[int],
    embed_dim: int = 768,
    segment_lengths: List[int] = [2048, 4096, 8192],
    dilation_rates: List[int] = [1, 2, 4],
    warmup: int = 3,
    iterations: int = 10,
    device: torch.device = None,
    dtype: torch.dtype = torch.float16,
) -> List[Dict[str, Any]]:
    """Run comparison benchmarks."""

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for seq_len in seq_lengths:
        # Adjust segment lengths for sequence
        adj_segment_lengths = [min(s, seq_len) for s in segment_lengths]

        for batch_size in batch_sizes:
            for num_heads in num_heads_list:
                head_dim = embed_dim // num_heads

                print(f"\n{'=' * 80}")
                print(
                    f"Configuration: seq_len={seq_len}, batch={batch_size}, heads={num_heads}"
                )
                print(
                    f"Segment lengths: {adj_segment_lengths}, dilation rates: {dilation_rates}"
                )
                print(f"{'=' * 80}")

                # Test Ring without Hilbert
                try:
                    module_no_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
                        dim=embed_dim,
                        heads=num_heads,
                        segment_lengths=adj_segment_lengths,
                        dilation_rates=dilation_rates,
                        use_hilbert=False,  # Disable Hilbert
                        dropout=0.0,
                    ).to(device, dtype)

                    result = benchmark_module(
                        module_no_hilbert,
                        "Ring (No Hilbert)",
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        warmup,
                        iterations,
                        device,
                        dtype,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Ring (No Hilbert) failed: {e}")

                # Test Ring with Hilbert
                try:
                    module_hilbert = RingDilatedAttentionHilbertOptimizedFixed(
                        dim=embed_dim,
                        heads=num_heads,
                        segment_lengths=adj_segment_lengths,
                        dilation_rates=dilation_rates,
                        use_hilbert=True,  # Enable Hilbert
                        dropout=0.0,
                    ).to(device, dtype)

                    result = benchmark_module(
                        module_hilbert,
                        "Ring (Hilbert)",
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        warmup,
                        iterations,
                        device,
                        dtype,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Ring (Hilbert) failed: {e}")

                # Test RingDilatedAttentionHilbertProper if available
                try:
                    module_proper = RingDilatedAttentionHilbertProper(
                        dim=embed_dim,
                        heads=num_heads,
                        segment_lengths=adj_segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                    ).to(device, dtype)

                    result = benchmark_module(
                        module_proper,
                        "Ring (Hilbert Proper)",
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        warmup,
                        iterations,
                        device,
                        dtype,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  Ring (Hilbert Proper) failed: {e}")

    return results


def generate_report(results: List[Dict[str, Any]], output_path: str):
    """Generate comparison report."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-UTC")
    report_path = output_path.replace(".md", f"-{timestamp}.md")

    # Ensure directory exists
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Ring Attention Hilbert Comparison Benchmark\n\n")
        f.write(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
        )

        # System info
        f.write("## System Information\n\n")
        if torch.cuda.is_available():
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"- CUDA Version: {torch.version.cuda}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        # Group results by configuration
        configs = {}
        for r in results:
            key = (r["seq_len"], r["batch_size"], r["num_heads"])
            if key not in configs:
                configs[key] = []
            configs[key].append(r)

        # Performance comparison tables
        f.write("## Performance Comparison\n\n")

        for (seq_len, batch_size, num_heads), config_results in sorted(configs.items()):
            f.write(f"### Seq={seq_len:,}, Batch={batch_size}, Heads={num_heads}\n\n")

            # Create comparison table
            table_data = []
            for r in config_results:
                table_data.append(
                    [
                        r["module_name"],
                        f"{r['forward_time_ms']:.2f}±{r['forward_std_ms']:.2f}",
                        f"{r['memory_mb']:.1f}±{r['memory_std_mb']:.1f}",
                        f"{r['throughput_tps']:.0f}",
                        f"{r['memory_per_token_kb']:.2f}",
                    ]
                )

            headers = [
                "Implementation",
                "Time (ms)",
                "Memory (MB)",
                "Throughput (tok/s)",
                "Mem/Token (KB)",
            ]
            f.write(tabulate(table_data, headers=headers, tablefmt="github"))
            f.write("\n\n")

            # Calculate speedup if both implementations exist
            no_hilbert = next(
                (r for r in config_results if "No Hilbert" in r["module_name"]), None
            )
            hilbert = next(
                (r for r in config_results if r["module_name"] == "Ring (Hilbert)"),
                None,
            )

            if no_hilbert and hilbert:
                speedup = no_hilbert["forward_time_ms"] / hilbert["forward_time_ms"]
                memory_ratio = hilbert["memory_mb"] / no_hilbert["memory_mb"]

                f.write("**Hilbert Impact:**\n")
                f.write(f"- Speedup: {speedup:.2f}x\n")
                f.write(f"- Memory ratio: {memory_ratio:.2f}x\n\n")

        # Summary analysis
        f.write("## Summary Analysis\n\n")

        # Calculate average speedups across all configurations
        speedups = []
        memory_ratios = []

        for config_results in configs.values():
            no_hilbert = next(
                (r for r in config_results if "No Hilbert" in r["module_name"]), None
            )
            hilbert = next(
                (r for r in config_results if r["module_name"] == "Ring (Hilbert)"),
                None,
            )

            if no_hilbert and hilbert:
                speedups.append(
                    no_hilbert["forward_time_ms"] / hilbert["forward_time_ms"]
                )
                memory_ratios.append(hilbert["memory_mb"] / no_hilbert["memory_mb"])

        if speedups:
            f.write("### Average Hilbert Impact\n\n")
            f.write(
                f"- Average speedup: {np.mean(speedups):.2f}x (±{np.std(speedups):.2f})\n"
            )
            f.write(
                f"- Average memory ratio: {np.mean(memory_ratios):.2f}x (±{np.std(memory_ratios):.2f})\n\n"
            )

        # Recommendations
        f.write("## Recommendations\n\n")

        if speedups and np.mean(speedups) > 1.1:
            f.write(
                "- ✅ **Hilbert ordering provides consistent performance benefits**\n"
            )
            f.write(
                f"- Average speedup of {np.mean(speedups):.2f}x justifies the additional complexity\n"
            )
        else:
            f.write("- ⚠️ **Hilbert ordering shows minimal performance benefits**\n")
            f.write("- Consider disabling Hilbert ordering to reduce complexity\n")

        if memory_ratios and np.mean(memory_ratios) > 1.2:
            f.write("- ⚠️ **Hilbert ordering increases memory usage**\n")
            f.write(
                f"- Average memory increase of {(np.mean(memory_ratios) - 1) * 100:.0f}%\n"
            )

        f.write("\n### Best Use Cases for Hilbert Ordering\n\n")
        f.write("Based on the benchmarks:\n")

        # Find configurations where Hilbert helps most
        best_speedups = []
        for (seq_len, batch_size, num_heads), config_results in configs.items():
            no_hilbert = next(
                (r for r in config_results if "No Hilbert" in r["module_name"]), None
            )
            hilbert = next(
                (r for r in config_results if r["module_name"] == "Ring (Hilbert)"),
                None,
            )

            if no_hilbert and hilbert:
                speedup = no_hilbert["forward_time_ms"] / hilbert["forward_time_ms"]
                if speedup > 1.1:
                    best_speedups.append((seq_len, batch_size, num_heads, speedup))

        if best_speedups:
            best_speedups.sort(key=lambda x: x[3], reverse=True)
            f.write("- **Most beneficial configurations:**\n")
            for seq_len, batch_size, num_heads, speedup in best_speedups[:5]:
                f.write(
                    f"  - Seq={seq_len:,}, Batch={batch_size}, Heads={num_heads}: {speedup:.2f}x speedup\n"
                )
        else:
            f.write(
                "- No configurations showed significant benefits from Hilbert ordering\n"
            )

    # Save raw results
    json_path = report_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport saved to: {report_path}")
    print(f"Raw data saved to: {json_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare Ring Attention with/without Hilbert"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2], help="Batch sizes to test"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[8, 16],
        help="Number of attention heads",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--segment-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        help="Segment lengths for dilated attention",
    )
    parser.add_argument(
        "--dilation-rates",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Dilation rates",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Benchmark iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/benchmarks/ring-hilbert-comparison.md",
        help="Output report path",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("Ring Attention Hilbert Comparison Benchmark")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of heads: {args.num_heads}")

    # Run comparison
    results = run_comparison(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        num_heads_list=args.num_heads,
        embed_dim=args.embed_dim,
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        dtype=dtype,
    )

    # Generate report
    generate_report(results, args.output)


if __name__ == "__main__":
    main()
