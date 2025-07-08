#!/usr/bin/env python3
"""
Comprehensive benchmarks for Ring Dilated Attention with Hilbert implementation.

This script benchmarks:
- Different sequence lengths (1K to 64K tokens)
- Different batch sizes and head counts
- With and without Hilbert ordering
- Different ring sizes (for multi-GPU)
- Comparison against standard attention and regular ring attention
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

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

from dilated_attention_pytorch import (
    MultiheadDilatedAttention,
)
from dilated_attention_pytorch.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)
from core.base_benchmark import BaseBenchmark
from core.utils.memory import get_memory_stats, cleanup_memory
from core.utils.timing import CUDATimer


class HilbertRingBenchmark(BaseBenchmark):
    """Comprehensive benchmark for Ring Hilbert attention."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        super().__init__(device, dtype, warmup_iterations, benchmark_iterations)
        self.results = []

    def run(self) -> Dict[str, Any]:
        """Run method to satisfy abstract base class."""
        # This is handled by run_comprehensive_benchmark
        return {"status": "Use run_comprehensive_benchmark instead"}

    def create_attention_module(
        self,
        module_type: str,
        embed_dim: int,
        num_heads: int,
        segment_lengths: List[int],
        dilation_rates: List[int],
        ring_size: int = 1,
        use_hilbert: bool = True,
    ) -> nn.Module:
        """Create an attention module based on type."""

        if module_type == "standard":
            # Standard PyTorch MultiheadAttention (for small sequences)
            return nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
                device=self.device,
                dtype=self.dtype,
            )
        elif module_type == "dilated":
            # Regular dilated attention
            return MultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=self.device,
                dtype=self.dtype,
            )
        elif module_type == "ring":
            # Ring attention without Hilbert
            return RingDistributedDilatedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                dropout=0.0,
            ).to(self.device, self.dtype)
        elif module_type == "ring_hilbert":
            # Ring attention with Hilbert
            return RingDilatedAttentionHilbertOptimizedFixed(
                dim=embed_dim,
                heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                dropout=0.0,
                use_hilbert=use_hilbert,
            ).to(self.device, self.dtype)
        else:
            raise ValueError(f"Unknown module type: {module_type}")

    def benchmark_attention(
        self,
        module: nn.Module,
        module_name: str,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
        measure_backward: bool = True,
    ) -> Dict[str, float]:
        """Benchmark a single attention module."""

        print(
            f"\nBenchmarking {module_name}: B={batch_size}, L={seq_len}, H={num_heads}"
        )

        # Create input tensors
        x = torch.randn(
            batch_size,
            seq_len,
            embed_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )

        # Warmup
        print("  Warming up...")
        for _ in range(self.warmup_iterations):
            cleanup_memory()
            with torch.amp.autocast(
                device_type="cuda", enabled=(self.dtype == torch.float16)
            ):
                if isinstance(module, nn.MultiheadAttention):
                    output, _ = module(x, x, x, need_weights=False)
                elif isinstance(
                    module,
                    (
                        RingDilatedAttentionHilbertOptimizedFixed,
                        RingDistributedDilatedAttention,
                    ),
                ):
                    # These modules expect q, k, v as separate arguments
                    if isinstance(module, RingDilatedAttentionHilbertOptimizedFixed):
                        # Shape: [batch, seq_len, num_heads, head_dim]
                        head_dim = embed_dim // num_heads
                        x_reshaped = x.view(batch_size, seq_len, num_heads, head_dim)
                        output = module(x_reshaped, x_reshaped, x_reshaped)
                        output = output.view(batch_size, seq_len, embed_dim)
                    else:
                        # RingDistributedDilatedAttention expects [batch, seq_len, embed_dim]
                        result = module(x, x, x)
                        # Handle tuple output from RingDistributedDilatedAttention
                        output = result[0] if isinstance(result, tuple) else result
                else:
                    output = module(x)
                if measure_backward and x.requires_grad:
                    loss = output.mean()
                    loss.backward()

        # Measure forward pass
        print("  Measuring forward pass...")
        forward_times = []
        forward_memory_peak = 0

        for i in range(self.benchmark_iterations):
            cleanup_memory()
            timer = CUDATimer("", self.device, verbose=False)

            with timer:
                with torch.amp.autocast(
                    device_type="cuda", enabled=(self.dtype == torch.float16)
                ):
                    if isinstance(module, nn.MultiheadAttention):
                        output, _ = module(x, x, x, need_weights=False)
                    elif isinstance(
                        module,
                        (
                            RingDilatedAttentionHilbertOptimizedFixed,
                            RingDistributedDilatedAttention,
                        ),
                    ):
                        # These modules expect q, k, v as separate arguments
                        if isinstance(
                            module, RingDilatedAttentionHilbertOptimizedFixed
                        ):
                            # Shape: [batch, seq_len, num_heads, head_dim]
                            head_dim = embed_dim // num_heads
                            x_reshaped = x.view(
                                batch_size, seq_len, num_heads, head_dim
                            )
                            output = module(x_reshaped, x_reshaped, x_reshaped)
                            output = output.view(batch_size, seq_len, embed_dim)
                        else:
                            # RingDistributedDilatedAttention expects [batch, seq_len, embed_dim]
                            output = module(x, x, x)
                    else:
                        output = module(x)

            forward_times.append(timer.elapsed_ms)
            memory_stats = get_memory_stats(self.device)
            forward_memory_peak = max(forward_memory_peak, memory_stats["allocated"])

        # Measure backward pass
        backward_times = []
        total_memory_peak = forward_memory_peak

        if measure_backward and x.requires_grad:
            print("  Measuring backward pass...")
            for i in range(self.benchmark_iterations):
                cleanup_memory()

                # Forward pass
                with torch.amp.autocast(
                    device_type="cuda", enabled=(self.dtype == torch.float16)
                ):
                    if isinstance(module, nn.MultiheadAttention):
                        output, _ = module(x, x, x, need_weights=False)
                    elif isinstance(
                        module,
                        (
                            RingDilatedAttentionHilbertOptimizedFixed,
                            RingDistributedDilatedAttention,
                        ),
                    ):
                        # These modules expect q, k, v as separate arguments
                        if isinstance(
                            module, RingDilatedAttentionHilbertOptimizedFixed
                        ):
                            # Shape: [batch, seq_len, num_heads, head_dim]
                            head_dim = embed_dim // num_heads
                            x_reshaped = x.view(
                                batch_size, seq_len, num_heads, head_dim
                            )
                            output = module(x_reshaped, x_reshaped, x_reshaped)
                            output = output.view(batch_size, seq_len, embed_dim)
                        else:
                            # RingDistributedDilatedAttention expects [batch, seq_len, embed_dim]
                            output = module(x, x, x)
                    else:
                        output = module(x)

                # Backward pass
                timer = CUDATimer("", self.device, verbose=False)
                with timer:
                    loss = output.mean()
                    loss.backward()

                backward_times.append(timer.elapsed_ms)
                memory_stats = get_memory_stats(self.device)
                total_memory_peak = max(total_memory_peak, memory_stats["allocated"])

        # Calculate statistics
        forward_mean = np.mean(forward_times)
        forward_std = np.std(forward_times)

        if backward_times:
            backward_mean = np.mean(backward_times)
            backward_std = np.std(backward_times)
            total_time = forward_mean + backward_mean
        else:
            backward_mean = 0
            backward_std = 0
            total_time = forward_mean

        # Calculate throughput
        total_tokens = batch_size * seq_len
        throughput_forward = total_tokens / (forward_mean / 1000)  # tokens/sec
        throughput_total = total_tokens / (total_time / 1000) if total_time > 0 else 0

        # Memory efficiency
        memory_per_token = (total_memory_peak * 1024) / total_tokens  # KB per token

        return {
            "module_name": module_name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "forward_time_ms": forward_mean,
            "forward_std_ms": forward_std,
            "backward_time_ms": backward_mean,
            "backward_std_ms": backward_std,
            "total_time_ms": total_time,
            "peak_memory_mb": total_memory_peak,
            "throughput_forward_tps": throughput_forward,
            "throughput_total_tps": throughput_total,
            "memory_per_token_kb": memory_per_token,
        }

    def run_comprehensive_benchmark(
        self,
        seq_lengths: List[int],
        batch_sizes: List[int],
        num_heads_list: List[int],
        embed_dim: int = 768,
        ring_sizes: List[int] = [1],
        include_standard: bool = True,
        max_standard_seq: int = 4096,
    ) -> List[Dict[str, Any]]:
        """Run comprehensive benchmarks across all configurations."""

        results = []

        # Determine segment lengths based on sequence length
        def get_segment_config(seq_len: int) -> Tuple[List[int], List[int]]:
            if seq_len <= 4096:
                return [1024, 2048], [1, 2]
            elif seq_len <= 16384:
                return [2048, 4096, 8192], [1, 2, 4]
            else:
                return [4096, 8192, 16384], [1, 2, 4]

        for seq_len in seq_lengths:
            segment_lengths, dilation_rates = get_segment_config(seq_len)

            for batch_size in batch_sizes:
                for num_heads in num_heads_list:
                    print(f"\n{'=' * 80}")
                    print(
                        f"Configuration: seq_len={seq_len}, batch={batch_size}, heads={num_heads}"
                    )
                    print(f"{'=' * 80}")

                    # Test standard attention (if sequence is small enough)
                    if include_standard and seq_len <= max_standard_seq:
                        try:
                            module = self.create_attention_module(
                                "standard",
                                embed_dim,
                                num_heads,
                                segment_lengths,
                                dilation_rates,
                            )
                            result = self.benchmark_attention(
                                module,
                                "Standard MHA",
                                batch_size,
                                seq_len,
                                embed_dim,
                                num_heads,
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"  Standard attention failed: {e}")

                    # Test regular dilated attention
                    try:
                        module = self.create_attention_module(
                            "dilated",
                            embed_dim,
                            num_heads,
                            segment_lengths,
                            dilation_rates,
                        )
                        result = self.benchmark_attention(
                            module,
                            "Dilated Attention",
                            batch_size,
                            seq_len,
                            embed_dim,
                            num_heads,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  Dilated attention failed: {e}")

                    # Test ring attention with different ring sizes
                    for ring_size in ring_sizes:
                        # Ring without Hilbert
                        try:
                            module = self.create_attention_module(
                                "ring",
                                embed_dim,
                                num_heads,
                                segment_lengths,
                                dilation_rates,
                                ring_size,
                            )
                            result = self.benchmark_attention(
                                module,
                                f"Ring (size={ring_size})",
                                batch_size,
                                seq_len,
                                embed_dim,
                                num_heads,
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"  Ring attention failed: {e}")

                        # Ring with Hilbert
                        try:
                            module = self.create_attention_module(
                                "ring_hilbert",
                                embed_dim,
                                num_heads,
                                segment_lengths,
                                dilation_rates,
                                ring_size,
                                True,
                            )
                            result = self.benchmark_attention(
                                module,
                                f"Ring+Hilbert (size={ring_size})",
                                batch_size,
                                seq_len,
                                embed_dim,
                                num_heads,
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"  Ring+Hilbert attention failed: {e}")

        return results

    def generate_report(self, results: List[Dict[str, Any]], output_path: str):
        """Generate a comprehensive benchmark report."""

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-UTC")
        # Handle relative paths from benchmarks directory
        if output_path.startswith("docs/"):
            report_path = os.path.join(
                "..", output_path.replace(".md", f"-{timestamp}.md")
            )
        else:
            report_path = output_path.replace(".md", f"-{timestamp}.md")

        with open(report_path, "w") as f:
            f.write(
                "# Ring Dilated Attention with Hilbert - Comprehensive Benchmark Report\n\n"
            )
            f.write(
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            )

            # System information
            f.write("## System Information\n\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Data Type: {self.dtype}\n")
            if self.device.type == "cuda":
                f.write(f"- GPU: {torch.cuda.get_device_name(self.device)}\n")
                f.write(f"- CUDA Version: {torch.version.cuda}\n")
            f.write(f"- PyTorch Version: {torch.__version__}\n\n")

            # Performance Summary Tables
            f.write("## Performance Summary\n\n")

            # Group results by sequence length
            seq_lengths = sorted(set(r["seq_len"] for r in results))

            for seq_len in seq_lengths:
                f.write(f"### Sequence Length: {seq_len:,} tokens\n\n")

                seq_results = [r for r in results if r["seq_len"] == seq_len]

                # Create performance table
                table_data = []
                for r in seq_results:
                    table_data.append(
                        [
                            r["module_name"],
                            r["batch_size"],
                            r["num_heads"],
                            f"{r['forward_time_ms']:.2f}±{r['forward_std_ms']:.2f}",
                            f"{r['backward_time_ms']:.2f}±{r['backward_std_ms']:.2f}",
                            f"{r['total_time_ms']:.2f}",
                            f"{r['peak_memory_mb']:.1f}",
                            f"{r['throughput_total_tps']:.0f}",
                            f"{r['memory_per_token_kb']:.2f}",
                        ]
                    )

                headers = [
                    "Module",
                    "Batch",
                    "Heads",
                    "Fwd (ms)",
                    "Bwd (ms)",
                    "Total (ms)",
                    "Peak Mem (MB)",
                    "Throughput (tok/s)",
                    "Mem/Token (KB)",
                ]

                f.write(tabulate(table_data, headers=headers, tablefmt="github"))
                f.write("\n\n")

            # Scaling Analysis
            f.write("## Scaling Analysis\n\n")

            # Analyze scaling with sequence length
            f.write("### Sequence Length Scaling\n\n")
            modules = sorted(set(r["module_name"] for r in results))

            for module in modules:
                f.write(f"#### {module}\n\n")
                module_results = [r for r in results if r["module_name"] == module]

                # Group by batch size and heads
                configs = sorted(
                    set((r["batch_size"], r["num_heads"]) for r in module_results)
                )

                for batch_size, num_heads in configs:
                    config_results = [
                        r
                        for r in module_results
                        if r["batch_size"] == batch_size and r["num_heads"] == num_heads
                    ]

                    if len(config_results) > 1:
                        seq_lens = [r["seq_len"] for r in config_results]
                        times = [r["total_time_ms"] for r in config_results]
                        memories = [r["peak_memory_mb"] for r in config_results]

                        f.write(f"- Batch={batch_size}, Heads={num_heads}:\n")
                        f.write(f"  - Sequence lengths: {seq_lens}\n")
                        f.write(
                            f"  - Total times (ms): {[f'{t:.1f}' for t in times]}\n"
                        )
                        f.write(
                            f"  - Peak memory (MB): {[f'{m:.1f}' for m in memories]}\n"
                        )

                        # Calculate scaling factor
                        if len(seq_lens) >= 2:
                            time_scaling = times[-1] / times[0]
                            seq_scaling = seq_lens[-1] / seq_lens[0]
                            f.write(
                                f"  - Time scaling: {time_scaling:.2f}x for {seq_scaling:.0f}x sequence length\n"
                            )
                            f.write(
                                f"  - Scaling efficiency: {(seq_scaling / time_scaling):.2f}\n"
                            )
                        f.write("\n")

            # Hilbert vs Non-Hilbert Comparison
            f.write("## Hilbert Ordering Impact\n\n")

            # Find matching ring and ring+hilbert results
            ring_results = [r for r in results if "Ring (size=" in r["module_name"]]
            hilbert_results = [r for r in results if "Ring+Hilbert" in r["module_name"]]

            comparison_data = []
            for ring in ring_results:
                # Find matching hilbert result
                hilbert = next(
                    (
                        h
                        for h in hilbert_results
                        if h["seq_len"] == ring["seq_len"]
                        and h["batch_size"] == ring["batch_size"]
                        and h["num_heads"] == ring["num_heads"]
                        and h["module_name"].split("(size=")[1]
                        == ring["module_name"].split("(size=")[1]
                    ),
                    None,
                )

                if hilbert:
                    speedup = ring["total_time_ms"] / hilbert["total_time_ms"]
                    memory_ratio = hilbert["peak_memory_mb"] / ring["peak_memory_mb"]

                    comparison_data.append(
                        [
                            ring["seq_len"],
                            ring["batch_size"],
                            ring["num_heads"],
                            f"{ring['total_time_ms']:.2f}",
                            f"{hilbert['total_time_ms']:.2f}",
                            f"{speedup:.2f}x",
                            f"{ring['peak_memory_mb']:.1f}",
                            f"{hilbert['peak_memory_mb']:.1f}",
                            f"{memory_ratio:.2f}x",
                        ]
                    )

            if comparison_data:
                headers = [
                    "Seq Len",
                    "Batch",
                    "Heads",
                    "Ring Time (ms)",
                    "Hilbert Time (ms)",
                    "Speedup",
                    "Ring Mem (MB)",
                    "Hilbert Mem (MB)",
                    "Mem Ratio",
                ]
                f.write(tabulate(comparison_data, headers=headers, tablefmt="github"))
                f.write("\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            f.write("### When to Use Hilbert Ordering\n\n")
            f.write("Based on the benchmark results:\n\n")

            # Analyze when Hilbert is beneficial
            hilbert_benefits = []
            for comp in comparison_data:
                seq_len = comp[0]
                speedup = float(comp[5][:-1])  # Remove 'x'
                if speedup > 1.1:
                    hilbert_benefits.append((seq_len, speedup))

            if hilbert_benefits:
                f.write("- **Hilbert ordering provides benefits for:**\n")
                for seq_len, speedup in sorted(hilbert_benefits):
                    f.write(
                        f"  - Sequences of {seq_len:,} tokens: {speedup:.2f}x speedup\n"
                    )
            else:
                f.write(
                    "- Hilbert ordering shows minimal performance benefits in these tests\n"
                )

            f.write("\n### Optimal Configurations\n\n")

            # Find best performing configurations
            by_seq_len = {}
            for r in results:
                seq_len = r["seq_len"]
                if seq_len not in by_seq_len:
                    by_seq_len[seq_len] = []
                by_seq_len[seq_len].append(r)

            for seq_len in sorted(by_seq_len.keys()):
                seq_results = by_seq_len[seq_len]

                # Find best throughput
                best_throughput = max(
                    seq_results, key=lambda x: x["throughput_total_tps"]
                )

                # Find best memory efficiency
                best_memory = min(seq_results, key=lambda x: x["memory_per_token_kb"])

                f.write(f"- **{seq_len:,} tokens:**\n")
                f.write(f"  - Best throughput: {best_throughput['module_name']} ")
                f.write(f"({best_throughput['throughput_total_tps']:.0f} tok/s)\n")
                f.write(f"  - Best memory efficiency: {best_memory['module_name']} ")
                f.write(f"({best_memory['memory_per_token_kb']:.2f} KB/token)\n")

            f.write("\n### General Guidelines\n\n")
            f.write(
                "1. **For sequences < 4K tokens**: Standard attention may be sufficient\n"
            )
            f.write(
                "2. **For sequences 4K-16K tokens**: Dilated attention provides good balance\n"
            )
            f.write(
                "3. **For sequences > 16K tokens**: Ring attention becomes necessary\n"
            )
            f.write(
                "4. **Hilbert ordering**: Most beneficial for very long sequences with specific access patterns\n"
            )
            f.write(
                "5. **Multi-GPU**: Ring attention scales well with increased ring size\n\n"
            )

            # Raw data
            f.write("## Raw Benchmark Data\n\n")
            f.write("Full results are saved in the accompanying JSON file.\n")

        # Save raw results as JSON
        json_path = report_path.replace(".md", ".json")
        # Ensure directory exists for JSON file too
        json_dir = os.path.dirname(json_path)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nReport saved to: {report_path}")
        print(f"Raw data saved to: {json_path}")

        return report_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark Ring Hilbert Attention")
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384, 32768],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[8, 16, 32],
        help="Number of attention heads to test",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--ring-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Ring sizes to test (for multi-GPU)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for computation",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/benchmarks/ring-hilbert-comprehensive-benchmark.md",
        help="Output report path",
    )
    parser.add_argument(
        "--no-standard", action="store_true", help="Skip standard attention benchmarks"
    )
    parser.add_argument(
        "--max-standard-seq",
        type=int,
        default=4096,
        help="Maximum sequence length for standard attention",
    )

    args = parser.parse_args()

    # Set up device and dtype
    device = torch.device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Create benchmark instance
    benchmark = HilbertRingBenchmark(
        device=device,
        dtype=dtype,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )

    # Run benchmarks
    print("Starting comprehensive Ring Hilbert attention benchmarks...")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Number of heads: {args.num_heads}")
    print(f"Ring sizes: {args.ring_sizes}")

    results = benchmark.run_comprehensive_benchmark(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        num_heads_list=args.num_heads,
        embed_dim=args.embed_dim,
        ring_sizes=args.ring_sizes,
        include_standard=not args.no_standard,
        max_standard_seq=args.max_standard_seq,
    )

    # Generate report
    benchmark.generate_report(results, args.output)


if __name__ == "__main__":
    main()
