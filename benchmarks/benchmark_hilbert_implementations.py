#!/usr/bin/env python3
"""
Benchmark comparison of different Hilbert attention implementations.

This script compares:
1. RingDilatedAttentionHilbertGPUOptimized - GPU-aware with automatic backend selection
2. RingDilatedAttentionHilbertProper - Original proper implementation
3. Standard attention (baseline)
"""

import argparse
import torch
import torch.nn as nn
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_gpu_optimized import (
    RingDilatedAttentionHilbertGPUOptimized,
)
from dilated_attention_pytorch.ring_dilated_attention_hilbert_proper import (
    RingDilatedAttentionHilbertProper,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info
from core.utils.memory import get_memory_stats, cleanup_memory
from core.utils.timing import CUDATimer


class HilbertImplementationBenchmark:
    """Benchmark different Hilbert attention implementations."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,  # Pascal friendly
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
    ):
        self.device = device
        self.dtype = dtype
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results = []

        # Get GPU info
        self.gpu_info = get_gpu_info(device)
        print(f"GPU: {self.gpu_info.name} ({self.gpu_info.architecture})")
        print(f"Compute capability: {self.gpu_info.compute_capability}")
        print(f"Recommended backend: {self.gpu_info.recommended_backend}")
        print(f"Optimal dtype: {self.gpu_info.optimal_dtype}")
        print()

    def benchmark_implementation(
        self,
        name: str,
        module: nn.Module,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        num_heads: int,
    ) -> Dict[str, Any]:
        """Benchmark a single implementation."""
        print(f"\nBenchmarking {name}: B={batch_size}, L={seq_len}, H={num_heads}")

        # Create input
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
            output = module(x)
            if x.requires_grad:
                loss = output.mean()
                loss.backward()

        # Forward pass timing
        print("  Measuring forward pass...")
        forward_times = []
        forward_memory = []

        for _ in range(self.benchmark_iterations):
            cleanup_memory()
            start_mem = get_memory_stats(self.device)["allocated"]

            timer = CUDATimer("forward", self.device, verbose=False)
            with timer:
                output = module(x)

            forward_times.append(timer.elapsed_ms)
            end_mem = get_memory_stats(self.device)["allocated"]
            forward_memory.append(end_mem - start_mem)

        # Backward pass timing
        backward_times = []
        total_memory = []

        if x.requires_grad:
            print("  Measuring backward pass...")
            for _ in range(self.benchmark_iterations):
                cleanup_memory()
                start_mem = get_memory_stats(self.device)["allocated"]

                # Forward
                output = module(x)

                # Backward
                timer = CUDATimer("backward", self.device, verbose=False)
                with timer:
                    loss = output.mean()
                    loss.backward()

                backward_times.append(timer.elapsed_ms)
                end_mem = get_memory_stats(self.device)["allocated"]
                total_memory.append(end_mem - start_mem)

        # Calculate statistics
        import numpy as np

        result = {
            "name": name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "forward_time_ms": np.mean(forward_times),
            "forward_std_ms": np.std(forward_times),
            "backward_time_ms": np.mean(backward_times) if backward_times else 0,
            "backward_std_ms": np.std(backward_times) if backward_times else 0,
            "forward_memory_mb": np.mean(forward_memory),
            "total_memory_mb": np.mean(total_memory)
            if total_memory
            else np.mean(forward_memory),
            "throughput_tokens_per_sec": (batch_size * seq_len)
            / (np.mean(forward_times) / 1000),
        }

        print(
            f"  Forward: {result['forward_time_ms']:.2f}±{result['forward_std_ms']:.2f} ms"
        )
        print(
            f"  Backward: {result['backward_time_ms']:.2f}±{result['backward_std_ms']:.2f} ms"
        )
        print(f"  Memory: {result['total_memory_mb']:.1f} MB")
        print(f"  Throughput: {result['throughput_tokens_per_sec']:.0f} tokens/sec")

        return result

    def run_comparison(
        self,
        seq_lengths: List[int],
        batch_sizes: List[int],
        num_heads_list: List[int],
        embed_dim: int = 768,
        segment_lengths: List[int] = [2048, 4096, 8192],
        dilation_rates: List[int] = [1, 2, 4],
    ) -> List[Dict[str, Any]]:
        """Run comprehensive comparison."""
        results = []

        for seq_len in seq_lengths:
            # Adjust segment lengths based on sequence length
            actual_segments = []
            actual_dilations = []
            for seg_len, dil_rate in zip(segment_lengths, dilation_rates):
                if seg_len <= seq_len:
                    actual_segments.append(seg_len)
                    actual_dilations.append(dil_rate)

            if not actual_segments:
                actual_segments = [seq_len]
                actual_dilations = [1]

            for batch_size in batch_sizes:
                for num_heads in num_heads_list:
                    print(f"\n{'=' * 80}")
                    print(
                        f"Configuration: seq_len={seq_len}, batch={batch_size}, heads={num_heads}"
                    )
                    print(f"Segments: {actual_segments}, Dilations: {actual_dilations}")
                    print(f"{'=' * 80}")

                    # Test standard attention for small sequences
                    if seq_len <= 4096:
                        try:
                            standard = nn.MultiheadAttention(
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                batch_first=True,
                                device=self.device,
                                dtype=self.dtype,
                            )

                            # Wrap for consistent interface
                            class StandardWrapper(nn.Module):
                                def __init__(self, mha):
                                    super().__init__()
                                    self.mha = mha

                                def forward(self, x):
                                    output, _ = self.mha(x, x, x, need_weights=False)
                                    return output

                            result = self.benchmark_implementation(
                                "Standard MHA",
                                StandardWrapper(standard),
                                batch_size,
                                seq_len,
                                embed_dim,
                                num_heads,
                            )
                            results.append(result)
                        except Exception as e:
                            print(f"  Standard attention failed: {e}")

                    # Test GPU-optimized implementation
                    try:
                        gpu_optimized = RingDilatedAttentionHilbertGPUOptimized(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=actual_segments,
                            dilation_rates=actual_dilations,
                            dropout=0.0,
                            ring_size=1,
                            use_hilbert=True,
                            device=self.device,
                            dtype=self.dtype,
                            benchmark_backends=False,  # We'll benchmark separately
                        )

                        result = self.benchmark_implementation(
                            f"GPU-Optimized (backend={gpu_optimized.attention_backend})",
                            gpu_optimized,
                            batch_size,
                            seq_len,
                            embed_dim,
                            num_heads,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  GPU-optimized failed: {e}")

                    # Test proper implementation
                    try:
                        proper = RingDilatedAttentionHilbertProper(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=actual_segments,
                            dilation_rates=actual_dilations,
                            dropout=0.0,
                            ring_size=1,
                            use_hilbert=True,
                            device=self.device,
                            dtype=self.dtype,
                        )

                        result = self.benchmark_implementation(
                            "Proper Implementation",
                            proper,
                            batch_size,
                            seq_len,
                            embed_dim,
                            num_heads,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  Proper implementation failed: {e}")

                    # Test GPU-optimized without Hilbert
                    try:
                        gpu_no_hilbert = RingDilatedAttentionHilbertGPUOptimized(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=actual_segments,
                            dilation_rates=actual_dilations,
                            dropout=0.0,
                            ring_size=1,
                            use_hilbert=False,  # Disable Hilbert
                            device=self.device,
                            dtype=self.dtype,
                        )

                        result = self.benchmark_implementation(
                            f"GPU-Optimized NoHilbert (backend={gpu_no_hilbert.attention_backend})",
                            gpu_no_hilbert,
                            batch_size,
                            seq_len,
                            embed_dim,
                            num_heads,
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"  GPU-optimized without Hilbert failed: {e}")

        return results

    def generate_report(
        self, results: List[Dict[str, Any]], output_dir: str = "docs/benchmarks"
    ):
        """Generate benchmark report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M-UTC")
        report_path = os.path.join(
            output_dir, f"hilbert-implementation-comparison-{timestamp}.md"
        )

        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

        with open(report_path, "w") as f:
            f.write("# Hilbert Attention Implementation Comparison\n\n")
            f.write(
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
            )

            # System info
            f.write("## System Information\n\n")
            f.write(f"- GPU: {self.gpu_info.name}\n")
            f.write(f"- Architecture: {self.gpu_info.architecture}\n")
            f.write(f"- Compute Capability: {self.gpu_info.compute_capability}\n")
            f.write(f"- Data Type: {self.dtype}\n")
            f.write(f"- PyTorch Version: {torch.__version__}\n")
            f.write(f"- CUDA Version: {torch.version.cuda}\n\n")

            # Performance comparison
            f.write("## Performance Comparison\n\n")

            # Group by sequence length
            seq_lengths = sorted(set(r["seq_len"] for r in results))

            for seq_len in seq_lengths:
                f.write(f"### Sequence Length: {seq_len:,} tokens\n\n")

                seq_results = [r for r in results if r["seq_len"] == seq_len]

                # Create comparison table
                f.write(
                    "| Implementation | Batch | Heads | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) | Throughput (tok/s) |\n"
                )
                f.write(
                    "|----------------|-------|-------|--------------|---------------|------------|-------------|--------------------|\n"
                )

                for r in seq_results:
                    f.write(f"| {r['name']} | {r['batch_size']} | {r['num_heads']} | ")
                    f.write(f"{r['forward_time_ms']:.2f}±{r['forward_std_ms']:.2f} | ")
                    f.write(
                        f"{r['backward_time_ms']:.2f}±{r['backward_std_ms']:.2f} | "
                    )
                    f.write(f"{r['forward_time_ms'] + r['backward_time_ms']:.2f} | ")
                    f.write(f"{r['total_memory_mb']:.1f} | ")
                    f.write(f"{r['throughput_tokens_per_sec']:.0f} |\n")

                f.write("\n")

            # Key findings
            f.write("## Key Findings\n\n")

            # Find best performers
            if results:
                # Best throughput
                best_throughput = max(
                    results, key=lambda x: x["throughput_tokens_per_sec"]
                )
                f.write(f"- **Best Throughput**: {best_throughput['name']} ")
                f.write(
                    f"({best_throughput['throughput_tokens_per_sec']:.0f} tokens/sec)\n"
                )

                # Best memory efficiency
                best_memory = min(results, key=lambda x: x["total_memory_mb"])
                f.write(f"- **Best Memory Efficiency**: {best_memory['name']} ")
                f.write(f"({best_memory['total_memory_mb']:.1f} MB)\n")

                # Fastest forward pass
                fastest_forward = min(results, key=lambda x: x["forward_time_ms"])
                f.write(f"- **Fastest Forward Pass**: {fastest_forward['name']} ")
                f.write(f"({fastest_forward['forward_time_ms']:.2f} ms)\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(
                "1. **GPU-Optimized Implementation** provides the best performance with automatic backend selection\n"
            )
            f.write(
                "2. **Hilbert ordering** impact varies by sequence length and access patterns\n"
            )
            f.write(
                "3. **Pascal GPUs** (like GTX 1080) benefit from FP32 computation\n"
            )
            f.write(
                "4. **Modern GPUs** can leverage Flash Attention for significant speedups\n\n"
            )

            # Save raw data
            json_path = report_path.replace(".md", ".json")
            with open(json_path, "w") as jf:
                json.dump(results, jf, indent=2)

            print(f"\nReport saved to: {report_path}")
            print(f"Raw data saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Hilbert attention implementations"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192, 16384],
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
        default=[8, 16],
        help="Number of attention heads",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: float32 for Pascal)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Benchmark iterations"
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

    # Create benchmark
    benchmark = HilbertImplementationBenchmark(
        device=device,
        dtype=dtype,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )

    # Run comparison
    print("Starting Hilbert implementation comparison...")
    results = benchmark.run_comparison(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        num_heads_list=args.num_heads,
        embed_dim=args.embed_dim,
    )

    # Generate report
    benchmark.generate_report(results)


if __name__ == "__main__":
    main()
