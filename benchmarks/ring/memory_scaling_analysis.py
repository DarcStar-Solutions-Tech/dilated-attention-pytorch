#!/usr/bin/env python3
"""
Memory scaling analysis for ring attention implementations.

This script specifically analyzes how memory usage scales with sequence length
to validate O(n/k) scaling claims.
"""

import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import gc
from typing import Dict, List, Tuple
import json
from pathlib import Path

from dilated_attention_pytorch import (
    StandardRingAttention,
    HilbertRingAttention,
    RingBlockSparseAttention,
    RingAttentionConfig,
)
from dilated_attention_pytorch.base import DilatedAttention


class MemoryScalingAnalysis:
    """Analyze memory scaling of ring attention vs standard attention."""

    def __init__(
        self,
        seq_lengths: List[int] = None,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        output_dir: str = "benchmarks/results/ring/memory",
    ):
        self.seq_lengths = seq_lengths or [512, 1024, 2048, 4096, 8192, 16384, 32768]
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def measure_memory(
        self, model: torch.nn.Module, seq_len: int, warmup: bool = True
    ) -> Tuple[float, float, float]:
        """Measure memory usage of a model."""
        if self.device.type != "cuda":
            return 0, 0, 0

        # Clear cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset stats
        torch.cuda.reset_peak_memory_stats()

        # Measure baseline
        baseline_mem = torch.cuda.memory_allocated()

        # Create inputs
        q = torch.randn(
            self.batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        input_mem = torch.cuda.memory_allocated() - baseline_mem

        # Warmup if requested
        if warmup:
            with torch.no_grad():
                _ = model(q, k, v)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Measure forward pass
        torch.cuda.reset_peak_memory_stats()
        _ = torch.cuda.memory_allocated()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()

        # Get measurements
        peak_mem = torch.cuda.max_memory_allocated() - baseline_mem
        output_mem = output.element_size() * output.nelement()

        # Cleanup
        del q, k, v, output
        gc.collect()
        torch.cuda.empty_cache()

        return input_mem / (1024**2), peak_mem / (1024**2), output_mem / (1024**2)

    def analyze_implementation(self, impl_name: str) -> List[Dict]:
        """Analyze memory scaling for a specific implementation."""
        results = []

        print(f"\nAnalyzing {impl_name}...")

        for seq_len in self.seq_lengths:
            print(f"  Sequence length: {seq_len}")

            try:
                # Create appropriate segment lengths
                segments = []
                dilations = []

                base_seg = min(1024, seq_len // 2)
                seg = base_seg
                dil = 1

                while seg <= seq_len and len(segments) < 4:
                    segments.append(min(seg, seq_len))
                    dilations.append(dil)
                    seg *= 2
                    dil *= 2

                # Create model
                config = RingAttentionConfig(
                    segment_lengths=segments,
                    dilation_rates=dilations,
                    dropout=0.0,
                )

                if impl_name == "standard_attention":
                    model = DilatedAttention(
                        segment_lengths=segments,
                        dilation_rates=dilations,
                        dropout=0.0,
                    ).to(self.device)
                elif impl_name == "ring_standard":
                    model = StandardRingAttention(
                        config=config,
                        device=self.device,
                        dtype=self.dtype,
                    )
                elif impl_name == "ring_hilbert":
                    model = HilbertRingAttention(
                        config=config,
                        device=self.device,
                        dtype=self.dtype,
                    )
                elif impl_name == "ring_block_sparse":
                    model = RingBlockSparseAttention(
                        config=config,
                        sparsity_ratio=0.1,
                        device=self.device,
                        dtype=self.dtype,
                    )
                else:
                    raise ValueError(f"Unknown implementation: {impl_name}")

                model.eval()

                # Measure memory
                input_mb, peak_mb, output_mb = self.measure_memory(model, seq_len)

                # Calculate memory per token
                total_tokens = self.batch_size * seq_len
                mem_per_token = peak_mb / total_tokens * 1024  # KB per token

                # Expected O(n²) memory for standard attention
                if impl_name == "standard_attention":
                    # Attention matrix is seq_len x seq_len
                    _ = (seq_len * seq_len * 2) / (1024**2)  # MB for float16
                    expected_scaling = "O(n²)"
                else:
                    # Ring attention should be O(n/k)
                    _ = (seq_len * seq_len / self.world_size * 2) / (1024**2)
                    expected_scaling = f"O(n/{self.world_size})"

                result = {
                    "implementation": impl_name,
                    "sequence_length": seq_len,
                    "input_memory_mb": input_mb,
                    "peak_memory_mb": peak_mb,
                    "output_memory_mb": output_mb,
                    "memory_per_token_kb": mem_per_token,
                    "expected_scaling": expected_scaling,
                    "world_size": self.world_size,
                }

                results.append(result)
                print(
                    f"    Peak memory: {peak_mb:.2f} MB ({mem_per_token:.3f} KB/token)"
                )

                # Cleanup
                del model
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Error: {str(e)}")
                results.append(
                    {
                        "implementation": impl_name,
                        "sequence_length": seq_len,
                        "error": str(e),
                    }
                )

        return results

    def run_analysis(self):
        """Run complete memory scaling analysis."""
        all_results = []

        # Test standard attention (for comparison)
        if max(self.seq_lengths) <= 16384:  # Only test standard for reasonable sizes
            all_results.extend(self.analyze_implementation("standard_attention"))

        # Test ring implementations
        for impl in ["ring_standard", "ring_hilbert", "ring_block_sparse"]:
            all_results.extend(self.analyze_implementation(impl))

        # Save results
        self.save_results(all_results)

        # Visualize
        if self.rank == 0:
            self.visualize_results(all_results)

    def save_results(self, results: List[Dict]):
        """Save analysis results."""
        output_file = self.output_dir / f"memory_scaling_w{self.world_size}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def visualize_results(self, results: List[Dict]):
        """Create visualization of memory scaling."""
        # Filter successful results
        successful = [r for r in results if "error" not in r]
        if not successful:
            print("No successful results to visualize")
            return

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Absolute memory usage
        implementations = list(set(r["implementation"] for r in successful))

        for impl in implementations:
            impl_results = [r for r in successful if r["implementation"] == impl]
            if impl_results:
                seq_lengths = [r["sequence_length"] for r in impl_results]
                peak_mem = [r["peak_memory_mb"] for r in impl_results]

                label = impl.replace("_", " ").title()
                if "ring" in impl:
                    label += f" ({self.world_size} GPU)"

                ax1.plot(
                    seq_lengths,
                    peak_mem,
                    marker="o",
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

        ax1.set_xlabel("Sequence Length", fontsize=12)
        ax1.set_ylabel("Peak Memory (MB)", fontsize=12)
        ax1.set_title("Memory Usage Scaling", fontsize=14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Memory per token (normalized)
        for impl in implementations:
            impl_results = [r for r in successful if r["implementation"] == impl]
            if impl_results:
                seq_lengths = [r["sequence_length"] for r in impl_results]
                mem_per_token = [r["memory_per_token_kb"] for r in impl_results]

                label = impl.replace("_", " ").title()
                ax2.plot(
                    seq_lengths,
                    mem_per_token,
                    marker="o",
                    label=label,
                    linewidth=2,
                    markersize=8,
                )

        ax2.set_xlabel("Sequence Length", fontsize=12)
        ax2.set_ylabel("Memory per Token (KB)", fontsize=12)
        ax2.set_title("Memory Efficiency", fontsize=14)
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add annotation about scaling
        ax2.text(
            0.05,
            0.95,
            f"World Size: {self.world_size} GPU(s)",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plot_path = self.output_dir / f"memory_scaling_{self.world_size}gpu.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nPlot saved to: {plot_path}")

        # Generate summary
        self.generate_summary(successful)

    def generate_summary(self, results: List[Dict]):
        """Generate summary report of memory scaling."""
        summary_path = (
            self.output_dir / f"memory_scaling_summary_{self.world_size}gpu.md"
        )

        with open(summary_path, "w") as f:
            f.write("# Memory Scaling Analysis\n\n")
            f.write(f"**World Size**: {self.world_size} GPU(s)\n\n")

            # Calculate scaling factors
            f.write("## Scaling Analysis\n\n")

            implementations = list(set(r["implementation"] for r in results))

            for impl in implementations:
                impl_results = sorted(
                    [r for r in results if r["implementation"] == impl],
                    key=lambda x: x["sequence_length"],
                )

                if len(impl_results) >= 2:
                    f.write(f"\n### {impl.replace('_', ' ').title()}\n\n")

                    # Calculate scaling factor
                    first = impl_results[0]
                    last = impl_results[-1]

                    seq_ratio = last["sequence_length"] / first["sequence_length"]
                    mem_ratio = last["peak_memory_mb"] / first["peak_memory_mb"]

                    # For O(n) scaling, mem_ratio ≈ seq_ratio
                    # For O(n²) scaling, mem_ratio ≈ seq_ratio²
                    if abs(mem_ratio - seq_ratio) < abs(mem_ratio - seq_ratio**2):
                        observed_scaling = "~O(n)"
                    else:
                        observed_scaling = "~O(n²)"

                    f.write(f"- Sequence length increased: {seq_ratio:.1f}x\n")
                    f.write(f"- Memory increased: {mem_ratio:.1f}x\n")
                    f.write(f"- Observed scaling: {observed_scaling}\n")
                    f.write(
                        f"- Expected scaling: {impl_results[0].get('expected_scaling', 'Unknown')}\n"
                    )

            # Memory efficiency table
            f.write("\n## Memory Efficiency (KB per Token)\n\n")
            f.write("| Sequence Length |")
            for impl in implementations:
                f.write(f" {impl.replace('_', ' ').title()} |")
            f.write("\n|" + "-" * 17 + "|" + "-" * 25 * len(implementations) + "|\n")

            seq_lengths = sorted(list(set(r["sequence_length"] for r in results)))
            for seq_len in seq_lengths:
                f.write(f"| {seq_len:15,} |")
                for impl in implementations:
                    result = next(
                        (
                            r
                            for r in results
                            if r["implementation"] == impl
                            and r["sequence_length"] == seq_len
                        ),
                        None,
                    )
                    if result and "memory_per_token_kb" in result:
                        f.write(f" {result['memory_per_token_kb']:23.3f} |")
                    else:
                        f.write(" " + "-" * 23 + " |")
                f.write("\n")

        print(f"\nSummary saved to: {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Memory scaling analysis for ring attention"
    )
    parser.add_argument(
        "--seq-lengths", nargs="+", type=int, help="Sequence lengths to test"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/ring/memory",
        help="Output directory",
    )

    args = parser.parse_args()

    analysis = MemoryScalingAnalysis(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    analysis.run_analysis()


if __name__ == "__main__":
    main()
