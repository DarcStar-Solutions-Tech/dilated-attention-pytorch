#!/usr/bin/env python3
"""
Benchmark comparing standard ring attention vs dilated ring attention.

This benchmark clarifies the important distinction:
- StandardRingAttention: Regular full attention with ring communication for O(n/k) memory
- RingDilatedAttentionSDPA: Dilated attention (sparse patterns) with ring communication
"""

import torch
import torch.distributed as dist
import time
import gc
import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt

from dilated_attention_pytorch import (
    StandardRingAttention,
    RingDilatedAttentionSDPA,
    RingAttentionConfig,
)
from dilated_attention_pytorch.utils import get_optimal_dtype


class DilatedVsStandardBenchmark:
    """Compare standard ring attention with dilated ring attention."""

    def __init__(
        self,
        sequence_lengths: List[int] = None,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        segment_lengths: List[int] = None,
        dilation_rates: List[int] = None,
        output_dir: str = "benchmarks/results/ring/dilated_vs_standard",
    ):
        self.sequence_lengths = sequence_lengths or [1024, 2048, 4096, 8192]
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.embed_dim = num_heads * head_dim
        self.segment_lengths = segment_lengths or [512, 1024, 2048]
        self.dilation_rates = dilation_rates or [1, 2, 4]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Device setup
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        self.dtype = get_optimal_dtype(self.device)

        # Results storage
        self.results = []

    def create_standard_ring_attention(self) -> torch.nn.Module:
        """Create standard ring attention (full attention with ring comm)."""
        config = RingAttentionConfig(
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            dropout=0.0,
            ring_size=self.world_size if self.world_size > 1 else None,
        )
        return StandardRingAttention(config, device=self.device, dtype=self.dtype)

    def create_dilated_ring_attention(self) -> torch.nn.Module:
        """Create dilated ring attention (sparse attention with ring comm)."""
        return RingDilatedAttentionSDPA(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            dropout=0.0,
            device=self.device,
            dtype=self.dtype,
        )

    def benchmark_implementation(
        self, impl_type: str, model: torch.nn.Module, seq_len: int
    ) -> Dict:
        """Benchmark a single implementation."""
        result = {
            "implementation": impl_type,
            "sequence_length": seq_len,
            "world_size": self.world_size,
            "rank": self.rank,
        }

        try:
            # Clear memory
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Create inputs based on implementation type
            if impl_type == "dilated_ring":
                # RingDilatedAttentionSDPA expects (batch, seq, embed_dim)
                x = torch.randn(
                    self.batch_size,
                    seq_len,
                    self.embed_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                input_shape = x.shape
            else:
                # StandardRingAttention expects (batch, seq, num_heads, head_dim)
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
                input_shape = q.shape

            # Memory measurement
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated() / (1024**2)

                # Warmup
                with torch.no_grad():
                    if impl_type == "dilated_ring":
                        _ = model(x)
                    else:
                        _ = model(q, k, v)

                torch.cuda.synchronize()

                # Timed run
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.no_grad():
                    if impl_type == "dilated_ring":
                        output = model(x)
                    else:
                        output = model(q, k, v)
                end_event.record()

                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
                mem_used = peak_mem - mem_before
            else:
                mem_before = 0
                start_time = time.perf_counter()
                with torch.no_grad():
                    if impl_type == "dilated_ring":
                        output = model(x)
                    else:
                        output = model(q, k, v)
                elapsed_time = time.perf_counter() - start_time
                peak_mem = 0
                mem_used = 0

            # Calculate metrics
            tokens_per_sec = (self.batch_size * seq_len) / elapsed_time

            # For dilated attention, calculate actual computation savings
            if impl_type == "dilated_ring":
                # Calculate how many positions are actually computed
                total_positions = 0
                for seg_len, dilation in zip(self.segment_lengths, self.dilation_rates):
                    if seg_len <= seq_len:
                        actual_positions = min(seg_len, seq_len // dilation)
                        total_positions += actual_positions

                sparsity = 1.0 - (total_positions / seq_len)
                computation_ratio = total_positions / seq_len
            else:
                sparsity = 0.0  # Standard attention computes all positions
                computation_ratio = 1.0

            result.update(
                {
                    "success": True,
                    "input_shape": str(input_shape),
                    "output_shape": str(output.shape),
                    "time_seconds": elapsed_time,
                    "throughput_tokens_per_sec": tokens_per_sec,
                    "memory_used_mb": mem_used,
                    "peak_memory_mb": peak_mem,
                    "sparsity": sparsity,
                    "computation_ratio": computation_ratio,
                    "error": None,
                }
            )

            # Cleanup
            del output
            if impl_type == "dilated_ring":
                del x
            else:
                del q, k, v
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            result.update(
                {
                    "success": False,
                    "error": str(e),
                }
            )

        return result

    def run_benchmarks(self):
        """Run comprehensive benchmarks comparing implementations."""
        print(f"\n{'=' * 60}")
        print("Dilated vs Standard Ring Attention Benchmark")
        print(f"{'=' * 60}")
        print(f"World size: {self.world_size}, Rank: {self.rank}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        print(f"Segment lengths: {self.segment_lengths}")
        print(f"Dilation rates: {self.dilation_rates}")

        # Create models
        print("\nCreating models...")
        standard_model = self.create_standard_ring_attention()
        dilated_model = self.create_dilated_ring_attention()

        # Benchmark each sequence length
        for seq_len in self.sequence_lengths:
            print(f"\n{'-' * 40}")
            print(f"Sequence length: {seq_len}")

            # Ensure sequence length is valid for segments
            max_segment = max(self.segment_lengths)
            if seq_len % max_segment != 0:
                print(
                    f"  Skipping: {seq_len} not divisible by max segment {max_segment}"
                )
                continue

            # Test standard ring attention
            print("\n  Testing StandardRingAttention (full attention + ring)...")
            standard_result = self.benchmark_implementation(
                "standard_ring", standard_model, seq_len
            )
            self.results.append(standard_result)

            if standard_result["success"]:
                print(f"    Time: {standard_result['time_seconds']:.3f}s")
                print(f"    Memory: {standard_result['memory_used_mb']:.1f} MB")
                print(
                    f"    Throughput: {standard_result['throughput_tokens_per_sec']:,.0f} tokens/sec"
                )

            # Test dilated ring attention
            print("\n  Testing RingDilatedAttentionSDPA (dilated attention + ring)...")
            dilated_result = self.benchmark_implementation(
                "dilated_ring", dilated_model, seq_len
            )
            self.results.append(dilated_result)

            if dilated_result["success"]:
                print(f"    Time: {dilated_result['time_seconds']:.3f}s")
                print(f"    Memory: {dilated_result['memory_used_mb']:.1f} MB")
                print(
                    f"    Throughput: {dilated_result['throughput_tokens_per_sec']:,.0f} tokens/sec"
                )
                print(f"    Sparsity: {dilated_result['sparsity']:.1%}")
                print(
                    f"    Computation ratio: {dilated_result['computation_ratio']:.1%}"
                )

            # Compare if both successful
            if standard_result["success"] and dilated_result["success"]:
                speedup = (
                    standard_result["time_seconds"] / dilated_result["time_seconds"]
                )
                memory_ratio = (
                    dilated_result["memory_used_mb"] / standard_result["memory_used_mb"]
                )
                print("\n  Dilated vs Standard comparison:")
                print(f"    Speedup: {speedup:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")

        # Save results
        self.save_results()
        if self.rank == 0:
            self.generate_report()

    def save_results(self):
        """Save benchmark results to file."""
        output_file = self.output_dir / f"results_w{self.world_size}_r{self.rank}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def generate_report(self):
        """Generate comparison report and visualizations."""
        if not self.results:
            return

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Filter successful results
        standard_results = [
            r
            for r in self.results
            if r["implementation"] == "standard_ring" and r["success"]
        ]
        dilated_results = [
            r
            for r in self.results
            if r["implementation"] == "dilated_ring" and r["success"]
        ]

        if standard_results and dilated_results:
            seq_lens = sorted(list(set(r["sequence_length"] for r in standard_results)))

            # Plot 1: Throughput comparison
            standard_throughput = [
                next(
                    r["throughput_tokens_per_sec"]
                    for r in standard_results
                    if r["sequence_length"] == s
                )
                for s in seq_lens
            ]
            dilated_throughput = [
                next(
                    r["throughput_tokens_per_sec"]
                    for r in dilated_results
                    if r["sequence_length"] == s
                )
                for s in seq_lens
            ]

            ax1.plot(
                seq_lens,
                standard_throughput,
                "o-",
                label="Standard Ring",
                linewidth=2,
                markersize=8,
            )
            ax1.plot(
                seq_lens,
                dilated_throughput,
                "s-",
                label="Dilated Ring",
                linewidth=2,
                markersize=8,
            )
            ax1.set_xlabel("Sequence Length")
            ax1.set_ylabel("Throughput (tokens/sec)")
            ax1.set_title("Throughput Comparison")
            ax1.set_xscale("log")
            ax1.set_yscale("log")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Memory usage
            standard_memory = [
                next(
                    r["memory_used_mb"]
                    for r in standard_results
                    if r["sequence_length"] == s
                )
                for s in seq_lens
            ]
            dilated_memory = [
                next(
                    r["memory_used_mb"]
                    for r in dilated_results
                    if r["sequence_length"] == s
                )
                for s in seq_lens
            ]

            ax2.plot(
                seq_lens,
                standard_memory,
                "o-",
                label="Standard Ring",
                linewidth=2,
                markersize=8,
            )
            ax2.plot(
                seq_lens,
                dilated_memory,
                "s-",
                label="Dilated Ring",
                linewidth=2,
                markersize=8,
            )
            ax2.set_xlabel("Sequence Length")
            ax2.set_ylabel("Memory Usage (MB)")
            ax2.set_title("Memory Usage Comparison")
            ax2.set_xscale("log")
            ax2.set_yscale("log")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Plot 3: Speedup ratio
            speedup_ratios = [
                s / d for s, d in zip(standard_throughput, dilated_throughput)
            ]
            ax3.plot(seq_lens, speedup_ratios, "g^-", linewidth=2, markersize=10)
            ax3.axhline(y=1, color="r", linestyle="--", alpha=0.5)
            ax3.set_xlabel("Sequence Length")
            ax3.set_ylabel("Speedup (Dilated/Standard)")
            ax3.set_title("Dilated Attention Speedup")
            ax3.set_xscale("log")
            ax3.grid(True, alpha=0.3)

            # Plot 4: Sparsity levels
            sparsity_levels = [
                next(
                    r["sparsity"] for r in dilated_results if r["sequence_length"] == s
                )
                for s in seq_lens
            ]
            ax4.plot(
                seq_lens,
                [s * 100 for s in sparsity_levels],
                "mo-",
                linewidth=2,
                markersize=8,
            )
            ax4.set_xlabel("Sequence Length")
            ax4.set_ylabel("Sparsity (%)")
            ax4.set_title("Dilated Attention Sparsity")
            ax4.set_xscale("log")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"comparison_{self.world_size}gpu.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Generate text report
        report_path = self.output_dir / f"report_{self.world_size}gpu.md"
        with open(report_path, "w") as f:
            f.write("# Dilated vs Standard Ring Attention Benchmark Report\n\n")
            f.write(f"**World Size**: {self.world_size} GPU(s)\n")
            f.write(f"**Device**: {self.device}\n")
            f.write(f"**Dtype**: {self.dtype}\n\n")

            f.write("## Key Differences\n\n")
            f.write(
                "- **StandardRingAttention**: Computes full attention matrix, uses ring communication for O(n/k) memory\n"
            )
            f.write(
                "- **RingDilatedAttentionSDPA**: Computes sparse dilated patterns, further reduces computation\n\n"
            )

            if standard_results and dilated_results:
                f.write("## Performance Summary\n\n")
                f.write(
                    "| Sequence Length | Standard Throughput | Dilated Throughput | Speedup | Sparsity |\n"
                )
                f.write(
                    "|-----------------|--------------------|--------------------|---------|----------|\n"
                )

                for seq_len in seq_lens:
                    std_res = next(
                        r for r in standard_results if r["sequence_length"] == seq_len
                    )
                    dil_res = next(
                        r for r in dilated_results if r["sequence_length"] == seq_len
                    )
                    speedup = std_res["time_seconds"] / dil_res["time_seconds"]

                    f.write(
                        f"| {seq_len:,} | "
                        f"{std_res['throughput_tokens_per_sec']:,.0f} | "
                        f"{dil_res['throughput_tokens_per_sec']:,.0f} | "
                        f"{speedup:.2f}x | "
                        f"{dil_res['sparsity']:.1%} |\n"
                    )

        print(f"\nReport saved to: {report_path}")
        print(f"Visualization saved to: {plot_path}")


def main():
    """Main entry point."""
    import argparse

    # Initialize distributed if running with torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

    parser = argparse.ArgumentParser(
        description="Dilated vs Standard Ring Attention Benchmark"
    )
    parser.add_argument(
        "--seq-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to test",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--segment-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048],
        help="Segment lengths for dilated attention",
    )
    parser.add_argument(
        "--dilation-rates",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Dilation rates for dilated attention",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/ring/dilated_vs_standard",
        help="Output directory",
    )

    args = parser.parse_args()

    benchmark = DilatedVsStandardBenchmark(
        sequence_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        output_dir=args.output_dir,
    )

    benchmark.run_benchmarks()

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
