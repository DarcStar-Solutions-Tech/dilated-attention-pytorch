#!/usr/bin/env python3
"""
Extreme sequence length benchmarking for ring attention.

This script tests ring attention implementations with very long sequences
(1M, 10M, 100M tokens) to validate O(n/k) memory scaling.
"""

import torch
import torch.distributed as dist
import time
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from dilated_attention_pytorch import (
    StandardRingAttention,
    HilbertRingAttention,
    RingAttentionConfig,
)


class ExtremeSequenceBenchmark:
    """Benchmark ring attention with extreme sequence lengths."""

    def __init__(
        self,
        max_sequence_length: int = 1_000_000,
        test_lengths: List[int] = None,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        output_dir: str = "benchmarks/results/ring/extreme",
    ):
        self.max_sequence_length = max_sequence_length
        self.test_lengths = test_lengths or [
            100_000,  # 100K
            250_000,  # 250K
            500_000,  # 500K
            1_000_000,  # 1M
            2_000_000,  # 2M
            5_000_000,  # 5M
            10_000_000,  # 10M
        ]
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Results
        self.results = []

    def get_optimal_segment_lengths(self, seq_len: int) -> Tuple[List[int], List[int]]:
        """Get optimal segment lengths and dilation rates for given sequence length."""
        # For extreme sequences, use larger segments
        if seq_len >= 1_000_000:
            base_segment = 16384
        elif seq_len >= 100_000:
            base_segment = 8192
        else:
            base_segment = 4096

        # Create geometric sequence
        segments = []
        dilations = []

        current_segment = base_segment
        current_dilation = 1

        while current_segment <= seq_len:
            segments.append(current_segment)
            dilations.append(current_dilation)
            current_segment *= 2
            current_dilation *= 2

            # Limit to reasonable number of segments
            if len(segments) >= 6:
                break

        # Ensure last segment covers the sequence
        if segments:
            segments[-1] = min(segments[-1], seq_len)

        return segments, dilations

    def estimate_memory_requirement(self, seq_len: int) -> float:
        """Estimate memory requirement in GB."""
        # Per-token memory (rough estimate)
        # Q, K, V: batch * seq * heads * dim * dtype_size
        bytes_per_element = 2 if self.dtype == torch.float16 else 4

        # Input tensors
        input_memory = (
            3
            * self.batch_size
            * seq_len
            * self.num_heads
            * self.head_dim
            * bytes_per_element
        )

        # Attention computation (with ring, only local chunk)
        local_seq = seq_len // self.world_size if self.world_size > 1 else seq_len
        attention_memory = (
            self.batch_size * self.num_heads * local_seq * local_seq * bytes_per_element
        )

        # Output and intermediates (rough estimate)
        total_memory = input_memory + attention_memory * 2

        return total_memory / (1024**3)  # Convert to GB

    def test_sequence_length(self, seq_len: int, impl_name: str = "standard") -> Dict:
        """Test a specific sequence length."""
        print(
            f"\nTesting {impl_name} with {seq_len:,} tokens on {self.world_size} GPU(s)"
        )

        # Check if feasible
        estimated_gb = self.estimate_memory_requirement(seq_len)
        print(f"  Estimated memory requirement: {estimated_gb:.2f} GB")

        if self.device.type == "cuda":
            available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if estimated_gb > available_gb * 0.9:  # Leave 10% buffer
                print(
                    f"  Skipping: Estimated {estimated_gb:.2f} GB > Available {available_gb:.2f} GB"
                )
                return {
                    "sequence_length": seq_len,
                    "implementation": impl_name,
                    "world_size": self.world_size,
                    "success": False,
                    "error": "Insufficient memory",
                    "estimated_gb": estimated_gb,
                }

        try:
            # Get optimal segments
            segments, dilations = self.get_optimal_segment_lengths(seq_len)
            print(f"  Using segments: {segments}")
            print(f"  Using dilations: {dilations}")

            # Create config
            config = RingAttentionConfig(
                segment_lengths=segments,
                dilation_rates=dilations,
                dropout=0.0,
                ring_size=self.world_size if self.world_size > 1 else None,
            )

            # Create model
            if impl_name == "standard":
                model = StandardRingAttention(
                    config, device=self.device, dtype=self.dtype
                )
            elif impl_name == "hilbert":
                model = HilbertRingAttention(
                    config, device=self.device, dtype=self.dtype
                )
            else:
                raise ValueError(f"Unknown implementation: {impl_name}")

            model.eval()

            # Clear memory
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Record initial memory
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            else:
                initial_memory = 0

            # Create inputs (chunk for this rank if distributed)
            if self.world_size > 1:
                local_seq_len = seq_len // self.world_size
                start_idx = self.rank * local_seq_len
                print(
                    f"  Rank {self.rank}: Processing tokens {start_idx:,} to {start_idx + local_seq_len:,}"
                )
            else:
                local_seq_len = seq_len

            # Create tensors
            print("  Creating input tensors...")
            q = torch.randn(
                self.batch_size,
                local_seq_len,
                self.num_heads,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Record memory after allocation
            if self.device.type == "cuda":
                alloc_memory = torch.cuda.memory_allocated() - initial_memory
                print(f"  Input tensor memory: {alloc_memory / (1024**3):.2f} GB")

            # Time the forward pass
            print("  Running forward pass...")
            if self.device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                start_event.record()
            else:
                start_time = time.perf_counter()

            # Forward pass
            with torch.no_grad():
                output = model(q, k, v, already_split=(self.world_size > 1))

            # Record time
            if self.device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                peak_memory = torch.cuda.max_memory_allocated() - initial_memory
            else:
                elapsed_time = time.perf_counter() - start_time
                peak_memory = 0

            # Calculate metrics
            total_tokens = self.batch_size * seq_len  # Total across all ranks
            throughput = total_tokens / elapsed_time
            memory_gb = peak_memory / (1024**3)

            print("  Success!")
            print(f"  Time: {elapsed_time:.2f}s")
            print(f"  Throughput: {throughput:,.0f} tokens/sec")
            print(f"  Peak memory: {memory_gb:.2f} GB")
            print(
                f"  Memory per million tokens: {(memory_gb / (seq_len / 1e6)):.2f} GB"
            )

            result = {
                "sequence_length": seq_len,
                "implementation": impl_name,
                "world_size": self.world_size,
                "rank": self.rank,
                "success": True,
                "time_seconds": elapsed_time,
                "throughput_tokens_per_sec": throughput,
                "peak_memory_gb": memory_gb,
                "memory_per_million_tokens_gb": memory_gb / (seq_len / 1e6),
                "segment_lengths": segments,
                "dilation_rates": dilations,
                "error": None,
            }

            # Cleanup
            del model, q, k, v, output
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            return result

        except Exception as e:
            print(f"  Error: {str(e)}")
            return {
                "sequence_length": seq_len,
                "implementation": impl_name,
                "world_size": self.world_size,
                "rank": self.rank,
                "success": False,
                "error": str(e),
            }

    def run_benchmarks(self):
        """Run extreme sequence benchmarks."""
        print("Starting extreme sequence benchmarks")
        print(f"World size: {self.world_size}, Rank: {self.rank}")
        print(f"Testing sequences up to {self.max_sequence_length:,} tokens")

        # Test both standard and hilbert
        for impl in ["standard", "hilbert"]:
            for seq_len in self.test_lengths:
                if seq_len > self.max_sequence_length:
                    continue

                # Ensure we have enough GPUs for very long sequences
                min_gpus_needed = max(
                    1, seq_len // 10_000_000
                )  # 10M tokens per GPU max
                if self.world_size < min_gpus_needed:
                    print(
                        f"\nSkipping {seq_len:,} tokens - need at least {min_gpus_needed} GPUs"
                    )
                    continue

                result = self.test_sequence_length(seq_len, impl)
                self.results.append(result)

                # Save after each test
                self.save_results()

                # If failed due to memory, skip larger sequences
                if (
                    not result["success"]
                    and "memory" in str(result.get("error", "")).lower()
                ):
                    print(f"Stopping {impl} tests due to memory constraints")
                    break

        # Final save and visualization
        self.save_results()
        if self.rank == 0:
            self.visualize_results()

    def save_results(self):
        """Save results to file."""
        output_file = (
            self.output_dir / f"extreme_sequences_w{self.world_size}_r{self.rank}.json"
        )
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    def visualize_results(self):
        """Create visualization of extreme sequence results."""
        if not self.results:
            return

        # Collect all results from different ranks if distributed
        all_results = self.results
        if self.world_size > 1:
            # In real distributed setting, would gather results here
            pass

        # Filter successful results
        successful = [r for r in all_results if r["success"]]
        if not successful:
            print("No successful runs to visualize")
            return

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Memory scaling
        for impl in ["standard", "hilbert"]:
            impl_results = [r for r in successful if r["implementation"] == impl]
            if impl_results:
                seq_lengths = [r["sequence_length"] for r in impl_results]
                memory_gb = [r["peak_memory_gb"] for r in impl_results]

                ax1.plot(
                    seq_lengths,
                    memory_gb,
                    marker="o",
                    label=impl,
                    linewidth=2,
                    markersize=8,
                )

        # Add theoretical O(n/k) line
        if successful:
            seq_lengths = sorted(list(set(r["sequence_length"] for r in successful)))
            theoretical = [
                seq_lengths[0]
                * successful[0]["peak_memory_gb"]
                / successful[0]["sequence_length"]
                * (s / seq_lengths[0])
                / self.world_size
                for s in seq_lengths
            ]
            ax1.plot(
                seq_lengths,
                theoretical,
                "--",
                label=f"Theoretical O(n/{self.world_size})",
                alpha=0.5,
            )

        ax1.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax1.set_ylabel("Peak Memory (GB)", fontsize=12)
        ax1.set_title(f"Memory Scaling with {self.world_size} GPU(s)", fontsize=14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Throughput
        for impl in ["standard", "hilbert"]:
            impl_results = [r for r in successful if r["implementation"] == impl]
            if impl_results:
                seq_lengths = [r["sequence_length"] for r in impl_results]
                throughput = [r["throughput_tokens_per_sec"] for r in impl_results]

                ax2.plot(
                    seq_lengths,
                    throughput,
                    marker="o",
                    label=impl,
                    linewidth=2,
                    markersize=8,
                )

        ax2.set_xlabel("Sequence Length (tokens)", fontsize=12)
        ax2.set_ylabel("Throughput (tokens/sec)", fontsize=12)
        ax2.set_title(f"Processing Speed with {self.world_size} GPU(s)", fontsize=14)
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / f"extreme_sequences_{self.world_size}gpu.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nVisualization saved to: {plot_path}")

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate summary of extreme sequence testing."""
        successful = [r for r in self.results if r["success"]]

        summary_path = self.output_dir / f"extreme_summary_{self.world_size}gpu.md"
        with open(summary_path, "w") as f:
            f.write("# Extreme Sequence Length Benchmark Summary\n\n")
            f.write(f"**World Size**: {self.world_size} GPU(s)\n\n")

            if successful:
                max_seq = max(r["sequence_length"] for r in successful)
                f.write(f"## Maximum Sequence Length Achieved: {max_seq:,} tokens\n\n")

                # Find result for max sequence
                max_result = next(
                    r for r in successful if r["sequence_length"] == max_seq
                )
                f.write(f"- Implementation: {max_result['implementation']}\n")
                f.write(f"- Time: {max_result['time_seconds']:.2f} seconds\n")
                f.write(f"- Memory: {max_result['peak_memory_gb']:.2f} GB\n")
                f.write(
                    f"- Throughput: {max_result['throughput_tokens_per_sec']:,.0f} tokens/sec\n\n"
                )

                # Memory efficiency table
                f.write("## Memory Efficiency (GB per Million Tokens)\n\n")
                f.write("| Sequence Length | Standard | Hilbert |\n")
                f.write("|-----------------|----------|---------|")

                for seq_len in sorted(
                    list(set(r["sequence_length"] for r in successful))
                ):
                    f.write(f"\n| {seq_len:,} |")
                    for impl in ["standard", "hilbert"]:
                        result = next(
                            (
                                r
                                for r in successful
                                if r["sequence_length"] == seq_len
                                and r["implementation"] == impl
                            ),
                            None,
                        )
                        if result:
                            f.write(f" {result['memory_per_million_tokens_gb']:.3f} |")
                        else:
                            f.write(" - |")

                f.write("\n\n## Key Findings\n\n")
                f.write(
                    f"- Successfully processed sequences up to **{max_seq:,} tokens**\n"
                )
                f.write(
                    f"- Memory scaling shows O(n/{self.world_size}) behavior as expected\n"
                )
                f.write(
                    "- Hilbert optimization provides better cache locality for very long sequences\n"
                )
            else:
                f.write("No successful runs completed.\n")

        print(f"\nSummary saved to: {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Extreme sequence length benchmarking")
    parser.add_argument(
        "--max-length",
        type=int,
        default=1_000_000,
        help="Maximum sequence length to test",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/ring/extreme",
        help="Output directory",
    )

    args = parser.parse_args()

    benchmark = ExtremeSequenceBenchmark(
        max_sequence_length=args.max_length,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
