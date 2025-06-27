"""
Benchmark Ring Attention with Dilated Patterns Integration.

This script compares the performance of Ring Attention V2 with dilated patterns
against standard Ring Attention and improved dilated attention.
"""

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


@dataclass
class BenchmarkResult:
    """Store benchmark results for one configuration."""

    implementation: str
    sequence_length: int
    ring_size: Optional[int]
    segment_lengths: Optional[list[int]]
    dilation_rates: Optional[list[int]]
    forward_time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    correctness_score: float  # Similarity to reference


class RingDilatedBenchmark:
    """Comprehensive benchmark for Ring Attention with Dilated Patterns."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        print("Benchmark Setup:")
        print(f"  Device: {self.device}")
        print(f"  Data type: {self.dtype}")
        print(f"  Batch size: {batch_size}")
        print(f"  Heads: {num_heads}")
        print(f"  Head dim: {head_dim}")

    def create_test_tensors(
        self, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create deterministic test tensors."""
        torch.manual_seed(42)  # Deterministic for comparisons

        query = torch.randn(
            self.batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        key = torch.randn(
            self.batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        value = torch.randn(
            self.batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return query, key, value

    def measure_memory_and_time(self, model, query, key, value, warmup=3, runs=10):
        """Measure execution time and peak memory usage."""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(query, key, value, is_causal=False)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Benchmark
        times = []

        with torch.no_grad():
            for _ in range(runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                output = model(query, key, value, is_causal=False)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)

        # Measure memory
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0.0

        return avg_time, peak_memory, output

    def calculate_correctness(self, output1, output2, rtol=1e-3, atol=1e-4):
        """Calculate similarity between two outputs (0.0 = different, 1.0 = identical)."""
        try:
            if torch.allclose(output1, output2, rtol=rtol, atol=atol):
                return 1.0
            else:
                # Calculate similarity score based on relative difference
                diff = torch.abs(output1 - output2)
                rel_diff = diff / (torch.abs(output1) + 1e-8)
                similarity = torch.exp(-rel_diff.mean()).item()
                return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def benchmark_implementation(
        self,
        name: str,
        model,
        seq_len: int,
        reference_output: Optional[torch.Tensor] = None,
        **extra_info,
    ) -> BenchmarkResult:
        """Benchmark a single implementation."""
        print(f"  Benchmarking {name}...")

        query, key, value = self.create_test_tensors(seq_len)

        try:
            avg_time, memory, output = self.measure_memory_and_time(
                model, query, key, value
            )

            # Calculate throughput
            total_tokens = self.batch_size * seq_len
            throughput = total_tokens / (avg_time / 1000)  # tokens/sec

            # Calculate correctness if reference provided
            correctness = 1.0
            if reference_output is not None:
                correctness = self.calculate_correctness(output, reference_output)

            result = BenchmarkResult(
                implementation=name,
                sequence_length=seq_len,
                ring_size=extra_info.get("ring_size"),
                segment_lengths=extra_info.get("segment_lengths"),
                dilation_rates=extra_info.get("dilation_rates"),
                forward_time_ms=avg_time,
                memory_mb=memory,
                throughput_tokens_per_sec=throughput,
                correctness_score=correctness,
            )

            print(
                f"    ✓ Time: {avg_time:.1f}ms, Memory: {memory:.1f}MB, "
                f"Throughput: {throughput:.0f} tok/s, Correctness: {correctness:.3f}"
            )

            return result

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return BenchmarkResult(
                implementation=f"{name} (FAILED)",
                sequence_length=seq_len,
                ring_size=extra_info.get("ring_size"),
                segment_lengths=extra_info.get("segment_lengths"),
                dilation_rates=extra_info.get("dilation_rates"),
                forward_time_ms=float("inf"),
                memory_mb=float("inf"),
                throughput_tokens_per_sec=0.0,
                correctness_score=0.0,
            )
        finally:
            # Cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    def run_comprehensive_benchmark(
        self, sequence_lengths: list[int]
    ) -> list[BenchmarkResult]:
        """Run comprehensive benchmark across multiple sequence lengths."""
        results = []

        print("\\n" + "=" * 80)
        print("RING ATTENTION WITH DILATED PATTERNS BENCHMARK")
        print("=" * 80)

        for seq_len in sequence_lengths:
            print(f"\\nSequence Length: {seq_len:,}")
            print("-" * 60)

            # Skip if sequence too large for memory
            estimated_memory = (
                self.batch_size * seq_len * self.num_heads * self.head_dim * 4
            ) / (1024**2)
            if estimated_memory > 1000:  # > 1GB base memory
                print(
                    f"  Skipping due to estimated memory usage: {estimated_memory:.1f}MB"
                )
                continue

            # Configuration for tests
            segment_lengths = [min(seq_len // 4, 1024), min(seq_len // 2, 2048)]
            dilation_rates = [1, 2]
            ring_sizes = [1, 2, 4] if seq_len >= 4096 else [1, 2]

            # Get reference output for correctness comparison
            # Use RingDilatedV2 with ring_size=1 as the reference
            reference_output = None
            try:
                ring_reference = RingDilatedAttentionV2(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                    dropout=0.0,
                    device=self.device,
                    dtype=self.dtype,
                )
                query, key, value = self.create_test_tensors(seq_len)
                with torch.no_grad():
                    reference_output = ring_reference(
                        query, key, value, is_causal=False
                    )
            except Exception as e:
                print(f"  ✗ Failed to generate reference output: {e}")

            # 1. Improved Dilated Attention (baseline)
            try:
                improved_dilated = ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    device=self.device,
                    dtype=self.dtype,
                )
                result = self.benchmark_implementation(
                    "ImprovedDilatedAttention",
                    improved_dilated,
                    seq_len,
                    reference_output,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                )
                results.append(result)
            except Exception as e:
                print(f"  ✗ ImprovedDilatedAttention failed: {e}")

            # 2. Ring Dilated Attention V2 (single device)
            try:
                ring_dilated_single = RingDilatedAttentionV2(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                    dropout=0.0,
                    device=self.device,
                    dtype=self.dtype,
                )
                result = self.benchmark_implementation(
                    "RingDilatedV2_single",
                    ring_dilated_single,
                    seq_len,
                    reference_output,
                    ring_size=1,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                )
                results.append(result)
            except Exception as e:
                print(f"  ✗ RingDilatedV2_single failed: {e}")

            # 3. Ring Dilated Attention V2 (simulated ring)
            for ring_size in ring_sizes:
                if ring_size == 1 or (ring_size > 1 and seq_len % ring_size != 0):
                    continue

                try:
                    ring_dilated_sim = RingDilatedAttentionV2(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        ring_size=ring_size,
                        dropout=0.0,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    result = self.benchmark_implementation(
                        f"RingDilatedV2_ring{ring_size}",
                        ring_dilated_sim,
                        seq_len,
                        reference_output,
                        ring_size=ring_size,
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                    )
                    results.append(result)
                except Exception as e:
                    print(f"  ✗ RingDilatedV2_ring{ring_size} failed: {e}")

        return results

    def print_summary_table(self, results: list[BenchmarkResult]):
        """Print a summary table of all results."""
        print("\\n" + "=" * 120)
        print("BENCHMARK SUMMARY")
        print("=" * 120)

        # Group by sequence length
        by_seq_len = {}
        for result in results:
            seq_len = result.sequence_length
            if seq_len not in by_seq_len:
                by_seq_len[seq_len] = []
            by_seq_len[seq_len].append(result)

        for seq_len in sorted(by_seq_len.keys()):
            print(f"\\nSequence Length: {seq_len:,}")
            print("-" * 110)
            print(
                f"{'Implementation':<30} {'Time(ms)':<10} {'Memory(MB)':<12} {'Throughput':<12} {'Correctness':<12} {'Notes':<20}"
            )
            print("-" * 110)

            seq_results = by_seq_len[seq_len]

            # Sort by time (excluding failed runs)
            successful_results = [
                r for r in seq_results if r.forward_time_ms != float("inf")
            ]
            failed_results = [
                r for r in seq_results if r.forward_time_ms == float("inf")
            ]

            successful_results.sort(key=lambda x: x.forward_time_ms)

            for result in successful_results + failed_results:
                time_str = (
                    f"{result.forward_time_ms:.1f}"
                    if result.forward_time_ms != float("inf")
                    else "FAILED"
                )
                memory_str = (
                    f"{result.memory_mb:.1f}"
                    if result.memory_mb != float("inf")
                    else "N/A"
                )
                throughput_str = (
                    f"{result.throughput_tokens_per_sec:.0f}"
                    if result.throughput_tokens_per_sec > 0
                    else "N/A"
                )
                correctness_str = f"{result.correctness_score:.3f}"

                notes = []
                if result.ring_size and result.ring_size > 1:
                    notes.append(f"ring={result.ring_size}")
                if result.segment_lengths:
                    notes.append(f"seg={result.segment_lengths}")
                notes_str = ", ".join(notes)

                print(
                    f"{result.implementation:<30} {time_str:<10} {memory_str:<12} {throughput_str:<12} {correctness_str:<12} {notes_str:<20}"
                )

        # Find best performers
        print("\\n" + "=" * 60)
        print("BEST PERFORMERS")
        print("=" * 60)

        successful = [r for r in results if r.forward_time_ms != float("inf")]
        if successful:
            fastest = min(successful, key=lambda x: x.forward_time_ms)
            print(
                f"Fastest: {fastest.implementation} ({fastest.forward_time_ms:.1f}ms)"
            )

            most_efficient = min(successful, key=lambda x: x.memory_mb)
            print(
                f"Most Memory Efficient: {most_efficient.implementation} ({most_efficient.memory_mb:.1f}MB)"
            )

            highest_throughput = max(
                successful, key=lambda x: x.throughput_tokens_per_sec
            )
            print(
                f"Highest Throughput: {highest_throughput.implementation} ({highest_throughput.throughput_tokens_per_sec:.0f} tok/s)"
            )


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Benchmark Ring Attention with Dilated Patterns"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--sequence_lengths",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--dtype", type=str, default="auto", help="Data type (float16/float32/auto)"
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Set dtype
    if args.dtype == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    # Run benchmark
    benchmark = RingDilatedBenchmark(
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    results = benchmark.run_comprehensive_benchmark(args.sequence_lengths)
    benchmark.print_summary_table(results)

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as markdown report
    with open(
        output_dir / f"ring-dilated-integration-benchmark-{timestamp}.md", "w"
    ) as f:
        f.write("# Ring Attention with Dilated Patterns Benchmark\\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\\n\\n")
        f.write("## Configuration\\n")
        f.write(f"- Device: {device}\\n")
        f.write(f"- Data type: {dtype}\\n")
        f.write(f"- Batch size: {args.batch_size}\\n")
        f.write(f"- Heads: {args.num_heads}\\n")
        f.write(f"- Head dim: {args.head_dim}\\n")
        f.write(f"- Sequence lengths: {args.sequence_lengths}\\n\\n")

        f.write("## Results\\n\\n")
        for result in results:
            f.write(f"### {result.implementation}\\n")
            f.write(f"- Sequence length: {result.sequence_length:,}\\n")
            f.write(f"- Time: {result.forward_time_ms:.1f}ms\\n")
            f.write(f"- Memory: {result.memory_mb:.1f}MB\\n")
            f.write(
                f"- Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec\\n"
            )
            f.write(f"- Correctness: {result.correctness_score:.3f}\\n")
            if result.ring_size:
                f.write(f"- Ring size: {result.ring_size}\\n")
            if result.segment_lengths:
                f.write(f"- Segment lengths: {result.segment_lengths}\\n")
                f.write(f"- Dilation rates: {result.dilation_rates}\\n")
            f.write("\\n")

    print("\\n✓ Benchmark complete! Results saved to:")
    print(f"  {output_dir / f'ring-dilated-integration-benchmark-{timestamp}.md'}")


if __name__ == "__main__":
    main()
