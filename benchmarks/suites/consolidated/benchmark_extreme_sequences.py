#!/usr/bin/env python3
"""
Extreme sequence length benchmark suite.

This consolidates functionality from:
- test_extreme_sequence_lengths.py
- test_max_sequence_length.py
- test_sequence_length_limits.py
- test_ultimate_limits.py
- test_long_seq_capability.py
- benchmark_extreme_sequences.py
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from benchmarks.core.config import BenchmarkConfig  # noqa: E402
from benchmarks.core.unified_runner import UnifiedBenchmarkRunner  # noqa: E402


class ExtremeLengthBenchmark:
    """Benchmark for extreme sequence lengths."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize extreme length benchmark.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.runner = UnifiedBenchmarkRunner(config)

    def find_max_sequence_length(
        self,
        implementation: str,
        min_len: int = 1024,
        max_len: int = 1_000_000,
        step_factor: float = 2.0,
    ) -> int:
        """Find maximum supported sequence length.

        Args:
            implementation: Implementation to test
            min_len: Minimum sequence length
            max_len: Maximum sequence length to try
            step_factor: Factor to increase sequence length

        Returns:
            Maximum working sequence length
        """
        print(f"\nFinding max sequence length for {implementation}...")

        batch_size = 1  # Use minimal batch size
        embed_dim = self.config.embed_dims[0]
        num_heads = self.config.num_heads[0]

        current_len = min_len
        max_working_len = 0

        while current_len <= max_len:
            # Find appropriate segment lengths
            segment_lengths = None
            dilation_rates = None

            # Try different segment configurations
            for base_seg in [512, 1024, 2048, 4096]:
                if current_len % (base_seg * 4) == 0:
                    segment_lengths = [base_seg, base_seg * 2, base_seg * 4]
                    dilation_rates = [1, 2, 4]
                    break
                elif current_len % (base_seg * 2) == 0:
                    segment_lengths = [base_seg, base_seg * 2]
                    dilation_rates = [1, 2]
                    break

            if segment_lengths is None:
                current_len = int(current_len * step_factor)
                continue

            print(f"  Testing seq_len={current_len:,}...", end=" ")

            # Test configuration
            try:
                result = self.runner.benchmark_single_configuration(
                    implementation=implementation,
                    batch_size=batch_size,
                    seq_len=current_len,
                    num_heads=num_heads,
                    embed_dim=embed_dim,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                )

                if result.error:
                    print(f"FAILED: {result.error}")
                    break
                else:
                    print(
                        f"OK (time={result.forward_time_ms:.1f}ms, "
                        f"memory={result.peak_memory_mb:.1f}MB)"
                    )
                    max_working_len = current_len
                    current_len = int(current_len * step_factor)

            except torch.cuda.OutOfMemoryError:
                print("OOM")
                break
            except Exception as e:
                print(f"ERROR: {e}")
                break

        return max_working_len

    def benchmark_memory_scaling(self, implementation: str, sequence_lengths: list):
        """Benchmark memory usage at different sequence lengths.

        Args:
            implementation: Implementation to test
            sequence_lengths: List of sequence lengths to test
        """
        print(f"\nMemory scaling for {implementation}:")
        print("-" * 60)
        print("Seq Length | Memory (MB) | MB per 1K tokens | Time (ms)")
        print("-" * 60)

        batch_size = 1
        embed_dim = self.config.embed_dims[0]
        num_heads = self.config.num_heads[0]

        for seq_len in sequence_lengths:
            # Find segment config
            segment_lengths = None
            dilation_rates = None

            for base_seg in [512, 1024, 2048, 4096]:
                if seq_len % (base_seg * 2) == 0:
                    segment_lengths = [base_seg, base_seg * 2]
                    dilation_rates = [1, 2]
                    break

            if segment_lengths is None:
                continue

            result = self.runner.benchmark_single_configuration(
                implementation=implementation,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                embed_dim=embed_dim,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            )

            if not result.error and result.peak_memory_mb:
                mb_per_1k = result.peak_memory_mb / (seq_len / 1000)
                print(
                    f"{seq_len:10,} | {result.peak_memory_mb:11.1f} | "
                    f"{mb_per_1k:16.2f} | {result.forward_time_ms:9.1f}"
                )

    def run_extreme_benchmarks(self):
        """Run all extreme sequence length benchmarks."""
        results = []

        # Test each implementation
        for impl in self.config.implementations:
            print(f"\n{'=' * 80}")
            print(f"Testing {impl}")
            print(f"{'=' * 80}")

            # Find max sequence length
            max_len = self.find_max_sequence_length(impl)
            print(f"\nMax sequence length: {max_len:,}")

            # Test memory scaling
            test_lengths = [
                length
                for length in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
                if length <= max_len
            ]
            self.benchmark_memory_scaling(impl, test_lengths)

        return results


def main():
    """Run extreme sequence length benchmarks."""
    parser = argparse.ArgumentParser(description="Extreme sequence length benchmarks")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["standard", "improved", "ring"],
        help="Implementations to test",
    )
    parser.add_argument(
        "--max-search-len",
        type=int,
        default=1_000_000,
        help="Maximum sequence length to search",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=768, help="Embedding dimension"
    )
    parser.add_argument(
        "--num-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        implementations=args.implementations,
        embed_dims=[args.embed_dim],
        num_heads=[args.num_heads],
        dtype=args.dtype,
        batch_sizes=[1],  # Use batch size 1 for extreme lengths
        warmup_iterations=1,
        benchmark_iterations=3,
        measure_memory=True,
    )

    print("=" * 80)
    print("Extreme Sequence Length Benchmarks")
    print("=" * 80)
    print(f"Implementations: {', '.join(config.implementations)}")
    print(f"Embed dim: {args.embed_dim}")
    print(f"Num heads: {args.num_heads}")
    print(f"Dtype: {args.dtype}")
    print(f"Device: {config.device}")
    print("=" * 80)

    # Run benchmarks
    benchmark = ExtremeLengthBenchmark(config)
    benchmark.run_extreme_benchmarks()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
