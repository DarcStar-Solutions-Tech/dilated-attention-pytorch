"""
Comprehensive sequence length benchmarking across all implementations.

This script provides systematic benchmarking across different sequence length ranges:
- Production (1K-16K): Common real-world lengths
- Medium (32K-128K): Extended context scenarios
- Long (256K-1M): Stress testing
- Custom: User-defined ranges

Includes both power-of-2 and real-world sequence lengths.
"""

import torch
import time
import argparse
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import gc

from dilated_attention_pytorch import (
    DilatedAttention,
    MultiheadDilatedAttention,
    ImprovedDilatedAttention,
    ImprovedDilatedAttentionV2,
    ImprovedMultiheadDilatedAttention,
)

# Ring Attention implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention_v2 import (
        RingDilatedAttentionV2,
    )

    RING_V2_AVAILABLE = True
except ImportError:
    RING_V2_AVAILABLE = False

try:
    from dilated_attention_pytorch.ring_dilated_attention_v3 import (
        RingDilatedAttentionV3,
    )

    RING_V3_AVAILABLE = True
except ImportError:
    RING_V3_AVAILABLE = False

try:
    from dilated_attention_pytorch import RingDilatedAttentionProduction

    RING_PRODUCTION_AVAILABLE = True
except ImportError:
    RING_PRODUCTION_AVAILABLE = False

# Block-Sparse implementations
try:
    from dilated_attention_pytorch import (
        BlockSparseRingDilatedAttention,
        BlockSparseRingMultiheadDilatedAttention,
    )

    BLOCK_SPARSE_AVAILABLE = True
except ImportError:
    BLOCK_SPARSE_AVAILABLE = False

# Distributed implementations
try:
    from dilated_attention_pytorch import (
        DistributedImprovedDilatedAttention,
        DistributedImprovedMultiheadDilatedAttention,
    )

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

from benchmark_utils import BenchmarkOutputManager


@dataclass
class SequenceRangeBenchmarkResult:
    """Result for a single sequence length benchmark."""

    implementation: str
    sequence_length: int
    segment_lengths: List[int]
    dilation_rates: List[int]
    batch_size: int
    num_heads: int
    head_dim: int
    time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    success: bool
    error: Optional[str] = None
    enable_pattern_cache: bool = False
    enable_memory_pool: bool = False


class SequenceRangeBenchmarker:
    """Benchmark different sequence length ranges."""

    # Predefined sequence ranges
    PRODUCTION_LENGTHS = [1024, 2048, 4096, 8192, 12288, 16384]
    MEDIUM_LENGTHS = [32768, 49152, 65536, 98304, 131072]
    LONG_LENGTHS = [262144, 393216, 524288, 786432, 1048576]

    # Real-world inspired lengths
    DOCUMENT_LENGTHS = [512, 1024, 2048, 4096]  # Typical documents
    CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]  # LLM context windows
    BOOK_LENGTHS = [50000, 100000, 200000]  # Long-form content

    def __init__(self, device: str = "cuda", output_dir: str = "benchmark_results"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_manager = BenchmarkOutputManager(base_dir=output_dir)

    def get_segment_config(self, seq_len: int) -> Tuple[List[int], List[int]]:
        """Get appropriate segment lengths and dilation rates for sequence length."""
        if seq_len <= 8192:
            segment_lengths = [1024, 2048, 4096]
            dilation_rates = [1, 2, 4]
        elif seq_len <= 65536:
            segment_lengths = [2048, 8192, 32768]
            dilation_rates = [1, 2, 4]
        elif seq_len <= 524288:
            segment_lengths = [8192, 65536, 262144]
            dilation_rates = [1, 4, 16]
        else:
            segment_lengths = [65536, 262144, 524288]
            dilation_rates = [1, 4, 16]

        # Filter out segments larger than sequence length
        valid_segments = []
        valid_dilations = []
        for seg, dil in zip(segment_lengths, dilation_rates):
            if seg <= seq_len:
                valid_segments.append(seg)
                valid_dilations.append(dil)

        return valid_segments, valid_dilations

    def adjust_batch_size(self, seq_len: int, base_batch_size: int = 2) -> int:
        """Adjust batch size based on sequence length."""
        if seq_len <= 8192:
            return base_batch_size
        elif seq_len <= 32768:
            return max(1, base_batch_size // 2)
        elif seq_len <= 131072:
            return 1
        else:
            return 1  # Always use batch size 1 for very long sequences

    def create_model(
        self,
        implementation: str,
        segment_lengths: List[int],
        dilation_rates: List[int],
        num_heads: int,
        head_dim: int,
        enable_pattern_cache: bool = False,
        enable_memory_pool: bool = False,
    ) -> torch.nn.Module:
        """Create model instance based on implementation name."""
        kwargs = {
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
        }

        # Basic implementations
        if implementation == "dilated":
            kwargs["enable_memory_pool"] = enable_memory_pool
            return DilatedAttention(**kwargs).to(self.device)
        elif implementation == "multihead":
            return MultiheadDilatedAttention(
                embed_dim=num_heads * head_dim, num_heads=num_heads, **kwargs
            ).to(self.device)

        # Improved implementations
        elif implementation == "improved":
            kwargs["enable_memory_pool"] = enable_memory_pool
            return ImprovedDilatedAttention(**kwargs).to(self.device)
        elif implementation == "improved_v2":
            kwargs["enable_memory_pool"] = enable_memory_pool
            return ImprovedDilatedAttentionV2(**kwargs).to(self.device)
        elif implementation == "improved_multihead":
            return ImprovedMultiheadDilatedAttention(
                embed_dim=num_heads * head_dim, num_heads=num_heads, **kwargs
            ).to(self.device)

        # Ring Attention implementations
        elif implementation == "ring_v2" and RING_V2_AVAILABLE:
            kwargs["enable_memory_pool"] = enable_memory_pool
            kwargs["use_pattern_cache"] = enable_pattern_cache
            return RingDilatedAttentionV2(**kwargs).to(self.device)
        elif implementation == "ring_v3" and RING_V3_AVAILABLE:
            kwargs["enable_memory_pool"] = enable_memory_pool
            kwargs["use_pattern_cache"] = enable_pattern_cache
            return RingDilatedAttentionV3(**kwargs).to(self.device)
        elif implementation == "ring_production" and RING_PRODUCTION_AVAILABLE:
            kwargs["enable_memory_pool"] = enable_memory_pool
            kwargs["use_pattern_cache"] = enable_pattern_cache
            return RingDilatedAttentionProduction(**kwargs).to(self.device)

        # Block-Sparse implementations
        elif implementation == "block_sparse" and BLOCK_SPARSE_AVAILABLE:
            return BlockSparseRingDilatedAttention(
                sparsity_ratio=0.9, pattern_type="dilated_sparse", **kwargs
            ).to(self.device)
        elif implementation == "block_sparse_multihead" and BLOCK_SPARSE_AVAILABLE:
            return BlockSparseRingMultiheadDilatedAttention(
                embed_dim=num_heads * head_dim,
                num_heads=num_heads,
                sparsity_ratio=0.9,
                pattern_type="dilated_sparse",
                **kwargs,
            ).to(self.device)

        # Distributed implementations
        elif implementation == "distributed_improved" and DISTRIBUTED_AVAILABLE:
            kwargs["enable_memory_pool"] = enable_memory_pool
            return DistributedImprovedDilatedAttention(**kwargs).to(self.device)
        elif implementation == "distributed_multihead" and DISTRIBUTED_AVAILABLE:
            return DistributedImprovedMultiheadDilatedAttention(
                embed_dim=num_heads * head_dim, num_heads=num_heads, **kwargs
            ).to(self.device)

        else:
            raise ValueError(f"Unknown or unavailable implementation: {implementation}")

    def benchmark_sequence_length(
        self,
        implementation: str,
        seq_len: int,
        num_heads: int = 8,
        head_dim: int = 64,
        base_batch_size: int = 2,
        enable_pattern_cache: bool = False,
        enable_memory_pool: bool = False,
        warmup_steps: int = 3,
        benchmark_steps: int = 10,
    ) -> SequenceRangeBenchmarkResult:
        """Benchmark a single sequence length."""
        # Get configuration
        segment_lengths, dilation_rates = self.get_segment_config(seq_len)
        batch_size = self.adjust_batch_size(seq_len, base_batch_size)

        # Ensure sequence length is divisible by largest segment
        if segment_lengths:
            seq_len = (seq_len // segment_lengths[-1]) * segment_lengths[-1]

        try:
            # Create model
            model = self.create_model(
                implementation,
                segment_lengths,
                dilation_rates,
                num_heads,
                head_dim,
                enable_pattern_cache,
                enable_memory_pool,
            )

            # Create input tensors
            if "multihead" in implementation:
                shape = (batch_size, seq_len, num_heads * head_dim)
            else:
                shape = (batch_size, seq_len, num_heads, head_dim)

            q = torch.randn(shape, device=self.device, dtype=torch.float16)
            k = torch.randn(shape, device=self.device, dtype=torch.float16)
            v = torch.randn(shape, device=self.device, dtype=torch.float16)

            # Warmup
            for _ in range(warmup_steps):
                _ = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Measure memory before
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated() / 1024 / 1024

            # Benchmark
            start_time = time.time()
            for _ in range(benchmark_steps):
                _ = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            # Calculate metrics
            time_ms = (end_time - start_time) * 1000 / benchmark_steps

            if self.device.type == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_mb = peak_memory - start_memory
            else:
                memory_mb = 0.0

            total_tokens = batch_size * seq_len
            throughput = total_tokens / (time_ms / 1000)

            return SequenceRangeBenchmarkResult(
                implementation=implementation,
                sequence_length=seq_len,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=time_ms,
                memory_mb=memory_mb,
                throughput_tokens_per_sec=throughput,
                success=True,
                enable_pattern_cache=enable_pattern_cache,
                enable_memory_pool=enable_memory_pool,
            )

        except Exception as e:
            return SequenceRangeBenchmarkResult(
                implementation=implementation,
                sequence_length=seq_len,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                time_ms=0.0,
                memory_mb=0.0,
                throughput_tokens_per_sec=0.0,
                success=False,
                error=str(e),
                enable_pattern_cache=enable_pattern_cache,
                enable_memory_pool=enable_memory_pool,
            )
        finally:
            # Cleanup
            if "model" in locals():
                del model
            if "q" in locals():
                del q, k, v
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def benchmark_range(
        self,
        sequence_lengths: List[int],
        implementations: List[str],
        range_name: str,
        **kwargs,
    ) -> Dict[str, List[SequenceRangeBenchmarkResult]]:
        """Benchmark a range of sequence lengths."""
        results = {impl: [] for impl in implementations}

        total_benchmarks = len(implementations) * len(sequence_lengths)
        completed = 0

        print(f"\nBenchmarking {range_name} range: {len(sequence_lengths)} lengths")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Implementations: {implementations}")
        print("=" * 80)

        for seq_len in sequence_lengths:
            print(f"\nSequence length: {seq_len:,} tokens")

            for impl in implementations:
                print(f"  Testing {impl}...", end=" ", flush=True)

                result = self.benchmark_sequence_length(impl, seq_len, **kwargs)
                results[impl].append(result)

                if result.success:
                    print(
                        f"✓ {result.time_ms:.1f}ms, {result.memory_mb:.1f}MB, {result.throughput_tokens_per_sec:.0f} tok/s"
                    )
                else:
                    print(f"✗ {result.error}")

                completed += 1
                progress = completed / total_benchmarks * 100
                print(f"  Progress: {progress:.1f}%")

        return results

    def generate_report(
        self,
        all_results: Dict[str, Dict[str, List[SequenceRangeBenchmarkResult]]],
        output_file: str,
    ):
        """Generate comprehensive benchmark report."""
        report_lines = []
        report_lines.append("# Sequence Length Range Benchmark Report")
        report_lines.append(
            f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}"
        )
        report_lines.append(f"\nDevice: {self.device}")

        # Summary by range
        for range_name, range_results in all_results.items():
            report_lines.append(f"\n## {range_name} Range")

            # Create comparison table
            report_lines.append("\n### Performance Comparison")
            report_lines.append(
                "\n| Implementation | Seq Length | Time (ms) | Memory (MB) | Throughput (tok/s) | Status |"
            )
            report_lines.append(
                "|----------------|------------|-----------|-------------|-------------------|---------|"
            )

            for impl, results in range_results.items():
                for result in results:
                    status = "✓" if result.success else "✗"
                    report_lines.append(
                        f"| {result.implementation} | {result.sequence_length:,} | "
                        f"{result.time_ms:.1f} | {result.memory_mb:.1f} | "
                        f"{result.throughput_tokens_per_sec:,.0f} | {status} |"
                    )

            # Find best implementation for each length
            report_lines.append("\n### Best Implementation by Sequence Length")
            seq_lengths = sorted(
                set(
                    r.sequence_length
                    for results in range_results.values()
                    for r in results
                )
            )

            for seq_len in seq_lengths:
                best_impl = None
                best_throughput = 0

                for impl, results in range_results.items():
                    for result in results:
                        if result.sequence_length == seq_len and result.success:
                            if result.throughput_tokens_per_sec > best_throughput:
                                best_throughput = result.throughput_tokens_per_sec
                                best_impl = impl

                if best_impl:
                    report_lines.append(
                        f"- {seq_len:,} tokens: {best_impl} ({best_throughput:,.0f} tok/s)"
                    )

        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.output_manager.get_output_path(output_file, "md")
        with open(report_path, "w") as f:
            f.write(report_content)

        print(f"\nReport saved to: {report_path}")

        # Also save raw results as JSON
        json_data = {}
        for range_name, range_results in all_results.items():
            json_data[range_name] = {}
            for impl, results in range_results.items():
                json_data[range_name][impl] = [
                    {
                        "sequence_length": r.sequence_length,
                        "time_ms": r.time_ms,
                        "memory_mb": r.memory_mb,
                        "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in results
                ]

        json_path = self.output_manager.get_output_path(
            output_file.replace(".md", ".json"), "json"
        )
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequence length ranges")
    parser.add_argument(
        "--range",
        choices=[
            "production",
            "medium",
            "long",
            "document",
            "context",
            "book",
            "all",
            "custom",
        ],
        default="production",
        help="Sequence length range to benchmark",
    )
    parser.add_argument(
        "--custom-lengths",
        nargs="+",
        type=int,
        help="Custom sequence lengths for 'custom' range",
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["dilated", "improved", "ring_v2"],  # V3 deprecated - use V2
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--head-dim", type=int, default=64, help="Attention head dimension"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Base batch size")
    parser.add_argument(
        "--enable-pattern-cache",
        action="store_true",
        help="Enable pattern caching for supported implementations",
    )
    parser.add_argument(
        "--enable-memory-pool",
        action="store_true",
        help="Enable memory pooling for supported implementations",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = SequenceRangeBenchmarker(output_dir=args.output_dir)

    # Determine sequence lengths to test
    ranges_to_test = {}

    if args.range == "custom" and args.custom_lengths:
        ranges_to_test["Custom"] = sorted(args.custom_lengths)
    elif args.range == "all":
        ranges_to_test["Production (1K-16K)"] = (
            SequenceRangeBenchmarker.PRODUCTION_LENGTHS
        )
        ranges_to_test["Medium (32K-128K)"] = SequenceRangeBenchmarker.MEDIUM_LENGTHS
        ranges_to_test["Long (256K-1M)"] = SequenceRangeBenchmarker.LONG_LENGTHS
        ranges_to_test["Document"] = SequenceRangeBenchmarker.DOCUMENT_LENGTHS
        ranges_to_test["Context Window"] = SequenceRangeBenchmarker.CONTEXT_LENGTHS
        ranges_to_test["Book Length"] = SequenceRangeBenchmarker.BOOK_LENGTHS
    else:
        range_map = {
            "production": (
                "Production (1K-16K)",
                SequenceRangeBenchmarker.PRODUCTION_LENGTHS,
            ),
            "medium": ("Medium (32K-128K)", SequenceRangeBenchmarker.MEDIUM_LENGTHS),
            "long": ("Long (256K-1M)", SequenceRangeBenchmarker.LONG_LENGTHS),
            "document": ("Document", SequenceRangeBenchmarker.DOCUMENT_LENGTHS),
            "context": ("Context Window", SequenceRangeBenchmarker.CONTEXT_LENGTHS),
            "book": ("Book Length", SequenceRangeBenchmarker.BOOK_LENGTHS),
        }
        name, lengths = range_map[args.range]
        ranges_to_test[name] = lengths

    # Run benchmarks
    all_results = {}

    for range_name, sequence_lengths in ranges_to_test.items():
        results = benchmarker.benchmark_range(
            sequence_lengths=sequence_lengths,
            implementations=args.implementations,
            range_name=range_name,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            base_batch_size=args.batch_size,
            enable_pattern_cache=args.enable_pattern_cache,
            enable_memory_pool=args.enable_memory_pool,
        )
        all_results[range_name] = results

    # Generate report
    benchmarker.generate_report(all_results, "sequence_range_benchmark")

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
