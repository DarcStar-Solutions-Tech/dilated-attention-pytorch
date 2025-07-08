#!/usr/bin/env python3
"""
Fixed comprehensive benchmark of all dilated attention implementations.

This script correctly handles parameter differences between implementations.
"""

import torch
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime, timezone
import gc

# Suppress some warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    implementation: str
    category: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    throughput_tokens_per_sec: float
    success: bool
    error: Optional[str] = None


class DilatedAttentionBenchmark:
    """Fixed benchmark harness for all dilated attention implementations."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []

    def get_implementations(self) -> Dict[str, Any]:
        """Get implementations with correct parameters."""
        implementations = {}

        # 1. Core implementations
        implementations["core"] = []

        # DilatedAttention - uses new core architecture
        try:
            from dilated_attention_pytorch import DilatedAttention

            implementations["core"].append(
                {
                    "name": "DilatedAttention",
                    "class": DilatedAttention,
                    "init": lambda: DilatedAttention(
                        segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
                    ),
                    "input_format": "4d",  # [batch, seq, heads, dim]
                }
            )
        except Exception as e:
            print(f"Failed to load DilatedAttention: {e}")

        # ImprovedDilatedAttention
        try:
            from dilated_attention_pytorch import ImprovedDilatedAttention

            implementations["core"].append(
                {
                    "name": "ImprovedDilatedAttention",
                    "class": ImprovedDilatedAttention,
                    "init": lambda: ImprovedDilatedAttention(
                        segment_lengths=[512, 1024, 2048], dilation_rates=[1, 2, 4]
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load ImprovedDilatedAttention: {e}")

        # 2. Multihead implementations
        implementations["multihead"] = []

        # MultiheadDilatedAttention
        try:
            from dilated_attention_pytorch import MultiheadDilatedAttention

            implementations["multihead"].append(
                {
                    "name": "MultiheadDilatedAttention",
                    "class": MultiheadDilatedAttention,
                    "init": lambda: MultiheadDilatedAttention(
                        embed_dim=768,
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ),
                    "input_format": "3d",  # [batch, seq, embed_dim]
                }
            )
        except Exception as e:
            print(f"Failed to load MultiheadDilatedAttention: {e}")

        # ImprovedMultiheadDilatedAttention
        try:
            from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention

            implementations["multihead"].append(
                {
                    "name": "ImprovedMultiheadDilatedAttention",
                    "class": ImprovedMultiheadDilatedAttention,
                    "init": lambda: ImprovedMultiheadDilatedAttention(
                        embed_dim=768,
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        dropout=0.0,
                    ),
                    "input_format": "3d",
                }
            )
        except Exception as e:
            print(f"Failed to load ImprovedMultiheadDilatedAttention: {e}")

        # 3. Ring attention implementations
        implementations["ring"] = []

        # RingDilatedAttentionProduction
        try:
            from dilated_attention_pytorch import RingDilatedAttentionProduction

            implementations["ring"].append(
                {
                    "name": "RingDilatedAttentionProduction",
                    "class": RingDilatedAttentionProduction,
                    "init": lambda: RingDilatedAttentionProduction(
                        config={
                            "dim": 768,
                            "heads": 12,
                            "segment_lengths": [512, 1024, 2048],
                            "dilation_rates": [1, 2, 4],
                            "ring_size": 1,
                        }
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load RingDilatedAttentionProduction: {e}")

        # 4. Block-sparse implementations
        implementations["block_sparse"] = []

        # BlockSparseRingDilatedAttention
        try:
            from dilated_attention_pytorch import BlockSparseRingDilatedAttention

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseRingDilatedAttention",
                    "class": BlockSparseRingDilatedAttention,
                    "init": lambda: BlockSparseRingDilatedAttention(
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        sparsity_ratio=0.1,
                        block_size=64,
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load BlockSparseRingDilatedAttention: {e}")

        # BlockSparseRingDilatedAttentionFixed
        try:
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention_fixed import (
                BlockSparseRingDilatedAttentionFixed,
            )

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseRingDilatedAttentionFixed",
                    "class": BlockSparseRingDilatedAttentionFixed,
                    "init": lambda: BlockSparseRingDilatedAttentionFixed(
                        dim=64,
                        heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        sparsity_ratio=0.1,
                        block_size=64,
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load BlockSparseRingDilatedAttentionFixed: {e}")

        # BlockSparseRingMultiheadDilatedAttention
        try:
            from dilated_attention_pytorch import (
                BlockSparseRingMultiheadDilatedAttention,
            )

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseRingMultiheadDilatedAttention",
                    "class": BlockSparseRingMultiheadDilatedAttention,
                    "init": lambda: BlockSparseRingMultiheadDilatedAttention(
                        embed_dim=768,
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        sparsity_ratio=0.1,
                        block_size=64,
                    ),
                    "input_format": "3d",
                }
            )
        except Exception as e:
            print(f"Failed to load BlockSparseRingMultiheadDilatedAttention: {e}")

        # BlockSparseAdaptive
        try:
            from dilated_attention_pytorch import BlockSparseAdaptive

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseAdaptive",
                    "class": BlockSparseAdaptive,
                    "init": lambda: BlockSparseAdaptive(
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        num_heads=12,
                        head_dim=64,
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load BlockSparseAdaptive: {e}")

        # BlockSparseRingDilatedAttentionHilbertPostPattern
        try:
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
                BlockSparseRingDilatedAttentionHilbertPostPattern,
            )
            from dilated_attention_pytorch import SparsePatternConfig

            implementations["block_sparse"].append(
                {
                    "name": "BlockSparseRingDilatedAttentionHilbertPostPattern",
                    "class": BlockSparseRingDilatedAttentionHilbertPostPattern,
                    "init": lambda: BlockSparseRingDilatedAttentionHilbertPostPattern(
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                        sparse_config=SparsePatternConfig(
                            sparsity_ratio=0.1, block_size=64
                        ),
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(
                f"Failed to load BlockSparseRingDilatedAttentionHilbertPostPattern: {e}"
            )

        # 5. Kernel implementations
        implementations["kernels"] = []

        # HilbertDilatedAttention
        try:
            from dilated_attention_pytorch.kernels.hilbert_dilated_attention import (
                HilbertDilatedAttention,
            )

            implementations["kernels"].append(
                {
                    "name": "HilbertDilatedAttention",
                    "class": HilbertDilatedAttention,
                    "init": lambda: HilbertDilatedAttention(
                        num_heads=12,
                        segment_lengths=[512, 1024, 2048],
                        dilation_rates=[1, 2, 4],
                    ),
                    "input_format": "4d",
                }
            )
        except Exception as e:
            print(f"Failed to load HilbertDilatedAttention: {e}")

        return implementations

    def create_inputs(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        input_format: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create properly formatted inputs."""
        if input_format == "4d":
            # [batch, seq_len, num_heads, head_dim]
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=torch.float32,
            )
        elif input_format == "3d":
            # [batch, seq_len, embed_dim]
            embed_dim = num_heads * head_dim
            q = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
            k = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
            v = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=torch.float32
            )
        else:
            raise ValueError(f"Unknown input format: {input_format}")

        return q, k, v

    def benchmark_implementation(
        self,
        impl_info: Dict[str, Any],
        category: str,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single implementation."""
        name = impl_info["name"]

        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            # Create model
            model = impl_info["init"]().to(self.device)
            model.eval()

            # Create inputs
            q, k, v = self.create_inputs(
                batch_size, seq_len, num_heads, head_dim, impl_info["input_format"]
            )

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model(q, k, v)

            # Measure forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            forward_times = []
            for _ in range(num_iterations):
                start = time.time()
                with torch.no_grad():
                    output = model(q, k, v)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_times.append(time.time() - start)

            forward_time = sum(forward_times) / len(forward_times)

            # Measure backward pass
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            backward_times = []
            for _ in range(num_iterations):
                start = time.time()
                output = model(q, k, v)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.sum()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_times.append(time.time() - start)

            backward_time = sum(backward_times) / len(backward_times)

            # Get peak memory
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mb = 0.0

            # Calculate throughput
            total_tokens = batch_size * seq_len
            throughput = total_tokens / forward_time

            return BenchmarkResult(
                implementation=name,
                category=category,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time_ms=forward_time * 1000,
                backward_time_ms=backward_time * 1000,
                total_time_ms=(forward_time + backward_time) * 1000,
                peak_memory_mb=peak_memory_mb,
                throughput_tokens_per_sec=throughput,
                success=True,
            )

        except Exception as e:
            traceback.print_exc()
            return BenchmarkResult(
                implementation=name,
                category=category,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time_ms=0.0,
                backward_time_ms=0.0,
                total_time_ms=0.0,
                peak_memory_mb=0.0,
                throughput_tokens_per_sec=0.0,
                success=False,
                error=str(e),
            )

    def run_benchmarks(
        self,
        batch_sizes: List[int] = [2],
        seq_lens: List[int] = [2048, 4096],
        num_heads: int = 12,
        head_dim: int = 64,
    ):
        """Run benchmarks for all implementations."""
        implementations = self.get_implementations()

        total_impls = sum(len(impls) for impls in implementations.values())
        print(
            f"Found {total_impls} implementations across {len(implementations)} categories"
        )
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\nBenchmarking: batch_size={batch_size}, seq_len={seq_len}")
                print("-" * 60)

                for category, impls in implementations.items():
                    if impls:
                        print(f"\n{category.upper()} implementations:")
                        for impl in impls:
                            print(f"  Testing {impl['name']}...", end=" ", flush=True)

                            result = self.benchmark_implementation(
                                impl_info=impl,
                                category=category,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                num_heads=num_heads,
                                head_dim=head_dim,
                            )

                            self.results.append(result)

                            if result.success:
                                print(
                                    f"✓ {result.forward_time_ms:.1f}ms fwd, "
                                    f"{result.backward_time_ms:.1f}ms bwd, "
                                    f"{result.peak_memory_mb:.0f}MB"
                                )
                            else:
                                print(f"✗ {result.error}")

    def save_results(self, filename: str):
        """Save results to JSON."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "results": [
                {
                    "implementation": r.implementation,
                    "category": r.category,
                    "batch_size": r.batch_size,
                    "seq_len": r.seq_len,
                    "num_heads": r.num_heads,
                    "head_dim": r.head_dim,
                    "forward_time_ms": r.forward_time_ms,
                    "backward_time_ms": r.backward_time_ms,
                    "total_time_ms": r.total_time_ms,
                    "peak_memory_mb": r.peak_memory_mb,
                    "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        for category, results in sorted(categories.items()):
            print(f"\n{category.upper()} Implementations:")
            print("-" * 40)

            # Group by implementation
            impls = {}
            for r in results:
                if r.implementation not in impls:
                    impls[r.implementation] = []
                impls[r.implementation].append(r)

            for impl, impl_results in sorted(impls.items()):
                successful = [r for r in impl_results if r.success]
                if successful:
                    avg_fwd = sum(r.forward_time_ms for r in successful) / len(
                        successful
                    )
                    avg_bwd = sum(r.backward_time_ms for r in successful) / len(
                        successful
                    )
                    avg_mem = sum(r.peak_memory_mb for r in successful) / len(
                        successful
                    )
                    avg_throughput = sum(
                        r.throughput_tokens_per_sec for r in successful
                    ) / len(successful)

                    print(f"  {impl}:")
                    print(f"    Success: {len(successful)}/{len(impl_results)}")
                    print(f"    Avg Forward: {avg_fwd:.1f}ms")
                    print(f"    Avg Backward: {avg_bwd:.1f}ms")
                    print(f"    Avg Memory: {avg_mem:.0f}MB")
                    print(f"    Avg Throughput: {avg_throughput:.0f} tokens/sec")
                else:
                    print(f"  {impl}: FAILED")
                    if impl_results[0].error:
                        print(f"    Error: {impl_results[0].error}")

        # Overall statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)

        successful_results = [r for r in self.results if r.success]
        if successful_results:
            # Find best performers
            fastest_fwd = min(successful_results, key=lambda r: r.forward_time_ms)
            fastest_bwd = min(successful_results, key=lambda r: r.backward_time_ms)
            lowest_mem = min(successful_results, key=lambda r: r.peak_memory_mb)
            highest_throughput = max(
                successful_results, key=lambda r: r.throughput_tokens_per_sec
            )

            print(
                f"\nFastest Forward Pass: {fastest_fwd.implementation} ({fastest_fwd.forward_time_ms:.1f}ms)"
            )
            print(
                f"Fastest Backward Pass: {fastest_bwd.implementation} ({fastest_bwd.backward_time_ms:.1f}ms)"
            )
            print(
                f"Lowest Memory Usage: {lowest_mem.implementation} ({lowest_mem.peak_memory_mb:.0f}MB)"
            )
            print(
                f"Highest Throughput: {highest_throughput.implementation} ({highest_throughput.throughput_tokens_per_sec:.0f} tokens/sec)"
            )

            print(f"\nTotal Successful: {len(successful_results)}/{len(self.results)}")
            print(
                f"Success Rate: {len(successful_results) / len(self.results) * 100:.1f}%"
            )


def main():
    """Main benchmark function."""
    benchmark = DilatedAttentionBenchmark()

    # Run benchmarks
    benchmark.run_benchmarks(
        batch_sizes=[2], seq_lens=[2048, 4096], num_heads=12, head_dim=64
    )

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/dilated_attention_benchmark_fixed_{timestamp}.json"
    benchmark.save_results(filename)
    print(f"\nResults saved to {filename}")

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
