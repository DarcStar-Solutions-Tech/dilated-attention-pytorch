#!/usr/bin/env python3
"""
Comprehensive benchmark of all dilated attention implementations.

This script benchmarks all 24 dilated attention implementations identified
in the codebase, testing both performance and memory usage.
"""

import torch
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime
import gc

# Suppress some warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    implementation: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    forward_time: float
    backward_time: float
    peak_memory_mb: float
    success: bool
    error: Optional[str] = None
    notes: Optional[str] = None


class DilatedAttentionBenchmark:
    """Benchmark harness for all dilated attention implementations."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results: List[BenchmarkResult] = []

    def get_all_implementations(self) -> Dict[str, Any]:
        """Get all dilated attention implementations to test."""
        implementations = {}

        # 1. Core implementations using new architecture
        try:
            from dilated_attention_pytorch import DilatedAttention

            implementations["DilatedAttention"] = {
                "class": DilatedAttention,
                "type": "core",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "dropout": 0.0,
                },
            }
        except Exception as e:
            print(f"Failed to import DilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import MultiheadDilatedAttention

            implementations["MultiheadDilatedAttention"] = {
                "class": MultiheadDilatedAttention,
                "type": "multihead",
                "params": {
                    "embed_dim": 768,
                    "num_heads": 12,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "dropout": 0.0,
                },
            }
        except Exception as e:
            print(f"Failed to import MultiheadDilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import ImprovedDilatedAttention

            implementations["ImprovedDilatedAttention"] = {
                "class": ImprovedDilatedAttention,
                "type": "core",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "dropout": 0.0,
                },
            }
        except Exception as e:
            print(f"Failed to import ImprovedDilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention

            implementations["ImprovedMultiheadDilatedAttention"] = {
                "class": ImprovedMultiheadDilatedAttention,
                "type": "multihead",
                "params": {
                    "embed_dim": 768,
                    "num_heads": 12,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "dropout": 0.0,
                },
            }
        except Exception as e:
            print(f"Failed to import ImprovedMultiheadDilatedAttention: {e}")

        # 2. Distributed implementations
        try:
            from dilated_attention_pytorch.improved_distributed_dilated_attention import (
                DistributedImprovedDilatedAttention,
            )

            # Skip distributed for single GPU benchmark
            implementations["DistributedImprovedDilatedAttention"] = {
                "class": DistributedImprovedDilatedAttention,
                "type": "distributed",
                "skip": True,
                "reason": "Requires multi-GPU setup",
            }
        except Exception as e:
            print(f"Failed to import DistributedImprovedDilatedAttention: {e}")

        # 3. Ring attention implementations
        try:
            from dilated_attention_pytorch import RingDilatedAttentionProduction

            implementations["RingDilatedAttentionProduction"] = {
                "class": RingDilatedAttentionProduction,
                "type": "ring",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "ring_size": 1,
                },
            }
        except Exception as e:
            print(f"Failed to import RingDilatedAttentionProduction: {e}")

        # 4. Block-sparse implementations
        try:
            from dilated_attention_pytorch import BlockSparseRingDilatedAttention

            implementations["BlockSparseRingDilatedAttention"] = {
                "class": BlockSparseRingDilatedAttention,
                "type": "block_sparse",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "sparsity_ratio": 0.1,
                    "block_size": 64,
                },
            }
        except Exception as e:
            print(f"Failed to import BlockSparseRingDilatedAttention: {e}")

        try:
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention_fixed import (
                BlockSparseRingDilatedAttentionFixed,
            )

            implementations["BlockSparseRingDilatedAttentionFixed"] = {
                "class": BlockSparseRingDilatedAttentionFixed,
                "type": "block_sparse",
                "params": {
                    "dim": 64,
                    "heads": 12,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "sparsity_ratio": 0.1,
                    "block_size": 64,
                },
            }
        except Exception as e:
            print(f"Failed to import BlockSparseRingDilatedAttentionFixed: {e}")

        try:
            from dilated_attention_pytorch import (
                BlockSparseRingMultiheadDilatedAttention,
            )

            implementations["BlockSparseRingMultiheadDilatedAttention"] = {
                "class": BlockSparseRingMultiheadDilatedAttention,
                "type": "block_sparse_multihead",
                "params": {
                    "embed_dim": 768,
                    "num_heads": 12,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "sparsity_ratio": 0.1,
                    "block_size": 64,
                },
            }
        except Exception as e:
            print(f"Failed to import BlockSparseRingMultiheadDilatedAttention: {e}")

        try:
            from dilated_attention_pytorch import BlockSparseAdaptive

            implementations["BlockSparseAdaptive"] = {
                "class": BlockSparseAdaptive,
                "type": "block_sparse_adaptive",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import BlockSparseAdaptive: {e}")

        try:
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
                BlockSparseRingDilatedAttentionHilbertPostPattern,
            )

            implementations["BlockSparseRingDilatedAttentionHilbertPostPattern"] = {
                "class": BlockSparseRingDilatedAttentionHilbertPostPattern,
                "type": "block_sparse_hilbert",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                    "sparsity_ratio": 0.1,
                    "block_size": 64,
                },
            }
        except Exception as e:
            print(
                f"Failed to import BlockSparseRingDilatedAttentionHilbertPostPattern: {e}"
            )

        # 5. Head-parallel implementations
        try:
            from dilated_attention_pytorch.head_parallel_dilated_attention import (
                HeadParallelDilatedAttentionOptimized,
            )

            implementations["HeadParallelDilatedAttentionOptimized"] = {
                "class": HeadParallelDilatedAttentionOptimized,
                "type": "head_parallel",
                "params": {
                    "num_heads": 12,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import HeadParallelDilatedAttentionOptimized: {e}")

        # 6. Kernel implementations
        try:
            from dilated_attention_pytorch.kernels.hilbert_dilated_attention import (
                HilbertDilatedAttention,
            )

            implementations["HilbertDilatedAttention"] = {
                "class": HilbertDilatedAttention,
                "type": "kernel",
                "params": {
                    "num_heads": 12,
                    "head_dim": 64,
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import HilbertDilatedAttention: {e}")

        # 7. Fixed/wrapper implementations
        try:
            from dilated_attention_pytorch.ring_dilated_attention_fixed import (
                RingDilatedAttentionFixed,
            )

            implementations["RingDilatedAttentionFixed"] = {
                "class": RingDilatedAttentionFixed,
                "type": "ring_fixed",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import RingDilatedAttentionFixed: {e}")

        # 8. Refactored implementations
        try:
            from dilated_attention_pytorch.ring_dilated_attention_refactored import (
                RingDilatedAttentionRefactored,
            )

            implementations["RingDilatedAttentionRefactored"] = {
                "class": RingDilatedAttentionRefactored,
                "type": "ring_refactored",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import RingDilatedAttentionRefactored: {e}")

        # 9. Ring Hilbert implementations
        try:
            from dilated_attention_pytorch.ring_hilbert_dilated_attention import (
                RingDilatedAttentionHilbert,
            )

            implementations["RingDilatedAttentionHilbert"] = {
                "class": RingDilatedAttentionHilbert,
                "type": "ring_hilbert",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import RingDilatedAttentionHilbert: {e}")

        # 10. V2 collective implementation
        try:
            from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
                RingDilatedAttentionV2Collective,
            )

            implementations["RingDilatedAttentionV2Collective"] = {
                "class": RingDilatedAttentionV2Collective,
                "type": "ring_v2",
                "params": {
                    "segment_lengths": [512, 1024, 2048],
                    "dilation_rates": [1, 2, 4],
                },
            }
        except Exception as e:
            print(f"Failed to import RingDilatedAttentionV2Collective: {e}")

        return implementations

    def create_test_inputs(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        input_type: str = "core",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create test inputs based on implementation type."""
        if input_type in [
            "core",
            "ring",
            "block_sparse",
            "head_parallel",
            "kernel",
            "ring_fixed",
            "ring_refactored",
            "ring_hilbert",
            "ring_v2",
            "block_sparse_hilbert",
            "block_sparse_adaptive",
        ]:
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
        elif input_type in ["multihead", "block_sparse_multihead"]:
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
            raise ValueError(f"Unknown input type: {input_type}")

        return q, k, v

    def measure_memory(self) -> float:
        """Measure current GPU memory usage in MB."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    def benchmark_implementation(
        self,
        name: str,
        impl_info: Dict[str, Any],
        batch_size: int = 2,
        seq_len: int = 2048,
        num_heads: int = 12,
        head_dim: int = 64,
        num_warmup: int = 3,
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark a single implementation."""

        # Skip if marked to skip
        if impl_info.get("skip", False):
            return BenchmarkResult(
                implementation=name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time=0.0,
                backward_time=0.0,
                peak_memory_mb=0.0,
                success=False,
                error=impl_info.get("reason", "Skipped"),
            )

        try:
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()

            # Create model
            model_class = impl_info["class"]
            params = impl_info.get("params", {}).copy()

            # Adjust parameters based on test configuration
            if "num_heads" in params:
                params["num_heads"] = num_heads
            if "head_dim" in params:
                params["head_dim"] = head_dim
            if "embed_dim" in params:
                params["embed_dim"] = num_heads * head_dim

            # Adjust segment lengths if needed
            if "segment_lengths" in params:
                # Make sure sequence length is divisible by largest segment
                max_segment = max(params["segment_lengths"])
                if seq_len % max_segment != 0:
                    seq_len = ((seq_len + max_segment - 1) // max_segment) * max_segment

            model = model_class(**params).to(self.device)
            model.eval()

            # Create inputs
            q, k, v = self.create_test_inputs(
                batch_size, seq_len, num_heads, head_dim, impl_info["type"]
            )

            # Warmup
            for _ in range(num_warmup):
                with torch.no_grad():
                    _ = model(q, k, v)

            # Measure forward pass
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            for _ in range(num_iterations):
                with torch.no_grad():
                    output = model(q, k, v)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_time = (time.time() - start_time) / num_iterations

            # Measure backward pass
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            for _ in range(num_iterations):
                output = model(q, k, v)
                if isinstance(output, tuple):
                    output = output[0]
                loss = output.sum()
                loss.backward()

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            backward_time = (time.time() - start_time) / num_iterations

            # Measure peak memory
            peak_memory = self.measure_memory()

            return BenchmarkResult(
                implementation=name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time=forward_time,
                backward_time=backward_time,
                peak_memory_mb=peak_memory,
                success=True,
            )

        except Exception as e:
            return BenchmarkResult(
                implementation=name,
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                forward_time=0.0,
                backward_time=0.0,
                peak_memory_mb=0.0,
                success=False,
                error=str(e),
            )

    def run_benchmarks(
        self,
        batch_sizes: List[int] = [1, 2, 4],
        seq_lens: List[int] = [2048, 4096, 8192],
        num_heads: int = 12,
        head_dim: int = 64,
    ):
        """Run benchmarks for all implementations."""
        implementations = self.get_all_implementations()

        print(f"Found {len(implementations)} implementations to benchmark")
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\nBenchmarking with batch_size={batch_size}, seq_len={seq_len}")
                print("-" * 60)

                for name, impl_info in implementations.items():
                    print(f"Testing {name}...", end=" ", flush=True)

                    result = self.benchmark_implementation(
                        name=name,
                        impl_info=impl_info,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_heads=num_heads,
                        head_dim=head_dim,
                    )

                    self.results.append(result)

                    if result.success:
                        print(
                            f"✓ Forward: {result.forward_time * 1000:.2f}ms, "
                            f"Backward: {result.backward_time * 1000:.2f}ms, "
                            f"Memory: {result.peak_memory_mb:.1f}MB"
                        )
                    else:
                        print(f"✗ {result.error}")

    def save_results(self, filename: str):
        """Save results to JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "device": str(self.device),
            "results": [
                {
                    "implementation": r.implementation,
                    "batch_size": r.batch_size,
                    "seq_len": r.seq_len,
                    "num_heads": r.num_heads,
                    "head_dim": r.head_dim,
                    "forward_time_ms": r.forward_time * 1000,
                    "backward_time_ms": r.backward_time * 1000,
                    "total_time_ms": (r.forward_time + r.backward_time) * 1000,
                    "peak_memory_mb": r.peak_memory_mb,
                    "success": r.success,
                    "error": r.error,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group by implementation
        impl_results = {}
        for r in self.results:
            if r.implementation not in impl_results:
                impl_results[r.implementation] = []
            impl_results[r.implementation].append(r)

        # Print summary for each implementation
        for impl, results in sorted(impl_results.items()):
            successful = [r for r in results if r.success]
            if successful:
                avg_forward = sum(r.forward_time for r in successful) / len(successful)
                avg_backward = sum(r.backward_time for r in successful) / len(
                    successful
                )
                avg_memory = sum(r.peak_memory_mb for r in successful) / len(successful)

                print(f"\n{impl}:")
                print(f"  Success rate: {len(successful)}/{len(results)}")
                print(f"  Avg forward time: {avg_forward * 1000:.2f}ms")
                print(f"  Avg backward time: {avg_backward * 1000:.2f}ms")
                print(f"  Avg memory usage: {avg_memory:.1f}MB")
            else:
                print(f"\n{impl}:")
                print("  All tests failed")
                if results[0].error:
                    print(f"  Error: {results[0].error}")


def main():
    """Main benchmark function."""
    benchmark = DilatedAttentionBenchmark()

    # Run benchmarks with different configurations
    benchmark.run_benchmarks(
        batch_sizes=[2], seq_lens=[2048, 4096], num_heads=12, head_dim=64
    )

    # Save results
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmarks/dilated_attention_benchmark_{timestamp}.json"
    benchmark.save_results(filename)
    print(f"\nResults saved to {filename}")

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
