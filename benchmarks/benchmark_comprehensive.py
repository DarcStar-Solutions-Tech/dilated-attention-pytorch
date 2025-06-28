#!/usr/bin/env python3
"""
Comprehensive performance benchmark for all dilated attention implementations.
Tests performance, memory usage, and maximum sequence lengths.
"""

import gc
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
)

try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.block_sparse_optimized import BlockSparseOptimized
    from dilated_attention_pytorch.block_sparse_hierarchical import (
        BlockSparseHierarchical,
    )
    from dilated_attention_pytorch.block_sparse_adaptive import (
        create_adaptive_block_sparse,
        AdaptiveConfig,
    )

    HAS_BLOCK_SPARSE = True
except ImportError:
    HAS_BLOCK_SPARSE = False
    print("Note: Block sparse implementations not available")

try:
    from dilated_attention_pytorch.ring_dilated_attention_v2 import (
        RingDilatedAttentionV2,
    )

    HAS_RING_V2 = True
except ImportError:
    HAS_RING_V2 = False


class ComprehensiveBenchmark:
    """Comprehensive benchmarking for all implementations."""

    def __init__(self, device="cuda", dtype=torch.float16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.results = []

    def measure_performance(
        self,
        model,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        num_runs: int = 10,
        warmup: int = 3,
    ) -> Dict:
        """Measure forward pass performance and memory usage."""
        # Create inputs
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=self.device, dtype=self.dtype)
        k = torch.randn(shape, device=self.device, dtype=self.dtype)
        v = torch.randn(shape, device=self.device, dtype=self.dtype)

        # Move model to device
        if hasattr(model, "to"):
            model = model.to(self.device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(q, k, v)

        # Clear cache
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Measure forward pass
        times = []
        for _ in range(num_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(q, k, v)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Get memory stats
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        return {
            "mean_time": np.mean(times) * 1000,  # ms
            "std_time": np.std(times) * 1000,
            "peak_memory": peak_memory,
            "output_shape": output.shape,
        }

    def test_maximum_sequence_length(
        self,
        model_fn,
        segment_lengths: List[int],
        dilation_rates: List[int],
        num_heads: int = 8,
        head_dim: int = 64,
        batch_size: int = 1,
    ) -> int:
        """Find maximum sequence length that can be processed."""
        max_seq_len = 0
        seq_len = 1024

        while seq_len <= 1_000_000:  # Test up to 1M tokens
            try:
                # Ensure sequence length is valid
                if seq_len % max(segment_lengths) != 0:
                    seq_len = ((seq_len // max(segment_lengths)) + 1) * max(
                        segment_lengths
                    )

                # Create model
                model = model_fn(segment_lengths, dilation_rates)
                if hasattr(model, "to"):
                    model = model.to(self.device)

                # Test forward pass
                shape = (batch_size, seq_len, num_heads, head_dim)
                q = torch.randn(shape, device=self.device, dtype=self.dtype)
                k = torch.randn(shape, device=self.device, dtype=self.dtype)
                v = torch.randn(shape, device=self.device, dtype=self.dtype)

                with torch.no_grad():
                    _ = model(q, k, v)

                max_seq_len = seq_len
                seq_len *= 2

                # Cleanup
                del q, k, v, model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                break

        return max_seq_len

    def run_comprehensive_benchmark(self):
        """Run benchmarks for all implementations."""
        # Test configurations
        configs = [
            {
                "name": "Small",
                "seq_len": 2048,
                "batch_size": 2,
                "num_heads": 8,
                "head_dim": 64,
            },
            {
                "name": "Medium",
                "seq_len": 8192,
                "batch_size": 1,
                "num_heads": 8,
                "head_dim": 64,
            },
            {
                "name": "Large",
                "seq_len": 16384,
                "batch_size": 1,
                "num_heads": 12,
                "head_dim": 64,
            },
        ]

        segment_lengths = [2048, 4096, 8192]
        dilation_rates = [1, 2, 4]

        implementations = []

        # Standard implementations
        implementations.extend(
            [
                ("DilatedAttention", lambda sl, dr: DilatedAttention(sl, dr)),
                (
                    "ImprovedDilatedAttention",
                    lambda sl, dr: ImprovedDilatedAttention(sl, dr),
                ),
            ]
        )

        # Block sparse implementations
        if HAS_BLOCK_SPARSE:
            sparse_config = SparsePatternConfig(
                pattern_type="local_window",
                sparsity_ratio=0.9,
                block_size=64,
                local_window_size=256,
            )

            implementations.extend(
                [
                    (
                        "BlockSparseRing",
                        lambda sl, dr: BlockSparseRingDilatedAttention(
                            sl,
                            dr,
                            sparse_config,
                            ring_size=1,
                            device=self.device,
                            dtype=self.dtype,
                        ),
                    ),
                    (
                        "BlockSparseOptimized",
                        lambda sl, dr: BlockSparseOptimized(sl, dr, sparse_config),
                    ),
                    (
                        "BlockSparseHierarchical",
                        lambda sl, dr: BlockSparseHierarchical(
                            sl, dr, device=self.device, dtype=self.dtype
                        ),
                    ),
                    (
                        "BlockSparseAdaptive",
                        lambda sl, dr: create_adaptive_block_sparse(
                            embed_dim=512,
                            num_heads=8,
                            segment_lengths=sl,
                            dilation_rates=dr,
                            adaptive_config=AdaptiveConfig(base_sparsity=0.9),
                            device=self.device,
                            dtype=self.dtype,
                        ),
                    ),
                ]
            )

        # Ring V2 implementation
        if HAS_RING_V2:
            implementations.append(
                (
                    "RingDilatedAttentionV2",
                    lambda sl, dr: RingDilatedAttentionV2(
                        sl, dr, ring_size=4, device=self.device
                    ),
                )
            )

        print("=" * 80)
        print("COMPREHENSIVE DILATED ATTENTION BENCHMARK")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Data type: {self.dtype}")
        print(f"Generated: {datetime.utcnow().isoformat()}Z\n")

        # Performance benchmarks
        print("PERFORMANCE BENCHMARKS")
        print("-" * 80)

        for config in configs:
            print(f"\n{config['name']} Configuration (seq_len={config['seq_len']}):")

            for impl_name, impl_fn in implementations:
                try:
                    model = impl_fn(segment_lengths, dilation_rates)
                    result = self.measure_performance(
                        model,
                        config["batch_size"],
                        config["seq_len"],
                        config["num_heads"],
                        config["head_dim"],
                    )

                    print(
                        f"  {impl_name:25} {result['mean_time']:8.2f} ± {result['std_time']:5.2f} ms, "
                        f"Memory: {result['peak_memory']:8.2f} MB"
                    )

                    self.results.append(
                        {
                            "implementation": impl_name,
                            "config": config["name"],
                            **config,
                            **result,
                        }
                    )

                    del model
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"  {impl_name:25} FAILED: {str(e)}")

        # Maximum sequence length tests
        print("\n" + "=" * 80)
        print("MAXIMUM SEQUENCE LENGTH TESTS")
        print("-" * 80)

        for impl_name, impl_fn in implementations:
            try:
                max_len = self.test_maximum_sequence_length(
                    impl_fn, segment_lengths, dilation_rates
                )
                print(f"{impl_name:25} Max sequence length: {max_len:,} tokens")

            except Exception as e:
                print(f"{impl_name:25} FAILED: {str(e)}")

        # Save results
        self._save_results()

    def _save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
        output_dir = Path("docs/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"comprehensive-benchmark-{timestamp}.md"

        with open(output_file, "w") as f:
            f.write("# Comprehensive Dilated Attention Benchmark\n\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            f.write("## Configuration\n")
            f.write(f"- Device: {self.device}\n")
            f.write(f"- Data type: {self.dtype}\n\n")

            f.write("## Performance Results\n\n")

            # Group by configuration
            for config_name in ["Small", "Medium", "Large"]:
                config_results = [r for r in self.results if r["config"] == config_name]
                if not config_results:
                    continue

                f.write(f"### {config_name} Configuration\n")
                f.write(f"- Sequence length: {config_results[0]['seq_len']}\n")
                f.write(f"- Batch size: {config_results[0]['batch_size']}\n")
                f.write(f"- Num heads: {config_results[0]['num_heads']}\n")
                f.write(f"- Head dim: {config_results[0]['head_dim']}\n\n")

                f.write("| Implementation | Time (ms) | Memory (MB) |\n")
                f.write("|----------------|-----------|-------------|\n")

                for result in config_results:
                    f.write(
                        f"| {result['implementation']} | "
                        f"{result['mean_time']:.2f} ± {result['std_time']:.2f} | "
                        f"{result['peak_memory']:.2f} |\n"
                    )

                f.write("\n")

            # Find best performers
            if self.results:
                best_time = min(self.results, key=lambda x: x["mean_time"])
                best_memory = min(self.results, key=lambda x: x["peak_memory"])

                f.write("## Summary\n\n")
                f.write(
                    f"- **Fastest**: {best_time['implementation']} "
                    f"({best_time['mean_time']:.2f} ms)\n"
                )
                f.write(
                    f"- **Most Memory Efficient**: {best_memory['implementation']} "
                    f"({best_memory['peak_memory']:.2f} MB)\n"
                )

        print(f"\nResults saved to: {output_file}")


def main():
    """Run comprehensive benchmarks."""
    benchmark = ComprehensiveBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
