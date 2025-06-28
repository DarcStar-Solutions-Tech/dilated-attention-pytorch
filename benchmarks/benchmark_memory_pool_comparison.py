#!/usr/bin/env python3
"""
Comprehensive benchmark comparing memory pool performance across attention implementations.

This benchmark measures:
1. Allocation/deallocation overhead
2. Memory reuse efficiency
3. Peak memory usage
4. Throughput with/without pools
5. Scaling behavior
"""

import argparse
import gc
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from dilated_attention_pytorch import (
    DilatedAttention,
    MultiheadDilatedAttention,
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)
from dilated_attention_pytorch.improved_dilated_attention_v2 import (
    ImprovedDilatedAttentionV2,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    implementation: str
    config: str
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int

    # Timing metrics
    avg_forward_time: float
    std_forward_time: float
    first_forward_time: float

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_allocated_mb: float
    memory_reserved_mb: float

    # Pool metrics
    pool_hits: int
    pool_misses: int
    pool_hit_rate: float
    pool_allocations: int

    # Throughput
    sequences_per_second: float
    tokens_per_second: float


class MemoryPoolBenchmark:
    """Benchmark suite for memory pool integration."""

    def __init__(
        self,
        device: str = "cuda",
        warmup_steps: int = 5,
        benchmark_steps: int = 20,
        profile: bool = False,
    ):
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self.benchmark_steps = benchmark_steps
        self.profile = profile
        self.results: List[BenchmarkResult] = []

    def clear_memory(self):
        """Clear GPU memory and caches."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def get_memory_stats(self) -> Tuple[float, float, float]:
        """Get current memory statistics in MB."""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**2
            return allocated, reserved, max_allocated
        return 0.0, 0.0, 0.0

    def get_pool_stats(self, module: nn.Module) -> Dict[str, int]:
        """Extract pool statistics from module."""
        stats = {
            "pool_hits": 0,
            "pool_misses": 0,
            "pool_allocations": 0,
        }

        # Try different ways to get pool stats
        if hasattr(module, "get_buffer_stats"):
            buffer_stats = module.get_buffer_stats()
            stats["pool_hits"] = buffer_stats.get("cache_hits", 0)
            stats["pool_misses"] = buffer_stats.get("cache_misses", 0)
        elif hasattr(module, "memory_pool"):
            pool = module.memory_pool
            if hasattr(pool, "hits"):
                stats["pool_hits"] = pool.hits
            if hasattr(pool, "misses"):
                stats["pool_misses"] = pool.misses
            if hasattr(pool, "total_allocations"):
                stats["pool_allocations"] = pool.total_allocations

        return stats

    def benchmark_implementation(
        self,
        impl_name: str,
        attention_class: type,
        attention_kwargs: Dict,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        multihead: bool = False,
    ) -> Optional[BenchmarkResult]:
        """Benchmark a single attention implementation."""

        # Skip certain combinations
        if impl_name.startswith("Ring") and self.device.type == "cpu":
            return None

        try:
            self.clear_memory()

            # Create module
            if multihead:
                embed_dim = num_heads * head_dim
                attention = attention_class(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                    dilation_rates=[1, 2, 4],
                    batch_first=True,
                    **attention_kwargs,
                ).to(self.device)

                # Create input
                x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

                # Define forward function
                def forward_fn(attn_module=attention):
                    return attn_module(x, x, x)[0]

            else:
                attention = attention_class(
                    segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                    dilation_rates=[1, 2, 4],
                    **attention_kwargs,
                ).to(self.device)

                # Create inputs
                shape = (batch_size, seq_len, num_heads, head_dim)
                q = torch.randn(shape, device=self.device)
                k = torch.randn(shape, device=self.device)
                v = torch.randn(shape, device=self.device)

                # Define forward function
                def forward_fn(attn_module=attention):
                    return attn_module(q, k, v)

            # Reset memory stats
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            # Warmup
            for _ in range(self.warmup_steps):
                _ = forward_fn()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            # Get initial pool stats
            initial_pool_stats = self.get_pool_stats(attention)

            # Benchmark
            forward_times = []
            memory_usage = []

            for i in range(self.benchmark_steps):
                # Time forward pass
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start_time = time.perf_counter()
                _ = forward_fn()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                forward_times.append(end_time - start_time)

                # Record memory
                allocated, reserved, max_allocated = self.get_memory_stats()
                memory_usage.append(allocated)

            # Get final pool stats
            final_pool_stats = self.get_pool_stats(attention)

            # Calculate metrics
            avg_forward_time = sum(forward_times[1:]) / (len(forward_times) - 1)
            std_forward_time = torch.std(torch.tensor(forward_times[1:])).item()
            first_forward_time = forward_times[0]

            peak_memory = max(memory_usage) if memory_usage else 0
            avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

            pool_hits = final_pool_stats["pool_hits"] - initial_pool_stats["pool_hits"]
            pool_misses = (
                final_pool_stats["pool_misses"] - initial_pool_stats["pool_misses"]
            )
            pool_allocations = (
                final_pool_stats["pool_allocations"]
                - initial_pool_stats["pool_allocations"]
            )

            total_accesses = pool_hits + pool_misses
            pool_hit_rate = pool_hits / total_accesses if total_accesses > 0 else 0

            sequences_per_second = 1.0 / avg_forward_time
            tokens_per_second = (batch_size * seq_len) / avg_forward_time

            # Create result
            result = BenchmarkResult(
                implementation=impl_name,
                config=str(attention_kwargs),
                batch_size=batch_size,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                avg_forward_time=avg_forward_time * 1000,  # Convert to ms
                std_forward_time=std_forward_time * 1000,
                first_forward_time=first_forward_time * 1000,
                peak_memory_mb=peak_memory,
                avg_memory_mb=avg_memory,
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                pool_hits=pool_hits,
                pool_misses=pool_misses,
                pool_hit_rate=pool_hit_rate,
                pool_allocations=pool_allocations,
                sequences_per_second=sequences_per_second,
                tokens_per_second=tokens_per_second,
            )

            # Cleanup
            if hasattr(attention, "cleanup_buffers"):
                attention.cleanup_buffers()
            del attention
            self.clear_memory()

            return result

        except Exception as e:
            print(f"Error benchmarking {impl_name}: {e}")
            return None

    def run_comparison(
        self,
        batch_sizes: List[int],
        seq_lens: List[int],
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        """Run comparison across implementations and configurations."""

        # Define implementations to test
        implementations = [
            # Standard implementations
            (
                "DilatedAttention-NoPool",
                DilatedAttention,
                {"enable_memory_pool": False},
            ),
            ("DilatedAttention-Pool", DilatedAttention, {"enable_memory_pool": True}),
            (
                "ImprovedDilated-NoPool",
                ImprovedDilatedAttention,
                {"enable_memory_pool": False},
            ),
            (
                "ImprovedDilated-Pool",
                ImprovedDilatedAttention,
                {"enable_memory_pool": True},
            ),
            (
                "ImprovedDilated-LightPool",
                ImprovedDilatedAttention,
                {"enable_memory_pool": True, "lightweight_pool": True},
            ),
            # V2 with buffer manager
            (
                "ImprovedDilatedV2-NoMgr",
                ImprovedDilatedAttentionV2,
                {"enable_buffer_manager": False},
            ),
            (
                "ImprovedDilatedV2-Mgr",
                ImprovedDilatedAttentionV2,
                {"enable_buffer_manager": True, "enable_buffer_reuse": True},
            ),
            (
                "ImprovedDilatedV2-MgrPrealloc",
                ImprovedDilatedAttentionV2,
                {
                    "enable_buffer_manager": True,
                    "enable_buffer_reuse": True,
                    "enable_preallocation": True,
                },
            ),
            # Ring attention
            (
                "RingDilatedV2-NoPool",
                RingDilatedAttentionV2,
                {"enable_memory_pool": False},
            ),
            (
                "RingDilatedV2-Pool",
                RingDilatedAttentionV2,
                {"enable_memory_pool": True},
            ),
            # Block sparse
            (
                "BlockSparse-NoPool",
                BlockSparseRingDilatedAttention,
                {"enable_memory_pool": False, "sparsity_ratio": 0.1},
            ),
            (
                "BlockSparse-Pool",
                BlockSparseRingDilatedAttention,
                {"enable_memory_pool": True, "sparsity_ratio": 0.1},
            ),
        ]

        # Multihead implementations
        multihead_implementations = [
            (
                "MultiheadDilated-NoPool",
                MultiheadDilatedAttention,
                {"enable_memory_pool": False},
            ),
            (
                "MultiheadDilated-Pool",
                MultiheadDilatedAttention,
                {"enable_memory_pool": True},
            ),
            (
                "ImprovedMultihead-NoPool",
                ImprovedMultiheadDilatedAttention,
                {"enable_memory_pool": False},
            ),
            (
                "ImprovedMultihead-Pool",
                ImprovedMultiheadDilatedAttention,
                {"enable_memory_pool": True},
            ),
        ]

        # Run benchmarks
        total_configs = (
            len(batch_sizes)
            * len(seq_lens)
            * (len(implementations) + len(multihead_implementations))
        )
        pbar = tqdm(total=total_configs, desc="Benchmarking")

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                # Test standard implementations
                for impl_name, impl_class, impl_kwargs in implementations:
                    result = self.benchmark_implementation(
                        impl_name,
                        impl_class,
                        impl_kwargs,
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        multihead=False,
                    )
                    if result:
                        self.results.append(result)
                    pbar.update(1)

                # Test multihead implementations
                for impl_name, impl_class, impl_kwargs in multihead_implementations:
                    result = self.benchmark_implementation(
                        impl_name,
                        impl_class,
                        impl_kwargs,
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        multihead=True,
                    )
                    if result:
                        self.results.append(result)
                    pbar.update(1)

        pbar.close()

    def analyze_results(self) -> Dict:
        """Analyze benchmark results."""
        _ = defaultdict(list)

        # Group by implementation
        impl_groups = defaultdict(list)
        for result in self.results:
            impl_groups[result.implementation].append(result)

        # Compare pool vs no-pool
        comparisons = []
        pool_implementations = [
            impl for impl in impl_groups.keys() if "Pool" in impl or "Mgr" in impl
        ]

        for pool_impl in pool_implementations:
            # Find corresponding no-pool implementation
            base_impl = pool_impl.replace("-Pool", "-NoPool").replace("-Mgr", "-NoMgr")
            base_impl = base_impl.replace("-LightPool", "-NoPool").replace(
                "-MgrPrealloc", "-NoMgr"
            )

            if base_impl in impl_groups:
                pool_results = impl_groups[pool_impl]
                base_results = impl_groups[base_impl]

                # Match results by configuration
                for pool_result in pool_results:
                    for base_result in base_results:
                        if (
                            pool_result.batch_size == base_result.batch_size
                            and pool_result.seq_len == base_result.seq_len
                        ):
                            speedup = (
                                base_result.avg_forward_time
                                / pool_result.avg_forward_time
                            )
                            memory_reduction = 1 - (
                                pool_result.peak_memory_mb / base_result.peak_memory_mb
                            )

                            comparison = {
                                "implementation": pool_impl,
                                "batch_size": pool_result.batch_size,
                                "seq_len": pool_result.seq_len,
                                "speedup": speedup,
                                "memory_reduction_pct": memory_reduction * 100,
                                "pool_hit_rate": pool_result.pool_hit_rate,
                                "base_time_ms": base_result.avg_forward_time,
                                "pool_time_ms": pool_result.avg_forward_time,
                                "base_memory_mb": base_result.peak_memory_mb,
                                "pool_memory_mb": pool_result.peak_memory_mb,
                            }
                            comparisons.append(comparison)

        return {
            "raw_results": self.results,
            "comparisons": comparisons,
            "summary": self._generate_summary(comparisons),
        }

    def _generate_summary(self, comparisons: List[Dict]) -> Dict:
        """Generate summary statistics."""
        if not comparisons:
            return {}

        summary = {}

        # Group by implementation
        impl_comparisons = defaultdict(list)
        for comp in comparisons:
            impl_comparisons[comp["implementation"]].append(comp)

        # Calculate statistics per implementation
        for impl, comps in impl_comparisons.items():
            speedups = [c["speedup"] for c in comps]
            memory_reductions = [c["memory_reduction_pct"] for c in comps]
            hit_rates = [c["pool_hit_rate"] for c in comps]

            summary[impl] = {
                "avg_speedup": sum(speedups) / len(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "avg_memory_reduction_pct": sum(memory_reductions)
                / len(memory_reductions),
                "avg_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
                "num_comparisons": len(comps),
            }

        return summary

    def save_results(self, filename: str):
        """Save results to JSON file."""
        results_dict = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "warmup_steps": self.warmup_steps,
                "benchmark_steps": self.benchmark_steps,
            },
            "results": [
                {
                    "implementation": r.implementation,
                    "config": r.config,
                    "batch_size": r.batch_size,
                    "seq_len": r.seq_len,
                    "num_heads": r.num_heads,
                    "head_dim": r.head_dim,
                    "avg_forward_time_ms": r.avg_forward_time,
                    "std_forward_time_ms": r.std_forward_time,
                    "first_forward_time_ms": r.first_forward_time,
                    "peak_memory_mb": r.peak_memory_mb,
                    "avg_memory_mb": r.avg_memory_mb,
                    "memory_allocated_mb": r.memory_allocated_mb,
                    "memory_reserved_mb": r.memory_reserved_mb,
                    "pool_hits": r.pool_hits,
                    "pool_misses": r.pool_misses,
                    "pool_hit_rate": r.pool_hit_rate,
                    "pool_allocations": r.pool_allocations,
                    "sequences_per_second": r.sequences_per_second,
                    "tokens_per_second": r.tokens_per_second,
                }
                for r in self.results
            ],
            "analysis": self.analyze_results(),
        }

        with open(filename, "w") as f:
            json.dump(results_dict, f, indent=2)

    def print_summary(self):
        """Print summary of results."""
        analysis = self.analyze_results()

        print("\n" + "=" * 80)
        print("MEMORY POOL BENCHMARK SUMMARY")
        print("=" * 80)

        summary = analysis.get("summary", {})
        if not summary:
            print("No comparison data available.")
            return

        # Sort by average speedup
        sorted_impls = sorted(
            summary.items(), key=lambda x: x[1]["avg_speedup"], reverse=True
        )

        for impl, stats in sorted_impls:
            print(f"\n{impl}:")
            print(f"  Average Speedup: {stats['avg_speedup']:.2f}x")
            print(
                f"  Speedup Range: {stats['min_speedup']:.2f}x - {stats['max_speedup']:.2f}x"
            )
            print(
                f"  Average Memory Reduction: {stats['avg_memory_reduction_pct']:.1f}%"
            )
            print(f"  Average Cache Hit Rate: {stats['avg_hit_rate']:.1%}")
            print(f"  Number of Configurations: {stats['num_comparisons']}")

        # Print detailed comparisons for best performing
        print("\n" + "-" * 80)
        print("DETAILED COMPARISONS (Top 5 by Speedup):")
        print("-" * 80)

        comparisons = analysis.get("comparisons", [])
        sorted_comparisons = sorted(
            comparisons, key=lambda x: x["speedup"], reverse=True
        )[:5]

        for comp in sorted_comparisons:
            print(
                f"\n{comp['implementation']} (B={comp['batch_size']}, L={comp['seq_len']}):"
            )
            print(
                f"  Speedup: {comp['speedup']:.2f}x ({comp['base_time_ms']:.1f}ms → {comp['pool_time_ms']:.1f}ms)"
            )
            print(
                f"  Memory: {comp['memory_reduction_pct']:.1f}% reduction ({comp['base_memory_mb']:.1f}MB → {comp['pool_memory_mb']:.1f}MB)"
            )
            print(f"  Cache Hit Rate: {comp['pool_hit_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory pool integration")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")
    parser.add_argument(
        "--benchmark-steps", type=int, default=20, help="Benchmark steps"
    )
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create benchmark
    benchmark = MemoryPoolBenchmark(
        device=args.device,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        profile=args.profile,
    )

    # Run comparison
    print(f"Running memory pool benchmarks on {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Heads: {args.num_heads}, Head dim: {args.head_dim}")

    benchmark.run_comparison(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    # Save results
    if args.output:
        benchmark.save_results(args.output)
        print(f"\nResults saved to {args.output}")

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
