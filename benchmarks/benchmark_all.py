#!/usr/bin/env python3
"""
Comprehensive benchmark script for all dilated attention implementations.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

# Import unified benchmark output management
sys.path.insert(0, str(Path(__file__).parent))
from core import BenchmarkOutputManager

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
    RingDilatedAttention,
    RingMultiheadDilatedAttention,
)

# Import block sparse implementations
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
        BlockSparseRingMultiheadDilatedAttention,
    )

    HAS_BLOCK_SPARSE = True
except ImportError:
    HAS_BLOCK_SPARSE = False
    print("Note: Block sparse implementations not available")

# Import ImprovedMultiheadDilatedAttention directly from module
try:
    from dilated_attention_pytorch.improved_multihead_dilated_attention import (
        ImprovedMultiheadDilatedAttention,
    )

    HAS_IMPROVED_MHA = True
except ImportError:
    HAS_IMPROVED_MHA = False
    print("Note: ImprovedMultiheadDilatedAttention not available")


class BenchmarkRunner:
    """Comprehensive benchmark runner for all attention implementations."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.results = []
        self.all_results_by_impl = {}  # Store results by implementation name

    def benchmark_implementation(
        self,
        name: str,
        attention_module,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        warmup_steps: int = 3,
        benchmark_steps: int = 10,
        is_multihead: bool = False,
    ) -> dict:
        """Benchmark a single implementation."""

        # Create input tensors
        if is_multihead:
            # Multihead expects (batch, seq, embed_dim)
            embed_dim = num_heads * head_dim
            query = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
            )
            key = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
            )
            value = torch.randn(
                batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
            )
        else:
            # Core attention expects (batch, seq, heads, head_dim)
            query = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            key = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            value = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        # Move module to device and dtype
        attention_module = attention_module.to(self.device, self.dtype)

        # Warmup
        for _ in range(warmup_steps):
            with torch.no_grad():
                try:
                    output = attention_module(query, key, value)
                    if isinstance(output, tuple):
                        output = output[0]
                except Exception as e:
                    return {
                        "name": name,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                        "error": str(e),
                        "success": False,
                    }

        # Clear cache and synchronize
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark forward pass
        forward_times = []
        peak_memory = 0

        for _ in range(benchmark_steps):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

            start_time = time.perf_counter()

            with torch.no_grad():
                output = attention_module(query, key, value)
                if isinstance(output, tuple):
                    output = output[0]

            if self.device.type == "cuda":
                torch.cuda.synchronize()
                end_mem = torch.cuda.memory_allocated()
                peak_memory = max(peak_memory, end_mem - start_mem)

            end_time = time.perf_counter()
            forward_times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(forward_times)
        std_time = np.std(forward_times)
        min_time = np.min(forward_times)

        # Calculate throughput (sequences per second)
        throughput = batch_size / mean_time

        # Memory in MB
        peak_memory_mb = (
            peak_memory / (1024 * 1024) if self.device.type == "cuda" else 0
        )

        return {
            "name": name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "mean_time_ms": mean_time * 1000,
            "std_time_ms": std_time * 1000,
            "min_time_ms": min_time * 1000,
            "throughput": throughput,
            "peak_memory_mb": peak_memory_mb,
            "success": True,
        }

    def run_benchmarks(
        self,
        batch_sizes: list[int] = [1, 2, 4],
        seq_lens: list[int] = [2048, 4096, 8192],
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        """Run benchmarks for all implementations."""

        # Common parameters
        segment_lengths = [1024, 2048, 4096]
        dilation_rates = [1, 2, 4]
        dropout = 0.0

        print(f"\nRunning benchmarks on {self.device} with dtype={self.dtype}")
        print("=" * 80)

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                print(f"\nBenchmarking batch_size={batch_size}, seq_len={seq_len}")
                print("-" * 60)

                # Adjust segment lengths for sequence length
                adjusted_segments = [min(s, seq_len // 2) for s in segment_lengths]

                # Core implementations
                implementations = [
                    (
                        "DilatedAttention",
                        DilatedAttention(
                            segment_lengths=adjusted_segments,
                            dilation_rates=dilation_rates,
                            attention_dropout=dropout,
                        ),
                        False,
                    ),
                    (
                        "ImprovedDilatedAttention",
                        ImprovedDilatedAttention(
                            segment_lengths=adjusted_segments,
                            dilation_rates=dilation_rates,
                            dropout=dropout,
                        ),
                        False,
                    ),
                    (
                        "RingDilatedAttention",
                        RingDilatedAttention(
                            segment_lengths=adjusted_segments,
                            dilation_rates=dilation_rates,
                            dropout=dropout,
                            ring_size=1,
                        ),
                        False,
                    ),
                ]

                # Add block sparse implementations if available
                if HAS_BLOCK_SPARSE:
                    # Test with different sparsity ratios
                    for sparsity_ratio in [0.1, 0.25, 0.5]:  # 90%, 75%, 50% sparse
                        sparse_config = SparsePatternConfig(
                            pattern_type="dilated_sparse",
                            sparsity_ratio=sparsity_ratio,
                            block_size=32,
                        )
                        implementations.append(
                            (
                                f"BlockSparseRingDilated_{int(sparsity_ratio * 100)}%",
                                BlockSparseRingDilatedAttention(
                                    segment_lengths=adjusted_segments,
                                    dilation_rates=dilation_rates,
                                    sparse_config=sparse_config,
                                    dropout=dropout,
                                ),
                                False,
                            )
                        )

                # Multihead implementations
                embed_dim = num_heads * head_dim
                multihead_implementations = [
                    (
                        "MultiheadDilatedAttention",
                        MultiheadDilatedAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=adjusted_segments,
                            dilation_rates=dilation_rates,
                            dropout=dropout,
                        ),
                        True,
                    ),
                ]

                # Add ImprovedMultiheadDilatedAttention if available
                if HAS_IMPROVED_MHA:
                    multihead_implementations.append(
                        (
                            "ImprovedMultiheadDilatedAttention",
                            ImprovedMultiheadDilatedAttention(
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                segment_lengths=adjusted_segments,
                                dilation_rates=dilation_rates,
                                dropout=dropout,
                            ),
                            True,
                        )
                    )

                multihead_implementations.append(
                    (
                        "RingMultiheadDilatedAttention",
                        RingMultiheadDilatedAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            segment_lengths=adjusted_segments,
                            dilation_rates=dilation_rates,
                            dropout=dropout,
                            ring_size=1,
                        ),
                        True,
                    )
                )

                # Add block sparse multihead implementations if available
                if HAS_BLOCK_SPARSE:
                    # Test with 25% sparsity (75% sparse) for multihead
                    sparse_config = SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=0.25,
                        block_size=32,
                    )
                    multihead_implementations.append(
                        (
                            "BlockSparseRingMultihead_25%",
                            BlockSparseRingMultiheadDilatedAttention(
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                segment_lengths=adjusted_segments,
                                dilation_rates=dilation_rates,
                                sparse_config=sparse_config,
                                dropout=dropout,
                            ),
                            True,
                        )
                    )

                # Run benchmarks
                all_implementations = implementations + multihead_implementations

                for name, module, is_multihead in all_implementations:
                    print(f"  {name}...", end="", flush=True)

                    result = self.benchmark_implementation(
                        name=name,
                        attention_module=module,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        is_multihead=is_multihead,
                    )

                    self.results.append(result)

                    # Store results by implementation for aggregation
                    impl_name = result["name"]
                    if impl_name not in self.all_results_by_impl:
                        self.all_results_by_impl[impl_name] = []
                    self.all_results_by_impl[impl_name].append(result)

                    if result["success"]:
                        print(f" ✓ {result['mean_time_ms']:.2f}ms")
                    else:
                        print(f" ✗ {result['error']}")

                    # Cleanup
                    del module
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

    def print_summary(self):
        """Print summary of benchmark results."""

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Group results by configuration
        configs = {}
        for result in self.results:
            if not result["success"]:
                continue

            key = (result["batch_size"], result["seq_len"])
            if key not in configs:
                configs[key] = []
            configs[key].append(result)

        # Print results for each configuration
        for (batch_size, seq_len), results in sorted(configs.items()):
            print(f"\nBatch Size: {batch_size}, Sequence Length: {seq_len}")
            print("-" * 70)

            # Prepare table data
            table_data = []
            for r in sorted(results, key=lambda x: x["mean_time_ms"]):
                table_data.append(
                    [
                        r["name"],
                        f"{r['mean_time_ms']:.2f} ± {r['std_time_ms']:.2f}",
                        f"{r['throughput']:.1f}",
                        f"{r['peak_memory_mb']:.1f}",
                    ]
                )

            headers = [
                "Implementation",
                "Time (ms)",
                "Throughput (seq/s)",
                "Memory (MB)",
            ]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Calculate speedups relative to baseline
            if len(results) > 1:
                baseline = next(
                    (r for r in results if r["name"] == "DilatedAttention"), results[0]
                )
                print("\nSpeedups relative to DilatedAttention:")
                for r in results:
                    if r["name"] != baseline["name"]:
                        speedup = baseline["mean_time_ms"] / r["mean_time_ms"]
                        print(f"  {r['name']}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all dilated attention implementations"
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "float32", "bfloat16"]
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Run benchmarks
    runner = BenchmarkRunner(device=args.device, dtype=dtype)
    runner.run_benchmarks(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )
    runner.print_summary()

    # Save results using unified benchmark output management
    output_manager = BenchmarkOutputManager(
        benchmark_type="all-implementations",
        parameters={
            "device": args.device,
            "dtype": args.dtype,
            "batch_sizes": args.batch_sizes,
            "seq_lens": args.seq_lens,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
    )

    # Add results organized by implementation
    for impl_name, impl_results in runner.all_results_by_impl.items():
        output_manager.add_result(impl_name, impl_results)

    # Calculate and add summary statistics
    summary_stats = {}
    for impl_name, impl_results in runner.all_results_by_impl.items():
        successful_results = [r for r in impl_results if r.get("success", False)]
        if successful_results:
            mean_times = [r["mean_time_ms"] for r in successful_results]
            summary_stats[impl_name] = {
                "avg_time_ms": np.mean(mean_times),
                "min_time_ms": np.min(mean_times),
                "max_time_ms": np.max(mean_times),
                "num_configs_tested": len(successful_results),
            }

    output_manager.set_summary(summary_stats)

    # Save all outputs
    output_paths = output_manager.save_results()
    print("\nResults saved to:")
    for path_type, path in output_paths.items():
        print(f"  {path_type}: {path}")


if __name__ == "__main__":
    main()
