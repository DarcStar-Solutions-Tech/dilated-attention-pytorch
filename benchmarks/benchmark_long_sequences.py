#!/usr/bin/env python3
"""
Benchmark for very long sequences (32K, 64K, 128K+) where Ring Attention benefits become apparent.

This script tests the scalability of different attention implementations with sequences
that are impractical for standard attention due to quadratic memory complexity.
"""

import argparse
import datetime
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch import (
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
)

# Import Ring implementations
try:
    from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
    from dilated_attention_pytorch.ring_multihead_dilated_attention import (
        RingMultiheadDilatedAttention,
    )

    RING_AVAILABLE = True
except ImportError:
    print("Warning: Ring implementations not available")
    RING_AVAILABLE = False

# Import Block Sparse implementations
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )

    BLOCK_SPARSE_AVAILABLE = True
except ImportError:
    print("Warning: Block sparse implementations not available")
    BLOCK_SPARSE_AVAILABLE = False


@dataclass
class LongSequenceBenchmarkResult:
    """Container for long sequence benchmark results."""

    implementation: str
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    ring_size: int | None
    mean_time_ms: float
    std_time_ms: float
    peak_memory_mb: float
    memory_per_token: float  # MB per token
    throughput_tokens_per_sec: float
    success: bool
    error: str | None = None

    def __str__(self):
        if not self.success:
            return f"{self.implementation} @ {self.seq_len//1024}K: FAILED - {self.error}"

        return (
            f"{self.implementation} @ {self.seq_len//1024}K: "
            f"{self.mean_time_ms:.1f}ms, "
            f"{self.peak_memory_mb:.0f}MB ({self.memory_per_token:.3f}MB/tok), "
            f"{self.throughput_tokens_per_sec/1e6:.2f}M tok/s"
        )


def get_optimal_batch_size(seq_len: int, available_memory_gb: float = 8.0) -> int:
    """Calculate optimal batch size based on sequence length and available memory."""
    # Rough estimation: each token needs ~4-8 bytes in attention
    bytes_per_token = 8
    tokens_per_batch = seq_len
    memory_per_batch_mb = (tokens_per_batch * bytes_per_token) / (1024 * 1024)

    # Leave 2GB for overhead
    available_mb = (available_memory_gb - 2) * 1024
    max_batch_size = int(available_mb / memory_per_batch_mb)

    # Practical limits
    if seq_len >= 128 * 1024:
        return 1
    elif seq_len >= 64 * 1024:
        return min(2, max_batch_size)
    elif seq_len >= 32 * 1024:
        return min(4, max_batch_size)
    else:
        return min(8, max_batch_size)


def create_long_sequence_attention(  # noqa: PLR0911, PLR0912
    impl_name: str,
    seq_len: int,
    num_heads: int = 8,
    head_dim: int = 64,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float16,
) -> torch.nn.Module | None:
    """Create attention module optimized for long sequences."""
    embed_dim = num_heads * head_dim

    # For very long sequences, use larger segment sizes
    if seq_len >= 128 * 1024:
        segment_lengths = [8192, 16384, 32768, seq_len]
        dilation_rates = [1, 2, 4, 8]
    elif seq_len >= 64 * 1024:
        segment_lengths = [4096, 8192, 16384, seq_len]
        dilation_rates = [1, 2, 4, 8]
    elif seq_len >= 32 * 1024:
        segment_lengths = [2048, 4096, 8192, seq_len]
        dilation_rates = [1, 2, 4, 8]
    else:
        segment_lengths = [2048, 4096, 8192]
        dilation_rates = [1, 2, 4]

    # Ensure segment lengths don't exceed sequence length
    segment_lengths = [min(seg, seq_len) for seg in segment_lengths]

    try:
        if impl_name == "ImprovedDilatedAttention":
            return (
                ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "ImprovedMultiheadDilatedAttention":
            return (
                ImprovedMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "RingDilatedAttention" and RING_AVAILABLE:
            # Calculate optimal ring size for long sequences
            if seq_len >= 128 * 1024:
                ring_size = 16
            elif seq_len >= 64 * 1024:
                ring_size = 8
            else:
                ring_size = 4

            # Ensure divisibility
            max_seg = max(segment_lengths)
            while seq_len % (ring_size * max_seg) != 0 and ring_size > 1:
                ring_size -= 1

            return (
                RingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                    use_checkpointing=True,  # Enable for memory efficiency
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "RingMultiheadDilatedAttention" and RING_AVAILABLE:
            if seq_len >= 128 * 1024:
                ring_size = 16
            elif seq_len >= 64 * 1024:
                ring_size = 8
            else:
                ring_size = 4

            max_seg = max(segment_lengths)
            while seq_len % (ring_size * max_seg) != 0 and ring_size > 1:
                ring_size -= 1

            return (
                RingMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "BlockSparseRingDilatedAttention" and BLOCK_SPARSE_AVAILABLE:
            # Very high sparsity for long sequences
            sparse_config = SparsePatternConfig(
                sparsity_ratio=0.95,  # 95% sparse for long sequences
                block_size=128,
                local_window_size=512,
                pattern_type="dilated_sparse",
            )

            return (
                BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparse_config=sparse_config,
                    enable_memory_pool=True,
                    enable_packed_comm=True,
                )
                .to(device)
                .to(dtype)
            )

    except Exception as e:
        print(f"Failed to create {impl_name} for seq_len={seq_len}: {e}")
        return None

    return None


def benchmark_long_sequence(
    module: torch.nn.Module,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> LongSequenceBenchmarkResult:
    """Benchmark a module with long sequences."""
    embed_dim = num_heads * head_dim
    impl_name = module.__class__.__name__

    # Get ring size if applicable
    ring_size = getattr(module, "ring_size", None)

    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    try:
        # Create inputs based on module type
        if "Multihead" in impl_name:
            q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        else:
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = module(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time runs
        times = []
        for _ in range(num_runs):
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = module(q, k, v)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)

        # Calculate metrics
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0
        )

        total_tokens = batch_size * seq_len
        memory_per_token = peak_memory / total_tokens if total_tokens > 0 else 0
        throughput = (total_tokens / mean_time) * 1000  # tokens per second

        return LongSequenceBenchmarkResult(
            implementation=impl_name,
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            ring_size=ring_size,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            peak_memory_mb=peak_memory,
            memory_per_token=memory_per_token,
            throughput_tokens_per_sec=throughput,
            success=True,
        )

    except Exception as e:
        return LongSequenceBenchmarkResult(
            implementation=impl_name,
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            ring_size=ring_size,
            mean_time_ms=0,
            std_time_ms=0,
            peak_memory_mb=0,
            memory_per_token=0,
            throughput_tokens_per_sec=0,
            success=False,
            error=str(e),
        )
    finally:
        # Clean up
        if device.type == "cuda":
            torch.cuda.empty_cache()


def plot_long_sequence_results(
    results: dict[str, list[LongSequenceBenchmarkResult]], output_dir: str
):
    """Plot results for long sequence benchmarks."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Execution time vs sequence length
    for impl_name, impl_results in results.items():
        successful = [r for r in impl_results if r.success]
        if successful:
            seq_lens = [r.seq_len / 1024 for r in successful]  # Convert to K
            times = [r.mean_time_ms for r in successful]
            ax1.plot(seq_lens, times, marker="o", label=impl_name)

    ax1.set_xlabel("Sequence Length (K tokens)")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("Execution Time vs Sequence Length")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory per token
    for impl_name, impl_results in results.items():
        successful = [r for r in impl_results if r.success]
        if successful:
            seq_lens = [r.seq_len / 1024 for r in successful]
            mem_per_tok = [r.memory_per_token for r in successful]
            ax2.plot(seq_lens, mem_per_tok, marker="o", label=impl_name)

    ax2.set_xlabel("Sequence Length (K tokens)")
    ax2.set_ylabel("Memory per Token (MB)")
    ax2.set_title("Memory Efficiency vs Sequence Length")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Throughput
    for impl_name, impl_results in results.items():
        successful = [r for r in impl_results if r.success]
        if successful:
            seq_lens = [r.seq_len / 1024 for r in successful]
            throughput = [r.throughput_tokens_per_sec / 1e6 for r in successful]  # M tokens/sec
            ax3.plot(seq_lens, throughput, marker="o", label=impl_name)

    ax3.set_xlabel("Sequence Length (K tokens)")
    ax3.set_ylabel("Throughput (M tokens/sec)")
    ax3.set_title("Throughput vs Sequence Length")
    ax3.set_xscale("log")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success rate
    impl_names = list(results.keys())
    success_rates = []
    for impl_name in impl_names:
        total = len(results[impl_name])
        successful = sum(1 for r in results[impl_name] if r.success)
        success_rates.append(successful / total * 100 if total > 0 else 0)

    ax4.bar(impl_names, success_rates)
    ax4.set_ylabel("Success Rate (%)")
    ax4.set_title("Implementation Success Rate for Long Sequences")
    ax4.set_ylim(0, 105)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    plot_path = os.path.join(output_dir, f"benchmark-long-sequences-{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark attention implementations on long sequences"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[32768, 65536, 131072],  # 32K, 64K, 128K
        help="Long sequence lengths to benchmark",
    )
    parser.add_argument(
        "--implementations", type=str, nargs="+", help="Specific implementations to benchmark"
    )
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for benchmarking",
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory for results"
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available. Long sequence benchmarks require GPU.")
        return

    # Get available memory
    total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Total Memory: {total_memory_gb:.2f} GB")
    print(f"Data Type: {args.dtype}")
    print(f"Sequence Lengths: {[f'{s//1024}K' for s in args.sequence_lengths]}")

    # Determine implementations
    all_implementations = ["ImprovedDilatedAttention", "ImprovedMultiheadDilatedAttention"]

    if RING_AVAILABLE:
        all_implementations.extend(["RingDilatedAttention", "RingMultiheadDilatedAttention"])

    if BLOCK_SPARSE_AVAILABLE:
        all_implementations.append("BlockSparseRingDilatedAttention")

    implementations = args.implementations or all_implementations

    print(f"\nImplementations to benchmark: {implementations}")

    # Run benchmarks
    results = {impl: [] for impl in implementations}

    for seq_len in args.sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Benchmarking sequence length: {seq_len//1024}K tokens")
        print(f"{'='*60}")

        # Calculate optimal batch size
        batch_size = get_optimal_batch_size(seq_len, total_memory_gb)
        print(f"Using batch size: {batch_size}")

        for impl_name in implementations:
            print(f"\n{impl_name}:")

            # Create module
            module = create_long_sequence_attention(
                impl_name, seq_len, args.num_heads, args.head_dim, device, dtype
            )

            if module is None:
                result = LongSequenceBenchmarkResult(
                    implementation=impl_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_heads=args.num_heads,
                    head_dim=args.head_dim,
                    ring_size=None,
                    mean_time_ms=0,
                    std_time_ms=0,
                    peak_memory_mb=0,
                    memory_per_token=0,
                    throughput_tokens_per_sec=0,
                    success=False,
                    error="Failed to create module",
                )
            else:
                # Run benchmark
                result = benchmark_long_sequence(
                    module,
                    seq_len,
                    batch_size,
                    args.num_heads,
                    args.head_dim,
                    device,
                    dtype,
                    num_runs=args.num_runs,
                )

                # Clean up module
                del module

            results[impl_name].append(result)
            print(f"  {result}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    results_dict = {
        "metadata": {
            "timestamp": timestamp,
            "device": torch.cuda.get_device_name(),
            "total_memory_gb": total_memory_gb,
            "dtype": args.dtype,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
        "results": {
            impl: [asdict(r) for r in impl_results] for impl, impl_results in results.items()
        },
    }

    json_path = os.path.join(args.output_dir, f"benchmark-long-sequences-{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Plot results
    plot_long_sequence_results(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("LONG SEQUENCE BENCHMARK SUMMARY")
    print("=" * 60)

    for seq_len in args.sequence_lengths:
        print(f"\nSequence Length: {seq_len//1024}K tokens")
        print("-" * 40)

        for impl in implementations:
            result = next((r for r in results[impl] if r.seq_len == seq_len), None)
            if result:
                if result.success:
                    print(
                        f"{impl:35s}: {result.mean_time_ms:8.1f}ms, "
                        f"Mem/tok: {result.memory_per_token:.3f}MB, "
                        f"Throughput: {result.throughput_tokens_per_sec/1e6:.2f}M tok/s"
                    )
                else:
                    print(f"{impl:35s}: FAILED - {result.error}")


if __name__ == "__main__":
    main()
