#!/usr/bin/env python3
"""
Comprehensive benchmark for ALL dilated attention implementations including:
- Standard: DilatedAttention, MultiheadDilatedAttention
- Improved: ImprovedDilatedAttention, ImprovedMultiheadDilatedAttention
- Ring: RingDilatedAttention, RingMultiheadDilatedAttention
- Block Sparse: BlockSparseRingDilatedAttention, BlockSparseRingMultiheadDilatedAttention
"""

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import torch

matplotlib.use("Agg")  # Non-interactive backend

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all implementations
from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    ImprovedMultiheadDilatedAttention,
    MultiheadDilatedAttention,
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
    from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
        BlockSparseRingMultiheadDilatedAttention,
    )

    BLOCK_SPARSE_AVAILABLE = True
except ImportError:
    print("Warning: Block sparse implementations not available")
    BLOCK_SPARSE_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    implementation: str
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int
    mean_time_ms: float
    std_time_ms: float
    peak_memory_mb: float
    samples: list[float]
    error: str | None = None

    def __str__(self):
        if self.error:
            return (
                f"{self.implementation} @ seq_len={self.seq_len}: ERROR - {self.error}"
            )
        return (
            f"{self.implementation} @ seq_len={self.seq_len}: "
            f"{self.mean_time_ms:.2f}±{self.std_time_ms:.2f}ms, "
            f"peak_mem={self.peak_memory_mb:.1f}MB"
        )


def measure_memory(device: torch.device) -> float:
    """Measure current memory usage in MB."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def benchmark_module(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    num_runs: int = 10,
    warmup_runs: int = 3,
    device: torch.device = torch.device("cuda"),
) -> tuple[float, float, list[float]]:
    """Benchmark a module and return timing statistics."""
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = module(*inputs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = module(*inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5

    return mean_time, std_time, times


def create_attention_module(  # noqa: PLR0911, PLR0912
    impl_name: str,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module | None:
    """Create an attention module based on implementation name."""
    embed_dim = num_heads * head_dim

    # Calculate segment lengths and dilation rates
    if seq_len <= 4096:
        segment_lengths = [seq_len]
        dilation_rates = [1]
    else:
        # Use powers of 2 up to seq_len
        segment_lengths = []
        dilation_rates = []
        seg = 2048
        rate = 1
        while seg <= seq_len:
            segment_lengths.append(min(seg, seq_len))
            dilation_rates.append(rate)
            seg *= 2
            rate *= 2

    try:
        if impl_name == "DilatedAttention":
            return (
                DilatedAttention(
                    segment_lengths=segment_lengths, dilation_rates=dilation_rates
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "MultiheadDilatedAttention":
            return (
                MultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "ImprovedDilatedAttention":
            return (
                ImprovedDilatedAttention(
                    segment_lengths=segment_lengths, dilation_rates=dilation_rates
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
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "RingDilatedAttention" and RING_AVAILABLE:
            # Ring size should divide seq_len evenly
            ring_size = min(8, seq_len // max(segment_lengths))
            while seq_len % (ring_size * max(segment_lengths)) != 0 and ring_size > 1:
                ring_size -= 1

            return (
                RingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=ring_size,
                )
                .to(device)
                .to(dtype)
            )

        elif impl_name == "RingMultiheadDilatedAttention" and RING_AVAILABLE:
            ring_size = min(8, seq_len // max(segment_lengths))
            while seq_len % (ring_size * max(segment_lengths)) != 0 and ring_size > 1:
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
            # BlockSparseRingDilatedAttention uses sparse_config parameter
            sparse_config = SparsePatternConfig(
                sparsity_ratio=0.1,  # Keep 10% of blocks (90% sparse)
                block_size=64,
                local_window_size=256,
                pattern_type="dilated_sparse",
            )
            return (
                BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparse_config=sparse_config,
                )
                .to(device)
                .to(dtype)
            )

        elif (
            impl_name == "BlockSparseRingMultiheadDilatedAttention"
            and BLOCK_SPARSE_AVAILABLE
        ):
            # BlockSparseRingMultiheadDilatedAttention uses sparse_config parameter
            sparse_config = SparsePatternConfig(
                sparsity_ratio=0.1,  # Keep 10% of blocks (90% sparse)
                block_size=64,
                local_window_size=256,
                pattern_type="dilated_sparse",
            )
            return (
                BlockSparseRingMultiheadDilatedAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparse_config=sparse_config,
                )
                .to(device)
                .to(dtype)
            )

    except Exception as e:
        print(f"Failed to create {impl_name}: {e}")
        return None

    return None


def run_benchmark(
    implementations: list[str],
    sequence_lengths: list[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
    num_runs: int = 10,
) -> dict[str, list[BenchmarkResult]]:
    """Run benchmarks for all implementations."""
    embed_dim = num_heads * head_dim
    results = {impl: [] for impl in implementations}

    for seq_len in sequence_lengths:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking sequence length: {seq_len}")
        print(f"{'=' * 60}")

        for impl_name in implementations:
            print(f"\n{impl_name}:")

            # Reset memory tracking
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Create module
            module = create_attention_module(
                impl_name, seq_len, num_heads, head_dim, device, dtype
            )

            if module is None:
                result = BenchmarkResult(
                    implementation=impl_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mean_time_ms=0,
                    std_time_ms=0,
                    peak_memory_mb=0,
                    samples=[],
                    error="Failed to create module",
                )
                results[impl_name].append(result)
                print(f"  {result}")
                continue

            # Create inputs
            if impl_name in [
                "MultiheadDilatedAttention",
                "ImprovedMultiheadDilatedAttention",
                "RingMultiheadDilatedAttention",
                "BlockSparseRingMultiheadDilatedAttention",
            ]:
                # Multihead format: (batch, seq, embed_dim)
                q = torch.randn(
                    batch_size, seq_len, embed_dim, device=device, dtype=dtype
                )
                k = torch.randn(
                    batch_size, seq_len, embed_dim, device=device, dtype=dtype
                )
                v = torch.randn(
                    batch_size, seq_len, embed_dim, device=device, dtype=dtype
                )
            else:
                # Standard format: (batch, seq, heads, head_dim)
                q = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )
                k = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )
                v = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )

            try:
                # Run benchmark
                mean_time, std_time, times = benchmark_module(
                    module, (q, k, v), num_runs=num_runs, device=device
                )

                peak_memory = measure_memory(device)

                result = BenchmarkResult(
                    implementation=impl_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mean_time_ms=mean_time,
                    std_time_ms=std_time,
                    peak_memory_mb=peak_memory,
                    samples=times,
                )

            except Exception as e:
                result = BenchmarkResult(
                    implementation=impl_name,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mean_time_ms=0,
                    std_time_ms=0,
                    peak_memory_mb=0,
                    samples=[],
                    error=str(e),
                )

            results[impl_name].append(result)
            print(f"  {result}")

            # Clean up
            del module
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def plot_results(results: dict[str, list[BenchmarkResult]], output_dir: str):
    """Plot benchmark results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot execution time
    for impl_name, impl_results in results.items():
        seq_lens = []
        mean_times = []
        std_times = []

        for result in impl_results:
            if result.error is None:
                seq_lens.append(result.seq_len)
                mean_times.append(result.mean_time_ms)
                std_times.append(result.std_time_ms)

        if seq_lens:
            ax1.errorbar(
                seq_lens,
                mean_times,
                yerr=std_times,
                label=impl_name,
                marker="o",
                capsize=5,
            )

    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_title("Execution Time vs Sequence Length")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot memory usage
    for impl_name, impl_results in results.items():
        seq_lens = []
        memories = []

        for result in impl_results:
            if result.error is None:
                seq_lens.append(result.seq_len)
                memories.append(result.peak_memory_mb)

        if seq_lens:
            ax2.plot(seq_lens, memories, label=impl_name, marker="o")

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Memory Usage vs Sequence Length")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    plot_path = os.path.join(
        output_dir, f"benchmark-all-implementations-{timestamp}.png"
    )
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def save_results(results: dict[str, list[BenchmarkResult]], output_dir: str):
    """Save results to JSON file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-UTC")

    # Convert results to dict format
    results_dict = {}
    for impl_name, impl_results in results.items():
        results_dict[impl_name] = [asdict(r) for r in impl_results]

    # Save to file
    json_path = os.path.join(
        output_dir, f"benchmark-all-implementations-{timestamp}.json"
    )
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all dilated attention implementations"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs per configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/benchmarks",
        help="Output directory for results",
    )
    parser.add_argument(
        "--implementations",
        type=str,
        nargs="+",
        help="Specific implementations to benchmark (default: all)",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    print(f"PyTorch version: {torch.__version__}")

    # Determine implementations to test
    all_implementations = [
        "DilatedAttention",
        "MultiheadDilatedAttention",
        "ImprovedDilatedAttention",
        "ImprovedMultiheadDilatedAttention",
    ]

    if RING_AVAILABLE:
        all_implementations.extend(
            ["RingDilatedAttention", "RingMultiheadDilatedAttention"]
        )

    if BLOCK_SPARSE_AVAILABLE:
        all_implementations.extend(
            [
                "BlockSparseRingDilatedAttention",
                "BlockSparseRingMultiheadDilatedAttention",
            ]
        )

    implementations = args.implementations or all_implementations

    print(f"\nImplementations to benchmark: {implementations}")
    print(f"Sequence lengths: {args.sequence_lengths}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}, Head dim: {args.head_dim}")

    # Run benchmarks
    results = run_benchmark(
        implementations=implementations,
        sequence_lengths=args.sequence_lengths,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        device=device,
        dtype=dtype,
        num_runs=args.num_runs,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save and plot results
    save_results(results, args.output_dir)
    plot_results(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for seq_len in args.sequence_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        for impl in implementations:
            results_for_len = [r for r in results[impl] if r.seq_len == seq_len]
            if results_for_len and results_for_len[0].error is None:
                r = results_for_len[0]
                print(
                    f"{impl:40s}: {r.mean_time_ms:8.2f}ms ± {r.std_time_ms:6.2f}ms, "
                    f"Mem: {r.peak_memory_mb:8.1f}MB"
                )
            elif results_for_len:
                print(f"{impl:40s}: ERROR - {results_for_len[0].error}")


if __name__ == "__main__":
    main()
