#!/usr/bin/env python3
"""
Benchmark memory pool optimizations for BlockSparseRingDilatedAttention.

This script tests the block-sparse implementation with and without memory pools.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


def benchmark_block_sparse(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: list,
    dilation_rates: list,
    sparse_config: SparsePatternConfig,
    num_iterations: int,
    device: torch.device,
    enable_memory_pool: bool = False,
) -> dict:
    """Benchmark block-sparse attention with optional memory pool."""
    print(f"\n{'With' if enable_memory_pool else 'Without'} Memory Pool:")

    # Create test data
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Create attention module
    attention = BlockSparseRingDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        enable_memory_pool=enable_memory_pool,
        lightweight_pool=True,  # Use lightweight pool for better performance
        device=device,
        dtype=torch.float32,
    )

    # Warmup
    for _ in range(3):
        _ = attention(query, key, value)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Benchmark forward pass
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        _ = attention(query, key, value)
        if device.type == "cuda":
            torch.cuda.synchronize()

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / num_iterations

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    # Test with attention weights return
    with_weights_time = 0
    if seq_len <= 4096:  # Only test weights for smaller sequences
        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        _, weights = attention(query, key, value, return_attention_weights=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        with_weights_time = time.perf_counter() - start_time

        # Verify sparse format
        if weights:
            num_blocks = len(weights["block_indices"])
            print(
                f"  Active blocks: {num_blocks} out of {weights['num_blocks'][0] * weights['num_blocks'][1]}"
            )

    # Cleanup
    attention.cleanup_buffers()
    del attention
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "time_per_iter": avg_time,
        "peak_memory_mb": peak_memory,
        "with_weights_time": with_weights_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark BlockSparseRingDilatedAttention memory pool optimizations"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=8192, help="Sequence length")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="dilated_sparse",
        choices=["local_window", "dilated_sparse", "global_local"],
        help="Sparse attention pattern type",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.1, help="Sparsity ratio (0.1 = 90% sparse)"
    )
    parser.add_argument(
        "--block-size", type=int, default=128, help="Block size for sparsity"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Block-Sparse Ring Dilated Attention Memory Pool Benchmark")
    print("=" * 60)
    print(f"Device: {device}")
    print(
        f"Configuration: batch={args.batch_size}, seq_len={args.seq_len}, "
        f"heads={args.num_heads}, head_dim={args.head_dim}"
    )
    print(
        f"Sparse config: pattern={args.pattern}, sparsity={args.sparsity}, "
        f"block_size={args.block_size}"
    )

    # Test configuration
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Create sparse configuration
    sparse_config = SparsePatternConfig(
        pattern_type=args.pattern,
        sparsity_ratio=args.sparsity,
        block_size=args.block_size,
        local_window_size=512,
        global_tokens=64,
    )

    # Benchmark without memory pool
    results_no_pool = benchmark_block_sparse(
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        segment_lengths,
        dilation_rates,
        sparse_config,
        args.iterations,
        device,
        enable_memory_pool=False,
    )

    # Benchmark with memory pool
    results_with_pool = benchmark_block_sparse(
        args.batch_size,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        segment_lengths,
        dilation_rates,
        sparse_config,
        args.iterations,
        device,
        enable_memory_pool=True,
    )

    # Calculate improvements
    time_diff = (
        (results_no_pool["time_per_iter"] - results_with_pool["time_per_iter"])
        / results_no_pool["time_per_iter"]
    ) * 100

    memory_diff = (
        (
            (results_no_pool["peak_memory_mb"] - results_with_pool["peak_memory_mb"])
            / results_no_pool["peak_memory_mb"]
        )
        * 100
        if results_no_pool["peak_memory_mb"] > 0
        else 0
    )

    # Print summary
    print("\nResults Summary:")
    print(
        f"Without pool: {results_no_pool['time_per_iter']:.4f}s/iter, {results_no_pool['peak_memory_mb']:.1f}MB"
    )
    print(
        f"With pool:    {results_with_pool['time_per_iter']:.4f}s/iter, {results_with_pool['peak_memory_mb']:.1f}MB"
    )
    print(f"Improvement:  {time_diff:+.1f}% time, {memory_diff:+.1f}% memory")

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"block-sparse-memory-pool-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Block-Sparse Ring Dilated Attention Memory Pool Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Batch Size: {args.batch_size}\n")
        f.write(f"- Sequence Length: {args.seq_len}\n")
        f.write(f"- Num Heads: {args.num_heads}\n")
        f.write(f"- Head Dim: {args.head_dim}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- Segment Lengths: {segment_lengths}\n")
        f.write(f"- Dilation Rates: {dilation_rates}\n")
        f.write(f"- Sparse Pattern: {args.pattern}\n")
        f.write(
            f"- Sparsity Ratio: {args.sparsity} ({int((1 - args.sparsity) * 100)}% sparse)\n"
        )
        f.write(f"- Block Size: {args.block_size}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        f.write("## Results\n\n")
        f.write("| Configuration | Time/Iter | Peak Memory | With Weights Time |\n")
        f.write("|---------------|-----------|-------------|-------------------|\n")
        f.write(
            f"| Without Pool | {results_no_pool['time_per_iter']:.4f}s | "
            f"{results_no_pool['peak_memory_mb']:.1f}MB | "
            f"{results_no_pool['with_weights_time']:.4f}s |\n"
        )
        f.write(
            f"| With Pool | {results_with_pool['time_per_iter']:.4f}s | "
            f"{results_with_pool['peak_memory_mb']:.1f}MB | "
            f"{results_with_pool['with_weights_time']:.4f}s |\n"
        )
        f.write(
            f"| **Improvement** | **{time_diff:+.1f}%** | "
            f"**{memory_diff:+.1f}%** | - |\n\n"
        )

        f.write("## Memory Pool Integration Details\n\n")
        f.write("### Optimizations Applied:\n")
        f.write(
            "1. **Inherited Memory Pool**: Leverages parent RingDilatedAttentionV2 memory pool\n"
        )
        f.write("2. **1MB Threshold**: Only uses pool for tensors ≥ 1MB\n")
        f.write("3. **Causal Mask Caching**: Reuses causal masks across blocks\n")
        f.write(
            "4. **Lightweight Pool Mode**: Uses bucketed allocation without fragmentation tracking\n"
        )
        f.write(
            "5. **Sparse-Aware Allocation**: Only allocates memory for active blocks\n\n"
        )

        f.write("### Performance Analysis:\n")
        if time_diff > 5:
            f.write(f"- ✅ **Time**: {time_diff:.1f}% faster with memory pool\n")
        elif time_diff > -5:
            f.write(f"- ✅ **Time**: Negligible overhead ({time_diff:.1f}%)\n")
        else:
            f.write(f"- ⚠️ **Time**: {abs(time_diff):.1f}% slower with memory pool\n")

        if memory_diff > 5:
            f.write(f"- ✅ **Memory**: {memory_diff:.1f}% reduction with memory pool\n")
        elif abs(memory_diff) < 5:
            f.write(f"- ✅ **Memory**: Similar usage ({memory_diff:.1f}%)\n")
        else:
            f.write(
                f"- ⚠️ **Memory**: {abs(memory_diff):.1f}% increase with memory pool\n"
            )

        f.write("\n### Block-Sparse Specific Benefits:\n")
        f.write(
            "- Memory pools are particularly beneficial for sparse patterns with many active blocks\n"
        )
        f.write(
            "- Causal mask caching reduces redundant allocations in diagonal blocks\n"
        )
        f.write(
            "- Sparse computation already minimizes memory usage, so pool benefits may be limited\n"
        )
        f.write(
            "- Best suited for long sequences where communication buffers dominate memory usage\n"
        )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
