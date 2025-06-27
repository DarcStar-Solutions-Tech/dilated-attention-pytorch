#!/usr/bin/env python3
"""
Extreme long sequence benchmark - push GPU memory to the limit.

This script tests the maximum sequence lengths possible on available GPUs.
"""

import argparse
import gc
import json
import os

# Add parent directory to path
import sys
from datetime import datetime

import torch
from torch import cuda

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch import (
    BlockSparseRingDilatedAttention,
    ImprovedDilatedAttention,
    RingDilatedAttention,
    SparsePatternConfig,
)


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage."""
    if not cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}

    return {
        "allocated": cuda.memory_allocated() / 1024**3,  # GB
        "reserved": cuda.memory_reserved() / 1024**3,
        "free": (cuda.get_device_properties(0).total_memory - cuda.memory_allocated())
        / 1024**3,
    }


def estimate_max_sequence_length(
    implementation: str,
    num_heads: int = 8,
    head_dim: int = 64,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate maximum sequence length based on available memory."""
    if not cuda.is_available():
        return 1024

    # Get available memory (leave 1GB buffer)
    total_memory = cuda.get_device_properties(0).total_memory
    available_memory = total_memory - 1024**3  # 1GB buffer

    # Estimate memory per token (rough approximation)
    bytes_per_element = 2 if dtype == torch.float16 else 4
    embed_dim = num_heads * head_dim

    # Memory formula (simplified):
    # - Inputs: 3 * batch * seq * embed * bytes (Q, K, V)
    # - Outputs: batch * seq * embed * bytes
    # - Attention workspace: varies by implementation

    if implementation == "BlockSparseRingDilatedAttention":
        # Sparse uses much less memory
        memory_per_token = 4 * embed_dim * bytes_per_element * batch_size
        workspace_factor = 0.1  # 10% of dense attention
    elif implementation == "RingDilatedAttention":
        # Ring attention has O(n) memory
        memory_per_token = 6 * embed_dim * bytes_per_element * batch_size
        workspace_factor = 0.5
    else:
        # Standard attention
        memory_per_token = 8 * embed_dim * bytes_per_element * batch_size
        workspace_factor = 1.0

    # Add workspace memory (rough estimate)
    memory_per_token *= 1 + workspace_factor

    # Calculate max tokens
    max_tokens = int(available_memory / memory_per_token)

    # Round down to nearest power of 2 for better performance
    import math

    max_tokens = 2 ** int(math.log2(max_tokens))

    return max_tokens


def test_sequence_length(  # noqa: PLR0912
    implementation: str,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
    num_runs: int = 3,
) -> dict:
    """Test a specific sequence length and return metrics."""
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    print(f"\nTesting {implementation} with sequence length {seq_len:,}...")

    # Clear cache
    gc.collect()
    if cuda.is_available():
        cuda.empty_cache()
        cuda.reset_peak_memory_stats()

    try:
        # Create module - ensure sequence length is divisible by largest segment
        if seq_len <= 8192:
            segment_lengths = [seq_len]
            dilation_rates = [1]
        else:
            # Determine largest segment that divides seq_len
            if seq_len % 131072 == 0:
                base_segment = 131072
            elif seq_len % 65536 == 0:
                base_segment = 65536
            elif seq_len % 32768 == 0:
                base_segment = 32768
            elif seq_len % 16384 == 0:
                base_segment = 16384
            elif seq_len % 8192 == 0:
                base_segment = 8192
            else:
                # Round seq_len up to nearest multiple of 8192
                seq_len = ((seq_len + 8191) // 8192) * 8192
                base_segment = 8192

            # Build segment lengths
            segment_lengths = []
            dilation_rates = []
            current = base_segment
            rate = 1

            while current <= seq_len and len(segment_lengths) < 4:
                segment_lengths.append(current)
                dilation_rates.append(rate)
                if current == seq_len:
                    break
                current = min(current * 2, seq_len)
                rate = min(rate * 2, 8)

        if implementation == "BlockSparseRingDilatedAttention":
            sparse_config = SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.05,  # 95% sparse
                block_size=128,
                local_window_size=512,
            )
            module = (
                BlockSparseRingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    sparse_config=sparse_config,
                )
                .to(device)
                .to(dtype)
            )
        elif implementation == "RingDilatedAttention":
            module = (
                RingDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    ring_size=1,
                )
                .to(device)
                .to(dtype)
            )
        else:  # ImprovedDilatedAttention
            module = (
                ImprovedDilatedAttention(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                )
                .to(device)
                .to(dtype)
            )

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Get memory after allocation
        _ = get_gpu_memory_info()  # Could be used for debugging

        # Warmup
        with torch.no_grad():
            _ = module(q, k, v)

        if cuda.is_available():
            cuda.synchronize()

        # Benchmark
        import time

        times = []

        for _ in range(num_runs):
            if cuda.is_available():
                cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = module(q, k, v)

            if cuda.is_available():
                cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        # Get peak memory
        peak_memory = (
            cuda.max_memory_allocated() / 1024**3 if cuda.is_available() else 0
        )

        # Calculate metrics
        mean_time = sum(times) / len(times)
        throughput = (seq_len * batch_size) / (mean_time / 1000) / 1e6  # M tokens/s
        memory_per_token = peak_memory / (seq_len * batch_size) * 1024  # MB/token

        result = {
            "success": True,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "mean_time_ms": mean_time,
            "throughput_mtoks": throughput,
            "peak_memory_gb": peak_memory,
            "memory_per_token_mb": memory_per_token,
            "times": times,
        }

        print(
            f"  ✓ Success: {mean_time:.1f}ms, {peak_memory:.2f}GB, {throughput:.2f}M tok/s"
        )

        # Cleanup
        del module, q, k, v, output
        gc.collect()
        if cuda.is_available():
            cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"  ✗ OOM at sequence length {seq_len:,}")
            return {
                "success": False,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "error": "OOM",
            }
        print(f"  ✗ Error: {e}")
        return {
            "success": False,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "error": str(e),
        }
    else:
        return result


def find_max_sequence_length(
    implementation: str,
    start_seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Binary search to find maximum sequence length."""
    print(f"\nFinding maximum sequence length for {implementation}...")

    # Start with estimated max
    estimated_max = estimate_max_sequence_length(
        implementation, num_heads, head_dim, batch_size, dtype
    )
    print(f"Estimated maximum: {estimated_max:,} tokens")

    # Binary search
    low = start_seq_len
    high = estimated_max * 2  # Try higher than estimate
    last_success = low

    while low <= high:
        mid = (low + high) // 2
        # Round to nearest 8192 for proper divisibility
        mid = (mid // 8192) * 8192

        if mid == 0:
            break

        result = test_sequence_length(
            implementation, mid, batch_size, num_heads, head_dim, dtype, num_runs=1
        )

        if result["success"]:
            last_success = mid
            low = mid + 8192
        else:
            high = mid - 8192

    return last_success


def main():
    parser = argparse.ArgumentParser(description="Extreme long sequence benchmark")
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=[
            "BlockSparseRingDilatedAttention",
            "RingDilatedAttention",
            "ImprovedDilatedAttention",
        ],
        help="Implementations to test",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--dtype", type=str, default="float16", help="Data type (float16/float32)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=3, help="Number of benchmark runs"
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        help="Specific sequence lengths to test (otherwise auto-detect)",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print("=" * 80)
    print("EXTREME LONG SEQUENCE BENCHMARK")
    print("=" * 80)
    print(f"Device: {device}")
    if cuda.is_available():
        print(f"GPU: {cuda.get_device_name()}")
        total_memory = cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Heads: {args.num_heads}, Head Dim: {args.head_dim}")
    print(f"Data Type: {dtype}")

    results = {}

    for impl in args.implementations:
        print(f"\n{'=' * 60}")
        print(f"Testing {impl}")
        print(f"{'=' * 60}")

        if args.sequence_lengths:
            # Test specific lengths
            impl_results = []
            for seq_len in args.sequence_lengths:
                result = test_sequence_length(
                    impl,
                    seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.num_runs,
                )
                impl_results.append(result)
        else:
            # Auto-detect maximum
            max_len = find_max_sequence_length(
                impl, 32768, args.batch_size, args.num_heads, args.head_dim, dtype
            )

            print(f"\nMaximum sequence length: {max_len:,} tokens")

            # Test a range of lengths up to max
            test_lengths = []
            current = 32768
            while current <= max_len:
                test_lengths.append(current)
                current *= 2

            if test_lengths[-1] != max_len:
                test_lengths.append(max_len)

            impl_results = []
            for seq_len in test_lengths:
                result = test_sequence_length(
                    impl,
                    seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    dtype,
                    args.num_runs,
                )
                impl_results.append(result)

        results[impl] = impl_results

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_file = f"docs/benchmarks/benchmark-extreme-sequences-{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": str(device),
                    "gpu": cuda.get_device_name() if cuda.is_available() else "N/A",
                    "total_memory_gb": cuda.get_device_properties(0).total_memory
                    / 1024**3
                    if cuda.is_available()
                    else 0,
                    "batch_size": args.batch_size,
                    "num_heads": args.num_heads,
                    "head_dim": args.head_dim,
                    "dtype": str(dtype),
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for impl, impl_results in results.items():
        print(f"\n{impl}:")
        successful = [r for r in impl_results if r.get("success", False)]
        if successful:
            max_result = max(successful, key=lambda x: x["seq_len"])
            print(f"  Maximum sequence length: {max_result['seq_len']:,} tokens")
            print(f"  Peak memory: {max_result['peak_memory_gb']:.2f} GB")
            print(f"  Throughput: {max_result['throughput_mtoks']:.2f} M tokens/s")
            print(f"  Memory per token: {max_result['memory_per_token_mb']:.3f} MB")
        else:
            print("  No successful runs")


if __name__ == "__main__":
    main()
