#!/usr/bin/env python3
"""
Analyze sparse pattern performance issues in Enhanced implementation.
"""

import torch
import time
import sys
import gc
import numpy as np

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def profile_sparse_pattern(model, x, warmup=3, runs=10):
    """Profile a model on sparse patterns."""
    torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": min(times),
        "max": max(times),
    }


def analyze_memory_access_patterns():
    """Analyze memory access patterns for sparse attention."""

    print("=== Memory Access Pattern Analysis ===\n")

    # Parameters
    seq_len = 8192
    segment_size = 128
    hidden_dim = 512
    num_heads = 8
    _ = 1

    # Test different dilation rates
    dilation_rates = [1, 2, 4, 8]

    print(f"Sequence length: {seq_len}")
    print(f"Segment size: {segment_size}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Num heads: {num_heads}")
    print()

    for d in dilation_rates:
        print(f"\nDilation rate: {d}")

        # Calculate sparse pattern statistics
        effective_len = seq_len // d
        num_segments = (seq_len + segment_size - 1) // segment_size

        print(f"  Effective length: {effective_len}")
        print(f"  Sparsity: {(1 - 1 / d) * 100:.1f}%")
        print(f"  Number of segments: {num_segments}")

        # Per-segment analysis
        total_active = 0
        min_active = float("inf")
        max_active = 0

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_end = min(seg_start + segment_size, seq_len)
            seg_len = seg_end - seg_start

            # Active positions in this segment
            active_positions = (seg_len + d - 1) // d
            total_active += active_positions
            min_active = min(min_active, active_positions)
            max_active = max(max_active, active_positions)

        avg_active = total_active / num_segments

        print(
            f"  Active positions per segment: min={min_active}, avg={avg_active:.1f}, max={max_active}"
        )
        print(f"  Total active positions: {total_active}")

        # Memory access pattern
        if d > 1:
            stride_bytes = d * hidden_dim * 4  # float32
            print(f"  Memory stride: {stride_bytes / 1024:.1f} KB")
            print(f"  Cache line efficiency: {100 / d:.1f}%")


def benchmark_triton_vs_pytorch_sparse():
    """Compare Triton vs PyTorch implementations for sparse patterns."""

    print("\n\n=== Triton vs PyTorch Sparse Implementation ===\n")

    # Parameters
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Test configurations
    configs = [
        (2048, 2),
        (4096, 2),
        (4096, 4),
        (8192, 2),
        (8192, 4),
        (16384, 4),
    ]

    print(
        f"{'Config':<12} | {'Unified':<12} | {'Enhanced':<12} | {'Ratio':<8} | {'Analysis':<40}"
    )
    print("-" * 85)

    for seq_len, dilation_rate in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Create models
        unified = (
            UnifiedHilbertAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        # Create input
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )

        # Profile both
        unified_stats = profile_sparse_pattern(unified, x, runs=5)
        enhanced_stats = profile_sparse_pattern(enhanced, x, runs=5)

        ratio = enhanced_stats["mean"] / unified_stats["mean"]

        # Analysis
        if ratio < 1.0:
            analysis = f"Enhanced {(1 / ratio - 1) * 100:.0f}% faster"
        else:
            analysis = f"Enhanced {(ratio - 1) * 100:.0f}% slower"

        # Check which path Enhanced uses
        M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
        uses_triton = M_padded > 512 and enhanced._triton_available

        if uses_triton:
            analysis += " (Triton)"
        else:
            analysis += " (PyTorch)"

        print(
            f"{seq_len}d{dilation_rate:<11} | {unified_stats['mean']:<12.2f} | "
            f"{enhanced_stats['mean']:<12.2f} | {ratio:<8.2f}x | {analysis:<40}"
        )


def test_block_size_impact_on_sparse():
    """Test impact of different block sizes on sparse patterns."""

    print("\n\n=== Block Size Impact on Sparse Patterns ===\n")

    # Focus on problematic configuration: 8K d=2
    seq_len = 8192
    dilation_rate = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Test different block configurations
    block_configs = [
        (32, 32, 2, "Small (like 4K d=4)"),
        (64, 64, 4, "Current default"),
        (128, 128, 8, "Large blocks"),
        (64, 32, 4, "Asymmetric 64x32"),
        (32, 64, 4, "Asymmetric 32x64"),
        (48, 48, 4, "Medium 48x48"),
    ]

    print(f"Testing 8K d=2 (effective length: {seq_len // dilation_rate})")
    print(f"{'Block Config':<20} | {'Time (ms)':<10} | {'vs Default':<12}")
    print("-" * 45)

    default_time = None

    for block_m, block_n, num_warps, desc in block_configs:

        class CustomEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def _get_optimal_config(self, seq_len):
                config = super()._get_optimal_config(seq_len)
                # Override for sparse patterns
                if self.dilation_rate > 1:
                    config["block_m"] = block_m
                    config["block_n"] = block_n
                    config["block_d"] = min(block_m, self.head_dim)
                    config["num_warps"] = num_warps
                    config["use_fused_softmax"] = False  # Try without fused
                return config

        try:
            model = (
                CustomEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            stats = profile_sparse_pattern(model, x, runs=5)

            if default_time is None:
                default_time = stats["mean"]
                ratio_str = "1.00x"
            else:
                ratio_str = f"{stats['mean'] / default_time:.2f}x"

            print(f"{desc:<20} | {stats['mean']:<10.2f} | {ratio_str:<12}")

        except Exception as e:
            print(f"{desc:<20} | ERROR: {str(e)}")


def analyze_sparse_kernel_inefficiencies():
    """Analyze specific inefficiencies in the sparse kernel."""

    print("\n\n=== Sparse Kernel Inefficiency Analysis ===\n")

    print("1. Memory Access Pattern Issues:")
    print("   - Strided access with dilation > 1 causes poor coalescing")
    print("   - Each thread loads non-contiguous memory locations")
    print("   - GPU cache utilization drops to 1/dilation_rate")

    print("\n2. Block Size Mismatch:")
    print("   - 64x64 blocks are too large for sparse patterns")
    print("   - Many threads remain idle when processing sparse blocks")
    print("   - Warp divergence when masking sparse positions")

    print("\n3. Online Softmax Overhead:")
    print("   - Online softmax has overhead for small active sets")
    print("   - For d=4, only 25% of positions are active")
    print("   - Statistics tracking overhead not amortized")

    print("\n4. Hilbert Ordering Impact:")
    print("   - Hilbert helps dense patterns but may hurt sparse")
    print("   - Sparse access already non-sequential")
    print("   - Additional indirection through Hilbert map")


def suggest_optimizations():
    """Suggest optimizations for sparse patterns."""

    print("\n\n=== Suggested Optimizations ===\n")

    print("1. Adaptive Block Sizing:")
    print("   - Use smaller blocks (32x32) for high dilation rates")
    print("   - Scale block size with effective sequence length")
    print("   - Consider separate kernels for different sparsity levels")

    print("\n2. Sparse-Specific Kernel:")
    print("   - Pack sparse positions for better coalescing")
    print("   - Process multiple sparse positions per thread")
    print("   - Use different memory layout for sparse K/V")

    print("\n3. Disable Overhead Features:")
    print("   - Skip online softmax for very sparse patterns")
    print("   - Disable Hilbert for sparse (already non-sequential)")
    print("   - Remove prefetching hints for sparse")

    print("\n4. Hybrid Approach:")
    print("   - Use PyTorch for small sparse patterns")
    print("   - Switch to optimized kernel only for large dense")
    print("   - Adaptive threshold based on sparsity")


def main():
    print("=== Sparse Pattern Performance Analysis ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Run all analyses
    analyze_memory_access_patterns()
    benchmark_triton_vs_pytorch_sparse()
    test_block_size_impact_on_sparse()
    analyze_sparse_kernel_inefficiencies()
    suggest_optimizations()

    print("\n\n=== Key Findings ===")
    print("1. Sparse patterns suffer from poor memory coalescing")
    print("2. Block sizes are too large for sparse active sets")
    print("3. Online softmax adds overhead for small computations")
    print("4. Need adaptive configuration based on sparsity level")


if __name__ == "__main__":
    main()
