#!/usr/bin/env python3
"""
Detailed investigation of the 4K sparse regression.

The 4K configurations show severe regression:
- 4K d=2: 7.54x slower (was 1.31x)
- 4K d=4: 2.14x slower (was 1.31x)

This script investigates:
1. Configuration differences between sequence lengths
2. Memory access patterns
3. Block size impact
4. Kernel behavior at 4K
"""

import torch
import time
import sys
import gc

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def profile_memory_access(model, x, runs=3):
    """Profile memory access patterns."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    peak_mem = torch.cuda.max_memory_allocated()
    mem_used = (peak_mem - start_mem) / (1024 * 1024)  # MB

    return sum(times) / len(times), mem_used


def analyze_configuration_impact():
    """Analyze how configuration affects performance at different sizes."""

    print("=== Configuration Impact Analysis ===\n")

    # Test parameters
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    _ = 2

    # Test configurations
    test_configs = [
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2"),  # Problematic
        (4096, 4, "4K d=4"),  # Problematic
        (8192, 2, "8K d=2"),  # Good
        (8192, 4, "8K d=4"),  # Good
    ]

    print("Configuration Details:")
    print("-" * 80)

    for seq_len, dilation_rate, desc in test_configs:
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda()

        config = enhanced._get_optimal_config(seq_len)
        effective_len = seq_len // dilation_rate

        # Check kernel path
        M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
        use_hilbert = M_padded > enhanced.hilbert_threshold
        use_pytorch = M_padded <= 512 or not enhanced._triton_available

        print(f"\n{desc}:")
        print(f"  Sequence length: {seq_len}")
        print(f"  Effective length: {effective_len}")
        print(f"  Padded length: {M_padded}")
        print(f"  Block size: {config['block_m']}x{config['block_n']}")
        print(f"  Block_d: {config['block_d']}")
        print(f"  Num warps: {config['num_warps']}")
        print(f"  Fused softmax: {config['use_fused_softmax']}")
        print(f"  Multi-row: {config['rows_per_block']}")
        print(f"  Will use Hilbert: {use_hilbert}")
        print(f"  Will use PyTorch: {use_pytorch}")
        print(f"  Kernel path: {'PyTorch' if use_pytorch else 'Triton'}")


def test_different_block_sizes():
    """Test how different block sizes affect 4K performance."""

    print("\n\n=== Block Size Impact on 4K d=2 ===\n")

    seq_len = 4096
    dilation_rate = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Test different block configurations
    block_configs = [
        (32, 32, 2),  # Small blocks (like Unified)
        (64, 64, 4),  # Current Enhanced config
        (128, 128, 8),  # Large blocks
        (64, 32, 4),  # Asymmetric
        (32, 64, 4),  # Asymmetric reverse
    ]

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    print(
        f"{'Block M':<8} {'Block N':<8} {'Warps':<6} {'Time (ms)':<12} {'Memory (MB)':<12}"
    )
    print("-" * 50)

    for block_m, block_n, num_warps in block_configs:
        # Create a custom model with specific config
        class CustomEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def _get_optimal_config(self, seq_len):
                return {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_d": min(block_m, self.head_dim),
                    "num_warps": num_warps,
                    "use_fused_softmax": True,
                    "rows_per_block": 1,
                    "fused_block_n": block_n,
                    "enable_prefetch": False,
                }

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

        try:
            avg_time, mem_used = profile_memory_access(model, x)
            print(
                f"{block_m:<8} {block_n:<8} {num_warps:<6} {avg_time:<12.2f} {mem_used:<12.2f}"
            )
        except Exception as e:
            print(f"{block_m:<8} {block_n:<8} {num_warps:<6} ERROR: {str(e)}")


def analyze_segment_processing():
    """Analyze how segments are processed for 4K sequences."""

    print("\n\n=== Segment Processing Analysis ===\n")

    seq_len = 4096
    segment_size = 128

    print(f"Sequence length: {seq_len}")
    print(f"Segment size: {segment_size}")
    print(f"Number of segments: {seq_len // segment_size}")
    print()

    # Analyze sparse patterns
    for dilation_rate in [2, 4]:
        print(f"\nDilation rate: {dilation_rate}")
        print(f"Effective sequence length: {seq_len // dilation_rate}")

        # Per-segment analysis
        num_segments = seq_len // segment_size
        total_active = 0

        print("\nPer-segment breakdown:")
        for seg_idx in range(min(5, num_segments)):  # Show first 5 segments
            seg_start = seg_idx * segment_size
            seg_end = min(seg_start + segment_size, seq_len)
            seg_len = seg_end - seg_start

            # Active positions in this segment
            active_positions = (seg_len + dilation_rate - 1) // dilation_rate
            total_active += active_positions

            print(
                f"  Segment {seg_idx}: positions {seg_start}-{seg_end}, "
                f"active: {active_positions}, "
                f"sparsity: {(1 - active_positions / seg_len) * 100:.1f}%"
            )

        print(
            f"\nTotal active positions: {total_active} out of {seq_len} "
            f"({total_active / seq_len * 100:.1f}% density)"
        )


def benchmark_specific_configs():
    """Benchmark specific problematic configurations in detail."""

    print("\n\n=== Detailed Benchmark of Problematic Configs ===\n")

    # Parameters
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Test the problematic 4K configurations
    configs = [
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
    ]

    for seq_len, dilation_rate, desc in configs:
        print(f"\n{desc}:")
        print("-" * 40)

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

        # Clear cache and memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Benchmark both
        print("\nUnified:")
        unified_time, unified_mem = profile_memory_access(unified, x, runs=5)
        print(f"  Time: {unified_time:.2f}ms")
        print(f"  Memory: {unified_mem:.2f}MB")

        print("\nEnhanced:")
        enhanced_time, enhanced_mem = profile_memory_access(enhanced, x, runs=5)
        print(f"  Time: {enhanced_time:.2f}ms")
        print(f"  Memory: {enhanced_mem:.2f}MB")

        print(f"\nRatio: {enhanced_time / unified_time:.2f}x slower")
        print(f"Memory ratio: {enhanced_mem / unified_mem:.2f}x")

        # Test with different dtypes
        print("\nTesting with float16:")
        x_fp16 = x.half()

        try:
            with torch.amp.autocast("cuda"):
                unified_fp16_time, _ = profile_memory_access(unified, x_fp16, runs=2)
                enhanced_fp16_time, _ = profile_memory_access(enhanced, x_fp16, runs=2)

            print(f"  Unified FP16: {unified_fp16_time:.2f}ms")
            print(f"  Enhanced FP16: {enhanced_fp16_time:.2f}ms")
            print(f"  FP16 Ratio: {enhanced_fp16_time / unified_fp16_time:.2f}x")
        except Exception as e:
            print(f"  FP16 Error: {str(e)}")


def test_threshold_impact():
    """Test impact of Hilbert threshold on 4K performance."""

    print("\n\n=== Hilbert Threshold Impact ===\n")

    seq_len = 4096
    dilation_rate = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Test different thresholds
    thresholds = [512, 1024, 2048, 4096, 8192]

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    print(f"{'Threshold':<10} {'Uses Hilbert':<13} {'Time (ms)':<12}")
    print("-" * 35)

    for threshold in thresholds:
        model = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                hilbert_threshold=threshold,
            )
            .cuda()
            .eval()
        )

        # Check if Hilbert will be used
        M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
        uses_hilbert = M_padded > threshold

        avg_time, _ = profile_memory_access(model, x, runs=3)
        print(f"{threshold:<10} {str(uses_hilbert):<13} {avg_time:<12.2f}")


def main():
    print("=== Investigating 4K Sparse Regression ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Run all analyses
    analyze_configuration_impact()
    test_different_block_sizes()
    analyze_segment_processing()
    benchmark_specific_configs()
    test_threshold_impact()

    # Summary
    print("\n\n=== SUMMARY OF FINDINGS ===\n")
    print("1. Configuration Analysis:")
    print("   - 4K uses 64x64 blocks with fused softmax")
    print("   - 8K uses same config but performs well")
    print("   - Effective length after dilation: 2048 (d=2) or 1024 (d=4)")
    print()
    print("2. Possible causes of regression:")
    print("   - Block size may be too large for the effective sequence length")
    print("   - Fused softmax overhead may not be worth it at this scale")
    print("   - Memory access patterns may be suboptimal for 4K")
    print("   - Hilbert ordering may add overhead without benefit at 4K")
    print()
    print("3. Recommendations to test:")
    print("   - Use smaller blocks (32x32) for 4K sparse")
    print("   - Disable fused softmax for 4K sparse")
    print("   - Increase Hilbert threshold to skip it for 4K")
    print("   - Add special case for 4K in configuration logic")


if __name__ == "__main__":
    main()
