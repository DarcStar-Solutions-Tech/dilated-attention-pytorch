#!/usr/bin/env python3
"""
Final verification of the 4K sparse fix.
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


def benchmark_config(model, x, warmup=3, runs=10):
    """Benchmark a model with given input."""
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

    return sum(times) / len(times)


def main():
    print("=== Final Verification of 4K Sparse Fix (FP32) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Detect if Pascal
    compute_capability = torch.cuda.get_device_capability()[0]
    if compute_capability < 7:
        print("⚠️  Pascal GPU detected - using FP32 for accurate benchmarks")
    print()

    # Test configurations
    configs = [
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (8192, 2, "8K d=2"),
        (8192, 4, "8K d=4"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Store results
    results = []

    print(
        f"{'Config':<10} | {'Unified (ms)':<12} | {'Enhanced (ms)':<13} | {'Ratio':<10} | {'Status':<30}"
    )
    print("-" * 90)

    for seq_len, dilation_rate, desc in configs:
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

        try:
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
                    enable_4k_optimization=True,  # Enable the fix
                )
                .cuda()
                .eval()
            )

            # Create input - ALWAYS FP32
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            # Get configuration used by Enhanced
            config = enhanced._get_optimal_config(seq_len)
            effective_len = seq_len // dilation_rate

            # Check if Hilbert will be used
            M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
            if M_padded == 4096 and enhanced.enable_4k_optimization:
                will_use_hilbert = False
            else:
                will_use_hilbert = M_padded > enhanced.hilbert_threshold

            # Benchmark
            unified_time = benchmark_config(unified, x)
            enhanced_time = benchmark_config(enhanced, x)

            # Calculate actual ratio
            actual_ratio = enhanced_time / unified_time

            if actual_ratio < 1.0:
                status = f"✓ Enhanced {(1 / actual_ratio - 1) * 100:.0f}% FASTER!"
            elif actual_ratio < 1.2:
                status = "✓ Acceptable (within 20%)"
            elif actual_ratio < 1.5:
                status = "⚠️  Slower but manageable"
            else:
                status = "✗ Significant regression"

            print(
                f"{desc:<10} | {unified_time:<12.2f} | {enhanced_time:<13.2f} | {actual_ratio:<10.2f}x | {status:<30}"
            )

            # Show configuration details
            print(
                f"           → Eff: {effective_len}, block={config['block_m']}x{config['block_n']}, "
                f"fused={config['use_fused_softmax']}, hilbert={will_use_hilbert}"
            )

            results.append((desc, unified_time, enhanced_time, actual_ratio))

        except Exception as e:
            print(f"{desc:<10} | Error: {str(e)}")
            continue

    # Summary
    print("\n=== SUMMARY ===")

    # Show before/after for 4K
    print("\n4K Improvements:")
    expected_old = {
        "4K d=2": 7.54,  # From FP32 results
        "4K d=4": 2.14,  # From FP32 results
    }

    for desc, unified_time, enhanced_time, actual_ratio in results:
        if desc.startswith("4K"):
            old_ratio = expected_old.get(desc, "?")
            print(f"{desc}: Was {old_ratio}x slower → Now {actual_ratio:.2f}x")
            if actual_ratio < 1.0:
                print(
                    f"      ✓ Enhanced is {(1 / actual_ratio - 1) * 100:.0f}% FASTER than Unified!"
                )
            elif isinstance(old_ratio, float) and actual_ratio < old_ratio:
                improvement = ((old_ratio - actual_ratio) / old_ratio) * 100
                print(f"      ✓ {improvement:.0f}% improvement from previous")

    # Overall assessment
    regressions = sum(1 for _, _, _, ratio in results if ratio > 1.5)
    improvements = sum(1 for _, _, _, ratio in results if ratio < 1.0)

    print(
        f"\nOverall: {improvements} configs faster than Unified, {regressions} significant regressions"
    )

    if regressions == 0:
        print("✓ Fix successfully addresses all regressions!")
    elif regressions <= 1:
        print("✓ Fix addresses most issues, minor regressions remain")
    else:
        print("⚠️  Fix needs more work")

    # Verify correctness for 4K d=4
    print("\n=== Correctness Check (4K d=4) ===")

    torch.manual_seed(42)
    x_test = torch.randn(1, 4096, hidden_dim, device="cuda", dtype=torch.float32)

    unified_test = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=4,
        )
        .cuda()
        .eval()
    )

    enhanced_test = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=4,
            enable_4k_optimization=True,
        )
        .cuda()
        .eval()
    )

    with torch.no_grad():
        out_unified = unified_test(x_test)
        out_enhanced = enhanced_test(x_test)

    diff = (out_unified - out_enhanced).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Outputs match within tolerance")
    else:
        print("⚠️  Outputs differ - may need to investigate")


if __name__ == "__main__":
    main()
