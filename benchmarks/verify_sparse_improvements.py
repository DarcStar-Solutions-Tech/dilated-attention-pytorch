#!/usr/bin/env python3
"""
Verify the improvements from sparse pattern optimizations.
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

# Force CUDA initialization
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.synchronize()


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
    print("=== Verifying Sparse Pattern Optimization Improvements ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test configurations matching our previous benchmarks
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
        f"{'Config':<10} | {'Unified (ms)':<12} | {'Enhanced (ms)':<13} | {'Ratio':<10} | {'Improvement vs Expected':<25}"
    )
    print("-" * 85)

    # Expected ratios from our optimization report
    expected_improvements = {
        "2K d=2": (1.93, 1.3),  # was 1.93x slower, expected to be 1.3x
        "4K d=2": (1.31, 1.1),  # was 1.31x slower, expected to be 1.1x
        "4K d=4": (1.31, 1.1),  # was 1.31x slower, expected to be 1.1x
        "8K d=2": (1.27, 1.1),  # was 1.27x slower, expected to be 1.1x
        "8K d=4": (1.32, 1.1),  # was 1.32x slower, expected to be 1.1x
    }

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
                )
                .cuda()
                .eval()
            )

            # Create input (float32 for Pascal compatibility)
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            # Get configuration used by Enhanced
            config = enhanced._get_optimal_config(seq_len)
            effective_len = seq_len // dilation_rate

            # Benchmark
            unified_time = benchmark_config(unified, x)
            enhanced_time = benchmark_config(enhanced, x)

            # Calculate actual ratio
            actual_ratio = enhanced_time / unified_time

            # Compare with expected
            old_ratio, expected_ratio = expected_improvements[desc]
            improvement_pct = ((old_ratio - actual_ratio) / old_ratio) * 100

            if actual_ratio < expected_ratio:
                improvement_str = (
                    f"✓ Better than expected! ({improvement_pct:.1f}% improvement)"
                )
            elif actual_ratio < old_ratio:
                improvement_str = f"✓ Improved ({improvement_pct:.1f}% improvement)"
            else:
                improvement_str = "✗ No improvement"

            print(
                f"{desc:<10} | {unified_time:<12.2f} | {enhanced_time:<13.2f} | {actual_ratio:<10.2f}x | {improvement_str:<25}"
            )

            # Show configuration details
            print(
                f"           → Effective length: {effective_len}, Config: block={config['block_m']}x{config['block_n']}, fused_softmax={config['use_fused_softmax']}"
            )

            results.append(
                (
                    desc,
                    unified_time,
                    enhanced_time,
                    actual_ratio,
                    old_ratio,
                    expected_ratio,
                )
            )

        except Exception as e:
            print(f"{desc:<10} | Error: {str(e)}")
            continue

    # Summary
    print("\n=== SUMMARY ===")
    print("\nActual vs Expected Performance:")

    total_improvement = 0
    count = 0

    for (
        desc,
        unified_time,
        enhanced_time,
        actual_ratio,
        old_ratio,
        expected_ratio,
    ) in results:
        improvement_pct = ((old_ratio - actual_ratio) / old_ratio) * 100
        total_improvement += improvement_pct
        count += 1

        print(
            f"{desc}: Was {old_ratio:.2f}x slower → Now {actual_ratio:.2f}x slower (Expected {expected_ratio:.2f}x)"
        )
        print(
            f"      Improvement: {improvement_pct:.1f}% (Expected {((old_ratio - expected_ratio) / old_ratio) * 100:.1f}%)"
        )

    if count > 0:
        avg_improvement = total_improvement / count
        print(f"\nAverage improvement: {avg_improvement:.1f}%")

        if avg_improvement >= 20:
            print("✓ Achieved expected 20-40% improvement range!")
        elif avg_improvement >= 10:
            print("✓ Good improvement, though below the 20-40% target")
        else:
            print("✗ Improvement below expectations")

    # Test very sparse configuration
    print("\n=== Testing Very Sparse Configurations ===")
    print("\nThese should use 32x32 blocks with simple softmax:")

    very_sparse_configs = [
        (2048, 4, "2K d=4 (effective: 512)"),
        (4096, 8, "4K d=8 (effective: 512)"),
        (1024, 4, "1K d=4 (effective: 256)"),
    ]

    for seq_len, dilation_rate, desc in very_sparse_configs:
        try:
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

            config = enhanced._get_optimal_config(seq_len)
            effective_len = seq_len // dilation_rate

            print(
                f"{desc}: block={config['block_m']}x{config['block_n']}, fused_softmax={config['use_fused_softmax']}"
            )

            if (
                effective_len <= 512
                and config["block_m"] == 32
                and not config["use_fused_softmax"]
            ):
                print("  ✓ Correctly using small blocks with simple softmax")
            else:
                print("  ✗ Not using expected configuration")

        except Exception as e:
            print(f"{desc}: Error - {str(e)}")


if __name__ == "__main__":
    main()
