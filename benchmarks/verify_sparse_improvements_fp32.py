#!/usr/bin/env python3
"""
Verify the improvements from sparse pattern optimizations using FP32.
Note: Pascal GPUs have limited FP16 support, so we use FP32 for accurate benchmarks.
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

    # Time runs (NO autocast to avoid FP16 on Pascal)
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
    print("=== Verifying Sparse Pattern Optimization Improvements (FP32) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Detect if Pascal
    compute_capability = torch.cuda.get_device_capability()[0]
    if compute_capability < 7:
        print("⚠️  Pascal GPU detected - using FP32 for accurate benchmarks")
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
        f"{'Config':<10} | {'Unified (ms)':<12} | {'Enhanced (ms)':<13} | {'Ratio':<10} | {'Status':<25}"
    )
    print("-" * 85)

    # Expected improvements from our optimization
    expected_old_ratios = {
        "2K d=2": 1.93,  # was 1.93x slower
        "4K d=2": 1.31,  # was 1.31x slower
        "4K d=4": 1.31,  # was 1.31x slower
        "8K d=2": 1.27,  # was 1.27x slower
        "8K d=4": 1.32,  # was 1.32x slower
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

            # Create input - ALWAYS FP32
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
            old_ratio = expected_old_ratios[desc]
            improvement_pct = ((old_ratio - actual_ratio) / old_ratio) * 100

            if actual_ratio < 1.0:
                status = "✓ FASTER than Unified!"
            elif actual_ratio < old_ratio * 0.8:  # 20% improvement
                status = f"✓ Good improvement ({improvement_pct:.0f}%)"
            elif actual_ratio < old_ratio:
                status = f"✓ Some improvement ({improvement_pct:.0f}%)"
            else:
                status = "✗ No improvement"

            print(
                f"{desc:<10} | {unified_time:<12.2f} | {enhanced_time:<13.2f} | {actual_ratio:<10.2f}x | {status:<25}"
            )

            # Show configuration details
            print(
                f"           → Effective: {effective_len}, block={config['block_m']}x{config['block_n']}, fused={config['use_fused_softmax']}"
            )

            results.append((desc, unified_time, enhanced_time, actual_ratio, old_ratio))

        except Exception as e:
            print(f"{desc:<10} | Error: {str(e)}")
            continue

    # Summary
    print("\n=== SUMMARY ===")

    improvements = []
    for desc, unified_time, enhanced_time, actual_ratio, old_ratio in results:
        improvement_pct = ((old_ratio - actual_ratio) / old_ratio) * 100
        improvements.append(improvement_pct)

        print(f"{desc}: Was {old_ratio:.2f}x slower → Now {actual_ratio:.2f}x")
        if actual_ratio < 1.0:
            print(
                f"      ✓ Enhanced is FASTER than Unified by {(1 / actual_ratio - 1) * 100:.0f}%!"
            )
        elif improvement_pct > 0:
            print(f"      ✓ {improvement_pct:.0f}% improvement")
        else:
            print(f"      ✗ {-improvement_pct:.0f}% regression")

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\nAverage improvement: {avg_improvement:.1f}%")

        if avg_improvement >= 20:
            print("✓ Achieved expected 20-40% improvement range!")
        elif avg_improvement >= 10:
            print("✓ Good improvement overall")
        else:
            print("⚠️  Improvement below expectations")

    # Configuration verification
    print("\n=== Configuration Verification ===")
    print("\nVery sparse configs (should use 32x32 blocks, no fused softmax):")

    very_sparse = [
        (2048, 4, 512),
        (4096, 8, 512),
        (1024, 4, 256),
    ]

    for seq_len, dilation_rate, effective in very_sparse:
        enhanced = UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda()

        config = enhanced._get_optimal_config(seq_len)

        print(f"  {seq_len}/d{dilation_rate} (eff={effective}): ", end="")
        if (
            effective <= 512
            and config["block_m"] == 32
            and not config["use_fused_softmax"]
        ):
            print("✓ Correct")
        else:
            print(
                f"✗ Got block={config['block_m']}, fused={config['use_fused_softmax']}"
            )


if __name__ == "__main__":
    main()
