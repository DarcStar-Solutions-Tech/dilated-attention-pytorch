#!/usr/bin/env python3
"""
Test fix for 4K sparse regression based on investigation findings.

Key findings:
1. Block size 32x32 is actually SLOWER (390ms vs 13ms for 64x64)!
2. Hilbert ordering adds overhead - disabling it improves from 15.68ms to 9.82ms
3. The current Enhanced (0.80x) is actually FASTER than Unified for 4K d=2!
4. Only 4K d=4 shows real regression (2.00x slower)

Fix strategy:
1. Disable Hilbert for 4K sequences
2. Special configuration for 4K d=4 (effective length 1024)
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


class FixedEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
    """Enhanced with fixes for 4K regression."""

    def __init__(self, *args, **kwargs):
        # Override default Hilbert threshold
        kwargs["hilbert_threshold"] = 4096  # Don't use Hilbert for 4K or smaller
        super().__init__(*args, **kwargs)

    def _get_optimal_config(self, seq_len: int):
        """Override config with special handling for 4K."""
        config = super()._get_optimal_config(seq_len)

        # Special handling for 4K sparse patterns
        if seq_len == 4096 and self.dilation_rate > 1:
            effective_len = seq_len // self.dilation_rate

            if effective_len == 1024:  # 4K d=4
                # Use configuration similar to what works for 2K d=2
                config["block_m"] = 32
                config["block_n"] = 32
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 2
                config["use_fused_softmax"] = False
            # For 4K d=2 (effective 2048), keep current config as it's already faster

        return config


def benchmark_fix():
    """Benchmark the fix against original implementations."""

    print("=== Testing 4K Sparse Fix ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Parameters
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    # Test configurations
    configs = [
        (2048, 2, "2K d=2"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
        (8192, 2, "8K d=2"),
        (8192, 4, "8K d=4"),
    ]

    print(
        f"{'Config':<10} | {'Unified':<10} | {'Original':<10} | {'Fixed':<10} | {'Orig Ratio':<12} | {'Fixed Ratio':<12}"
    )
    print("-" * 80)

    for seq_len, dilation_rate, desc in configs:
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

        original = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        fixed = (
            FixedEnhanced(
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

        # Benchmark
        def benchmark(model, runs=5):
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            torch.cuda.synchronize()

            # Time
            times = []
            for _ in range(runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            return sum(times) / len(times)

        try:
            unified_time = benchmark(unified)
            original_time = benchmark(original)
            fixed_time = benchmark(fixed)

            orig_ratio = original_time / unified_time
            fixed_ratio = fixed_time / unified_time

            print(
                f"{desc:<10} | {unified_time:<10.2f} | {original_time:<10.2f} | {fixed_time:<10.2f} | "
                f"{orig_ratio:<12.2f}x | {fixed_ratio:<12.2f}x"
            )

            # Show configuration used
            fixed_config = fixed._get_optimal_config(seq_len)
            if seq_len == 4096:
                print(
                    f"           → Fixed config: block={fixed_config['block_m']}x{fixed_config['block_n']}, "
                    f"fused={fixed_config['use_fused_softmax']}, "
                    f"hilbert_threshold={fixed.hilbert_threshold}"
                )

        except Exception as e:
            print(f"{desc:<10} | Error: {str(e)}")


def verify_correctness():
    """Verify that the fix produces correct results."""

    print("\n\n=== Verifying Correctness ===")

    # Test on 4K d=4 (the most problematic config)
    seq_len = 4096
    dilation_rate = 4
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 1

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

    fixed = (
        FixedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        out_unified = unified(x)
        out_fixed = fixed(x)

    # Compare outputs
    diff = (out_unified - out_fixed).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Outputs match within tolerance")
    else:
        print("✗ Outputs differ significantly!")


def test_alternative_fixes():
    """Test alternative configuration approaches."""

    print("\n\n=== Testing Alternative Configurations for 4K d=4 ===")

    seq_len = 4096
    dilation_rate = 4
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    batch_size = 2

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    # Test different configurations
    alt_configs = [
        {
            "block_m": 32,
            "block_n": 32,
            "use_fused_softmax": False,
            "desc": "32x32 no fused",
        },
        {
            "block_m": 48,
            "block_n": 48,
            "use_fused_softmax": True,
            "desc": "48x48 with fused",
        },
        {
            "block_m": 64,
            "block_n": 32,
            "use_fused_softmax": False,
            "desc": "64x32 no fused",
        },
        {
            "block_m": 32,
            "block_n": 64,
            "use_fused_softmax": False,
            "desc": "32x64 no fused",
        },
        {
            "block_m": 64,
            "block_n": 64,
            "use_fused_softmax": False,
            "desc": "64x64 no fused",
        },
    ]

    print(f"{'Config':<20} | {'Time (ms)':<10} | {'vs Unified':<12}")
    print("-" * 45)

    # First get Unified baseline
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

    # Warmup and benchmark
    for _ in range(3):
        with torch.no_grad():
            _ = unified(x)
    torch.cuda.synchronize()

    unified_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = unified(x)
        torch.cuda.synchronize()
        unified_times.append((time.perf_counter() - start) * 1000)

    unified_time = sum(unified_times) / len(unified_times)
    print(f"{'Unified baseline':<20} | {unified_time:<10.2f} | {1.00:<12.2f}x")
    print("-" * 45)

    for cfg in alt_configs:

        class AltEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def __init__(self, *args, **kwargs):
                kwargs["hilbert_threshold"] = 8192  # No Hilbert for 4K
                super().__init__(*args, **kwargs)

            def _get_optimal_config(self, seq_len):
                return {
                    "block_m": cfg["block_m"],
                    "block_n": cfg["block_n"],
                    "block_d": min(cfg["block_m"], self.head_dim),
                    "num_warps": 4,
                    "use_fused_softmax": cfg["use_fused_softmax"],
                    "rows_per_block": 1,
                    "fused_block_n": cfg["block_n"],
                    "enable_prefetch": False,
                }

        try:
            model = (
                AltEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            torch.cuda.synchronize()

            # Benchmark
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            avg_time = sum(times) / len(times)
            ratio = avg_time / unified_time

            print(f"{cfg['desc']:<20} | {avg_time:<10.2f} | {ratio:<12.2f}x")

        except Exception as e:
            print(f"{cfg['desc']:<20} | ERROR: {str(e)}")


def main():
    benchmark_fix()
    verify_correctness()
    test_alternative_fixes()

    print("\n\n=== RECOMMENDATIONS ===")
    print("1. Set hilbert_threshold to 4096 to skip Hilbert for 4K sequences")
    print("2. For 4K d=4, use 32x32 blocks without fused softmax")
    print("3. For 4K d=2, keep current config as it's already faster than Unified")
    print("4. Consider using 64x64 without fused softmax as a compromise")


if __name__ == "__main__":
    main()
