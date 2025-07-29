#!/usr/bin/env python3
"""
Test the corrected 4K fix that preserves Hilbert SFC.
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
    print("=== Testing Corrected 4K Fix (Preserves Hilbert SFC) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test configurations focusing on 4K
    configs = [
        (4096, 1, "4K d=1 (dense)"),
        (4096, 2, "4K d=2"),
        (4096, 4, "4K d=4"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    print(
        f"{'Config':<15} | {'Unified (ms)':<12} | {'Enhanced (ms)':<13} | {'Ratio':<10} | {'Uses Hilbert':<13} | {'Block Size':<12}"
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
                    enable_4k_optimization=True,
                    hilbert_threshold=1024,  # Standard threshold
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
            will_use_hilbert = M_padded > enhanced.hilbert_threshold

            # Benchmark
            unified_time = benchmark_config(unified, x)
            enhanced_time = benchmark_config(enhanced, x)

            # Calculate actual ratio
            actual_ratio = enhanced_time / unified_time

            block_str = f"{config['block_m']}x{config['block_n']}"

            print(
                f"{desc:<15} | {unified_time:<12.2f} | {enhanced_time:<13.2f} | "
                f"{actual_ratio:<10.2f}x | {str(will_use_hilbert):<13} | {block_str:<12}"
            )

            # Show detailed config for sparse patterns
            if dilation_rate > 1:
                print(
                    f"                → Effective length: {effective_len}, "
                    f"fused_softmax: {config['use_fused_softmax']}"
                )

        except Exception as e:
            print(f"{desc:<15} | Error: {str(e)}")
            continue

    # Test with different Hilbert thresholds
    print("\n=== Impact of Hilbert Threshold on 4K d=4 ===")

    seq_len = 4096
    dilation_rate = 4
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    thresholds = [512, 1024, 2048, 4096, 8192]

    print(
        f"{'Threshold':<10} | {'Time (ms)':<10} | {'Uses Hilbert':<13} | {'vs Base':<10}"
    )
    print("-" * 50)

    base_time = None

    for threshold in thresholds:
        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                enable_4k_optimization=True,
                hilbert_threshold=threshold,
            )
            .cuda()
            .eval()
        )

        M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
        uses_hilbert = M_padded > threshold

        time_ms = benchmark_config(enhanced, x, runs=5)

        if base_time is None:
            base_time = time_ms
            ratio_str = "1.00x"
        else:
            ratio_str = f"{time_ms / base_time:.2f}x"

        print(
            f"{threshold:<10} | {time_ms:<10.2f} | {str(uses_hilbert):<13} | {ratio_str:<10}"
        )

    print("\n=== Summary ===")
    print("1. The fix now only changes block configuration for 4K d=4")
    print("2. Hilbert SFC is used based on the threshold (default 1024)")
    print("3. 4K sequences (4096 > 1024) will use Hilbert by default")
    print("4. The 32x32 block configuration provides speedup for very sparse patterns")


if __name__ == "__main__":
    main()
