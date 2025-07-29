#!/usr/bin/env python3
"""
Fast test of maximum sequence lengths for the remaining Hilbert implementations.
"""

import torch
import gc
import sys
import time
from typing import Optional, Tuple

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def test_sequence_length(
    impl_class: type,
    impl_name: str,
    seq_len: int,
    batch_size: int = 1,
    hidden_dim: int = 512,
    num_heads: int = 8,
    segment_size: int = 128,
    dilation_rate: int = 1,
    dtype: torch.dtype = torch.float16,
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """Test if implementation can handle given sequence length.

    Returns:
        (success, memory_gb, time_ms, error_msg)
    """
    try:
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3

        # Create model
        model = (
            impl_class(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                hilbert_threshold=1024,
            )
            .cuda()
            .eval()
        )

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)

        # Warmup
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(dtype == torch.float16)):
                _ = model(x)
        torch.cuda.synchronize()

        # Timed run
        start = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(dtype == torch.float16)):
                out = model(x)
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) * 1000

        # Check output
        assert out.shape == x.shape

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        memory_used = peak_mem - initial_mem

        # Cleanup
        del out, x, model
        gc.collect()
        torch.cuda.empty_cache()

        return True, memory_used, time_ms, None

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return False, None, None, "OOM"

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        return False, None, None, f"{type(e).__name__}: {str(e)}"


def run_sequence_limit_tests():
    """Test sequence limits for both implementations."""

    print("=== Hilbert Attention Sequence Length Limits (Fast) ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print()

    # Test specific sequence lengths
    test_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

    # Configurations to test
    configs = [
        (
            "Dense (d=1)",
            {"dilation_rate": 1, "segment_size": 128, "dtype": torch.float16},
        ),
        (
            "Sparse (d=2)",
            {"dilation_rate": 2, "segment_size": 128, "dtype": torch.float16},
        ),
        (
            "Sparse (d=4)",
            {"dilation_rate": 4, "segment_size": 128, "dtype": torch.float16},
        ),
    ]

    implementations = [
        (UnifiedHilbertAttention, "Unified"),
        (UnifiedHilbertAttentionOptimizedEnhanced, "Enhanced"),
    ]

    # Results table header
    print(
        f"{'Seq Length':<12} | {'Config':<15} | {'Implementation':<15} | {'Memory (GB)':<12} | {'Time (ms)':<12} | {'Status':<10}"
    )
    print("-" * 100)

    results = {}

    for config_name, config in configs:
        results[config_name] = {}

        for impl_class, impl_name in implementations:
            max_success = 0

            for seq_len in test_lengths:
                # Skip if we already failed at a smaller size
                if max_success > 0 and seq_len > max_success * 2:
                    continue

                success, memory_gb, time_ms, error = test_sequence_length(
                    impl_class, impl_name, seq_len, **config
                )

                if success:
                    print(
                        f"{seq_len:<12,} | {config_name:<15} | {impl_name:<15} | {memory_gb:<12.2f} | {time_ms:<12.1f} | {'✓':<10}"
                    )
                    max_success = seq_len
                else:
                    print(
                        f"{seq_len:<12,} | {config_name:<15} | {impl_name:<15} | {'-':<12} | {'-':<12} | {error:<10}"
                    )
                    # Don't test larger sizes after OOM
                    if error == "OOM":
                        break

            results[config_name][impl_name] = max_success

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Maximum Successful Sequence Lengths")
    print("=" * 80)
    print(
        f"{'Configuration':<20} | {'Unified':<20} | {'Enhanced':<20} | {'Winner':<20}"
    )
    print("-" * 80)

    for config_name, impl_results in results.items():
        unified = impl_results.get("Unified", 0)
        enhanced = impl_results.get("Enhanced", 0)

        if unified > enhanced:
            winner = f"Unified ({unified / enhanced:.1f}x)"
        elif enhanced > unified:
            winner = f"Enhanced ({enhanced / unified:.1f}x)"
        else:
            winner = "Tie"

        print(f"{config_name:<20} | {unified:>18,} | {enhanced:>18,} | {winner:<20}")

    # Performance comparison at common sequence lengths
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON (ms)")
    print("=" * 80)

    common_lengths = [2048, 4096, 8192, 16384]

    for seq_len in common_lengths:
        print(f"\nSequence Length: {seq_len:,}")
        print(f"{'Config':<15} | {'Unified':<15} | {'Enhanced':<15} | {'Speedup':<15}")
        print("-" * 65)

        for config_name, config in configs:
            # Test both implementations at this length
            unified_result = test_sequence_length(
                UnifiedHilbertAttention, "Unified", seq_len, **config
            )
            enhanced_result = test_sequence_length(
                UnifiedHilbertAttentionOptimizedEnhanced, "Enhanced", seq_len, **config
            )

            if unified_result[0] and enhanced_result[0]:
                unified_time = unified_result[2]
                enhanced_time = enhanced_result[2]

                if unified_time < enhanced_time:
                    speedup = f"Unified {enhanced_time / unified_time:.1f}x"
                else:
                    speedup = f"Enhanced {unified_time / enhanced_time:.1f}x"

                print(
                    f"{config_name:<15} | {unified_time:<15.1f} | {enhanced_time:<15.1f} | {speedup:<15}"
                )
            else:
                unified_str = (
                    f"{unified_result[2]:.1f}" if unified_result[0] else "Failed"
                )
                enhanced_str = (
                    f"{enhanced_result[2]:.1f}" if enhanced_result[0] else "Failed"
                )
                print(
                    f"{config_name:<15} | {unified_str:<15} | {enhanced_str:<15} | {'-':<15}"
                )


if __name__ == "__main__":
    run_sequence_limit_tests()
