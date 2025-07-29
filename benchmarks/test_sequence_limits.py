#!/usr/bin/env python3
"""
Test maximum sequence lengths for the remaining Hilbert implementations.
"""

import torch
import gc
import sys
import traceback
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
) -> Tuple[bool, Optional[float], Optional[str]]:
    """Test if implementation can handle given sequence length.

    Returns:
        (success, memory_gb, error_msg)
    """
    try:
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3

        print("  Creating model...", end="", flush=True)

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

        print(" ✓", flush=True)

        # Create input
        print(f"  Creating input tensor ({seq_len} tokens)...", end="", flush=True)
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=dtype)
        print(" ✓", flush=True)

        # Forward pass
        print("  Running forward pass...", end="", flush=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(dtype == torch.float16)):
                out = model(x)
        torch.cuda.synchronize()
        print(" ✓", flush=True)

        # Check output
        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        memory_used = peak_mem - initial_mem

        # Cleanup
        del out, x, model
        gc.collect()
        torch.cuda.empty_cache()

        return True, memory_used, None

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return False, None, "Out of memory"

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        error_msg = f"{type(e).__name__}: {str(e)}"
        # Print traceback for debugging
        print(f"\n  ERROR: {error_msg}")
        traceback.print_exc()
        return False, None, error_msg


def find_max_sequence_length(impl_class: type, impl_name: str, **kwargs):
    """Binary search to find maximum sequence length."""

    print(f"\n{'=' * 60}")
    print(f"Testing {impl_name}")
    print(f"{'=' * 60}")

    # Get GPU info
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {total_memory:.1f} GB")
    print()

    # Test configurations
    segment_size = kwargs.get("segment_size", 128)
    dilation_rate = kwargs.get("dilation_rate", 1)
    dtype = kwargs.get("dtype", torch.float16)

    print("Configuration:")
    print(f"  Segment size: {segment_size}")
    print(f"  Dilation rate: {dilation_rate}")
    print(f"  Dtype: {dtype}")
    print()

    # Start with powers of 2
    test_lengths = [
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
    ]

    last_success = 0
    first_failure = None

    # First, find rough bounds
    print("Phase 1: Finding rough bounds...")
    for seq_len in test_lengths:
        print(f"\nTesting {seq_len:,} tokens:")
        success, memory_gb, error = test_sequence_length(
            impl_class, impl_name, seq_len, **kwargs
        )

        if success:
            print(f"  ✓ Success! Memory used: {memory_gb:.2f} GB")
            last_success = seq_len
        else:
            print(f"  ✗ Failed: {error}")
            first_failure = seq_len
            break

    # If we didn't hit a failure, try larger
    if first_failure is None:
        print("\nTesting even larger sequences...")
        for multiplier in [1.5, 2, 3, 4, 5]:
            seq_len = int(last_success * multiplier)
            print(f"\nTesting {seq_len:,} tokens:")
            success, memory_gb, error = test_sequence_length(
                impl_class, impl_name, seq_len, **kwargs
            )

            if success:
                print(f"  ✓ Success! Memory used: {memory_gb:.2f} GB")
                last_success = seq_len
            else:
                print(f"  ✗ Failed: {error}")
                first_failure = seq_len
                break

    # Binary search for exact limit
    if first_failure is not None and first_failure > last_success:
        print(
            f"\nPhase 2: Binary search between {last_success:,} and {first_failure:,}"
        )

        low = last_success
        high = first_failure

        while high - low > segment_size:
            mid = (
                (low + high) // 2 // segment_size
            ) * segment_size  # Round to segment size

            print(f"\nTesting {mid:,} tokens:")
            success, memory_gb, error = test_sequence_length(
                impl_class, impl_name, mid, **kwargs
            )

            if success:
                print(f"  ✓ Success! Memory used: {memory_gb:.2f} GB")
                low = mid
                last_success = mid
            else:
                print(f"  ✗ Failed: {error}")
                high = mid

    print(f"\n{'=' * 60}")
    print(f"Maximum sequence length for {impl_name}: {last_success:,} tokens")
    print(f"{'=' * 60}")

    return last_success


def run_sequence_limit_tests():
    """Test sequence limits for both implementations."""

    print("=== Hilbert Attention Sequence Length Limits ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # Test configurations
    configs = [
        # Dense patterns
        {"dilation_rate": 1, "segment_size": 128, "dtype": torch.float16},
        {"dilation_rate": 1, "segment_size": 256, "dtype": torch.float16},
        # Sparse patterns
        {"dilation_rate": 2, "segment_size": 128, "dtype": torch.float16},
        {"dilation_rate": 4, "segment_size": 128, "dtype": torch.float16},
        # Float32 test (more memory intensive)
        {"dilation_rate": 1, "segment_size": 128, "dtype": torch.float32},
    ]

    results = {}

    for config in configs:
        config_name = f"d={config['dilation_rate']}, seg={config['segment_size']}, {config['dtype']}"
        print(f"\n\n{'#' * 80}")
        print(f"Configuration: {config_name}")
        print(f"{'#' * 80}")

        results[config_name] = {}

        # Test Unified
        max_len = find_max_sequence_length(
            UnifiedHilbertAttention, "UnifiedHilbertAttention", **config
        )
        results[config_name]["Unified"] = max_len

        # Test Enhanced
        max_len = find_max_sequence_length(
            UnifiedHilbertAttentionOptimizedEnhanced,
            "UnifiedHilbertAttentionOptimizedEnhanced",
            **config,
        )
        results[config_name]["Enhanced"] = max_len

    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY: Maximum Sequence Lengths")
    print("=" * 80)
    print(f"{'Configuration':<40} | {'Unified':<15} | {'Enhanced':<15}")
    print("-" * 80)

    for config_name, impl_results in results.items():
        unified = impl_results.get("Unified", 0)
        enhanced = impl_results.get("Enhanced", 0)
        print(f"{config_name:<40} | {unified:>13,} | {enhanced:>13,}")

    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)

    # Analyze results
    for config_name, impl_results in results.items():
        unified = impl_results.get("Unified", 0)
        enhanced = impl_results.get("Enhanced", 0)

        if unified > enhanced:
            ratio = unified / enhanced if enhanced > 0 else float("inf")
            print(f"{config_name}: Unified handles {ratio:.1f}x longer sequences")
        elif enhanced > unified:
            ratio = enhanced / unified if unified > 0 else float("inf")
            print(f"{config_name}: Enhanced handles {ratio:.1f}x longer sequences")
        else:
            print(f"{config_name}: Both handle same max length")


if __name__ == "__main__":
    run_sequence_limit_tests()
