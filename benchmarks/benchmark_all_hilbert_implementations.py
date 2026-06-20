#!/usr/bin/env python3
"""
Comprehensive benchmark of all Hilbert attention implementations.
"""

import torch
import time
import sys
import traceback
from typing import Dict, Tuple, Optional

sys.path.append("..")

# Import all implementations
implementations = {}

# Try to import each implementation
try:
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        UnifiedHilbertAttention,
    )

    implementations["Core"] = UnifiedHilbertAttention
except Exception as e:
    print(f"Failed to import Core: {e}")

try:
    from dilated_attention_pytorch.kernels.hilbert_attention_unified import (
        UnifiedHilbertAttention,
    )

    implementations["Unified"] = UnifiedHilbertAttention
except Exception as e:
    print(f"Failed to import Unified: {e}")

try:
    from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized import (
        UnifiedHilbertAttentionOptimized,
    )

    implementations["UnifiedOptimized"] = UnifiedHilbertAttentionOptimized
except Exception as e:
    print(f"Failed to import UnifiedOptimized: {e}")

try:
    from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced import (
        UnifiedHilbertAttentionOptimizedEnhanced,
    )

    implementations["UnifiedOptimizedEnhanced"] = (
        UnifiedHilbertAttentionOptimizedEnhanced
    )
except Exception as e:
    print(f"Failed to import UnifiedOptimizedEnhanced: {e}")


def benchmark_implementation(
    impl_class: type,
    impl_name: str,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_heads: int,
    segment_size: int,
    dilation_rate: int,
    warmup: int = 3,
    runs: int = 10,
) -> Optional[Tuple[float, bool]]:
    """Benchmark a single implementation."""
    try:
        # Create module
        kwargs = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "segment_size": segment_size,
            "dilation_rate": dilation_rate,
            "hilbert_threshold": 1024,
        }

        # Add extra params for enhanced version
        if "Enhanced" in impl_name:
            kwargs["enable_8k_optimization"] = True
            kwargs["enable_multi_row"] = True

        module = impl_class(**kwargs).cuda()
        module.eval()

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                try:
                    _ = module(x)
                except Exception as e:
                    print(f"  ✗ Runtime error during warmup: {str(e)}")
                    return None
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            with torch.no_grad():
                out = module(x)
            torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time = (end - start) / runs * 1000  # ms

        # Check if output is valid
        if torch.isnan(out).any() or torch.isinf(out).any():
            print("  ⚠ Warning: Output contains NaN or Inf")
            return avg_time, False

        return avg_time, True

    except Exception as e:
        print(f"  ✗ Failed to benchmark: {str(e)}")
        if "CompilationError" in str(type(e)):
            print("  Triton compilation error details:")
            traceback.print_exc()
        return None


def main():
    # Test parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    # Test configurations
    test_configs = [
        # (seq_len, dilation_rate, description)
        (512, 1, "Dense 512"),
        (1024, 1, "Dense 1K"),
        (2048, 1, "Dense 2K"),
        (4096, 1, "Dense 4K"),
        (8192, 1, "Dense 8K"),
        (2048, 2, "Sparse 2K (d=2)"),
        (4096, 2, "Sparse 4K (d=2)"),
        (4096, 4, "Sparse 4K (d=4)"),
    ]

    print("=== Hilbert Attention Implementations Benchmark ===")
    print(f"Batch: {batch_size}, Hidden: {hidden_dim}, Heads: {num_heads}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()

    # Results storage
    results: Dict[str, Dict[str, Optional[float]]] = {}

    # Benchmark each implementation
    for impl_name, impl_class in implementations.items():
        print(f"\n--- {impl_name} Implementation ---")
        results[impl_name] = {}

        for seq_len, dilation_rate, desc in test_configs:
            print(f"\n{desc}:")

            result = benchmark_implementation(
                impl_class=impl_class,
                impl_name=impl_name,
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )

            if result is not None:
                time_ms, valid = result
                results[impl_name][desc] = time_ms
                status = "✓" if valid else "⚠"
                print(f"  {status} Time: {time_ms:.2f}ms")
            else:
                results[impl_name][desc] = None
                print("  ✗ Failed")

    # Print comparison table
    print("\n\n=== Comparison Table ===")
    print(f"{'Config':<20}", end="")
    for impl_name in implementations.keys():
        print(f" | {impl_name:<15}", end="")
    print()
    print("-" * (20 + len(implementations) * 18))

    for seq_len, dilation_rate, desc in test_configs:
        print(f"{desc:<20}", end="")

        # Find baseline (first successful implementation)
        baseline_time = None
        for impl_name in implementations.keys():
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                baseline_time = results[impl_name][desc]
                break

        for impl_name in implementations.keys():
            if desc in results[impl_name] and results[impl_name][desc] is not None:
                time_ms = results[impl_name][desc]
                if baseline_time and baseline_time > 0:
                    speedup = baseline_time / time_ms
                    print(f" | {time_ms:>6.2f}ms ({speedup:.2f}x)", end="")
                else:
                    print(f" | {time_ms:>6.2f}ms", end="")
            else:
                print(f" | {'Failed':>15}", end="")
        print()

    # Special tests for 8K optimization
    if "UnifiedOptimizedEnhanced" in implementations:
        print("\n\n=== 8K Optimization Test (Enhanced Implementation) ===")

        # Test with 8K optimization disabled
        enhanced_no_8k = implementations["UnifiedOptimizedEnhanced"](
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=1,
            hilbert_threshold=1024,
            enable_8k_optimization=False,
        ).cuda()

        # Test with 8K optimization enabled
        enhanced_8k = implementations["UnifiedOptimizedEnhanced"](
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=1,
            hilbert_threshold=1024,
            enable_8k_optimization=True,
        ).cuda()

        x_8k = torch.randn(1, 8192, hidden_dim).cuda()

        # Benchmark both
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = enhanced_no_8k(x_8k)
                _ = enhanced_8k(x_8k)
            torch.cuda.synchronize()

            # No 8K optimization
            start = time.perf_counter()
            for _ in range(10):
                _ = enhanced_no_8k(x_8k)
            torch.cuda.synchronize()
            time_no_8k = (time.perf_counter() - start) / 10 * 1000

            # With 8K optimization
            start = time.perf_counter()
            for _ in range(10):
                _ = enhanced_8k(x_8k)
            torch.cuda.synchronize()
            time_8k = (time.perf_counter() - start) / 10 * 1000

        print(f"Without 8K optimization: {time_no_8k:.2f}ms")
        print(f"With 8K optimization:    {time_8k:.2f}ms")
        print(f"Speedup:                 {time_no_8k / time_8k:.2f}x")


if __name__ == "__main__":
    main()
