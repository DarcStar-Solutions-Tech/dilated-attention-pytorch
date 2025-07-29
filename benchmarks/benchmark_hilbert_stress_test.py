#!/usr/bin/env python3
"""
Stress test for Hilbert implementations with edge cases and various configurations.
"""

import torch
import time
import sys
from typing import Dict, Tuple, Optional
import gc

sys.path.append("..")

# Import the best performing implementations
from dilated_attention_pytorch.kernels.hilbert_attention_unified import (
    UnifiedHilbertAttention,
)
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized import (
    UnifiedHilbertAttentionOptimized,
)
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced import (
    UnifiedHilbertAttentionOptimizedEnhanced,
)

implementations = {
    "Unified": UnifiedHilbertAttention,
    "Optimized": UnifiedHilbertAttentionOptimized,
    "Enhanced": UnifiedHilbertAttentionOptimizedEnhanced,
}


def test_configuration(
    impl_class: type,
    impl_name: str,
    config: Dict[str, any],
    x: torch.Tensor,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """Test a specific configuration."""
    try:
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Create module
        module = impl_class(**config).cuda()
        module.eval()

        # Test forward pass
        with torch.no_grad():
            # First pass (may compile)
            out = module(x)
            torch.cuda.synchronize()

            # Check output validity
            if torch.isnan(out).any():
                return False, None, "Output contains NaN"
            if torch.isinf(out).any():
                return False, None, "Output contains Inf"

            # Time subsequent passes
            start = time.perf_counter()
            for _ in range(5):
                out = module(x)
            torch.cuda.synchronize()
            end = time.perf_counter()

            avg_time = (end - start) / 5 * 1000  # ms

            return True, avg_time, None

    except torch.cuda.OutOfMemoryError:
        return False, None, "OOM"
    except Exception as e:
        error_type = type(e).__name__
        if "CompilationError" in error_type:
            # Extract line info from compilation error
            error_msg = str(e)
            if "at " in error_msg:
                line_info = error_msg.split("at ")[1].split(":")[0]
                return False, None, f"Triton compile error at {line_info}"
            return False, None, "Triton compilation error"
        return False, None, f"{error_type}: {str(e)[:50]}"
    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()


def main():
    print("=== Hilbert Implementation Stress Test ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print()

    # Test configurations
    test_cases = [
        # Standard tests
        ("Small", 2, 256, 256, 4, 64, 1),
        ("Medium", 2, 1024, 512, 8, 128, 1),
        ("Large", 1, 4096, 512, 8, 128, 1),
        ("XLarge", 1, 8192, 512, 8, 128, 1),
        # Edge cases
        ("Tiny", 4, 64, 128, 4, 32, 1),
        ("Odd size", 2, 1023, 512, 8, 128, 1),
        ("Prime size", 2, 1009, 512, 8, 128, 1),
        ("Large heads", 1, 1024, 512, 16, 128, 1),
        ("Small heads", 2, 1024, 512, 2, 128, 1),
        # Sparse patterns
        ("Sparse d=2", 2, 2048, 512, 8, 128, 2),
        ("Sparse d=4", 1, 4096, 512, 8, 128, 4),
        ("Sparse d=8", 1, 8192, 512, 8, 128, 8),
        # Segment size variations
        ("Small seg", 2, 1024, 512, 8, 64, 1),
        ("Large seg", 2, 2048, 512, 8, 256, 1),
        ("Huge seg", 1, 4096, 512, 8, 512, 1),
        # Memory stress
        ("Max batch", 8, 512, 512, 8, 128, 1),
        ("Max seq", 1, 16384, 256, 4, 128, 1),
        ("Max dim", 1, 1024, 1024, 16, 128, 1),
    ]

    results = []

    for (
        test_name,
        batch_size,
        seq_len,
        hidden_dim,
        num_heads,
        segment_size,
        dilation_rate,
    ) in test_cases:
        print(f"\n--- {test_name} ---")
        print(
            f"Config: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, heads={num_heads}, seg={segment_size}, dil={dilation_rate}"
        )

        # Create input
        try:
            x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
        except torch.cuda.OutOfMemoryError:
            print("✗ Cannot allocate input tensor (OOM)")
            continue

        test_results = {
            "test": test_name,
            "config": (
                batch_size,
                seq_len,
                hidden_dim,
                num_heads,
                segment_size,
                dilation_rate,
            ),
        }

        for impl_name, impl_class in implementations.items():
            # Build config
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": segment_size,
                "dilation_rate": dilation_rate,
                "hilbert_threshold": 512,  # Use Hilbert for most tests
            }

            # Add implementation-specific params
            if impl_name == "Enhanced":
                config["enable_8k_optimization"] = seq_len == 8192
                config["enable_multi_row"] = seq_len >= 2048

            success, time_ms, error = test_configuration(
                impl_class, impl_name, config, x
            )

            if success:
                print(f"  {impl_name}: ✓ {time_ms:.2f}ms")
                test_results[impl_name] = ("success", time_ms)
            else:
                print(f"  {impl_name}: ✗ {error}")
                test_results[impl_name] = ("failed", error)

        results.append(test_results)

    # Summary table
    print("\n\n=== Summary Table ===")
    print(f"{'Test':<15} | {'Unified':<20} | {'Optimized':<20} | {'Enhanced':<20}")
    print("-" * 80)

    for result in results:
        test_name = result["test"]
        print(f"{test_name:<15}", end="")

        for impl in ["Unified", "Optimized", "Enhanced"]:
            if impl in result:
                status, value = result[impl]
                if status == "success":
                    print(f" | {value:>6.1f}ms {'':11}", end="")
                else:
                    print(f" | FAIL: {value:<14}", end="")
            else:
                print(f" | {'N/A':<20}", end="")
        print()

    # Failure analysis
    print("\n\n=== Failure Analysis ===")
    failures = {}

    for result in results:
        for impl in ["Unified", "Optimized", "Enhanced"]:
            if impl in result and result[impl][0] == "failed":
                error = result[impl][1]
                config = result["config"]

                if impl not in failures:
                    failures[impl] = []
                failures[impl].append((result["test"], config, error))

    for impl, fail_list in failures.items():
        if fail_list:
            print(f"\n{impl} Implementation Failures:")
            for test_name, config, error in fail_list:
                print(f"  - {test_name} {config}: {error}")

    # Performance patterns
    print("\n\n=== Performance Patterns ===")

    # Find where each implementation excels
    for impl in ["Unified", "Optimized", "Enhanced"]:
        wins = []
        for result in results:
            if all(
                i in result and result[i][0] == "success"
                for i in ["Unified", "Optimized", "Enhanced"]
            ):
                times = {i: result[i][1] for i in ["Unified", "Optimized", "Enhanced"]}
                if min(times, key=times.get) == impl:
                    wins.append((result["test"], times[impl]))

        if wins:
            print(f"\n{impl} performs best on:")
            for test_name, time_ms in wins:
                print(f"  - {test_name} ({time_ms:.1f}ms)")


if __name__ == "__main__":
    main()
