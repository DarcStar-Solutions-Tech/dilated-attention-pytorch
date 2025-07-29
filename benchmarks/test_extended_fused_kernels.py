#!/usr/bin/env python3
"""Test extended fused kernel support for 8K-16K sequences."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def benchmark_sequence(seq_len, batch_size=1, num_runs=10, warmup=3):
    """Benchmark a specific sequence length."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("CUDA not available, skipping benchmarks")
        return {}

    # Create module
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=seq_len + 1,  # Disable Hilbert for pure kernel test
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, 768, device=device)

    results = {}

    # Test configurations
    configs = [
        ("PyTorch Baseline", False),
        ("Fused Kernel", True),
    ]

    for config_name, use_fused in configs:
        # Control which backend is used
        original_triton = module._triton_available
        original_fused = module._fused_kernels_available

        if config_name == "PyTorch Baseline":
            module._triton_available = False
            module._fused_kernels_available = False
        else:
            module._triton_available = True
            module._fused_kernels_available = True

        try:
            with torch.no_grad():
                # Warmup
                for _ in range(warmup):
                    _ = module(x, use_hilbert=False)

                # Time
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(num_runs):
                    _ = module(x, use_hilbert=False)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / num_runs * 1000

                results[config_name] = elapsed

        except Exception as e:
            results[config_name] = f"Error: {str(e)}"
        finally:
            # Restore
            module._triton_available = original_triton
            module._fused_kernels_available = original_fused

    return results


def main():
    print("Testing Extended Fused Kernel Support (8K-16K)")
    print("=" * 60)

    # Test extended range
    seq_lengths = [2048, 4096, 8192, 12288, 16384]

    all_results = []

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len:,}")
        print("-" * 40)

        # For very long sequences, reduce number of runs
        num_runs = 5 if seq_len > 8192 else 10

        results = benchmark_sequence(seq_len, num_runs=num_runs)

        if isinstance(results.get("PyTorch Baseline"), float) and isinstance(
            results.get("Fused Kernel"), float
        ):
            baseline = results["PyTorch Baseline"]
            fused = results["Fused Kernel"]
            speedup = baseline / fused

            print(f"  PyTorch Baseline: {baseline:>8.2f}ms")
            print(f"  Fused Kernel:     {fused:>8.2f}ms")
            print(f"  Speedup:          {speedup:>8.2f}x")

            all_results.append(
                {
                    "seq_len": seq_len,
                    "baseline": baseline,
                    "fused": fused,
                    "speedup": speedup,
                }
            )
        else:
            for name, result in results.items():
                print(f"  {name}: {result}")

    # Summary
    if all_results:
        print("\n\nSUMMARY")
        print("=" * 60)
        print(
            f"{'Seq Length':<12} {'Baseline (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10}"
        )
        print("-" * 60)

        for result in all_results:
            print(
                f"{result['seq_len']:<12,} {result['baseline']:<15.2f} {result['fused']:<15.2f} {result['speedup']:<10.2f}x"
            )

        # Recommendations
        print("\n\nRECOMMENDATIONS")
        print("=" * 60)

        # Check 8K-16K performance
        extended_results = [r for r in all_results if 8192 <= r["seq_len"] <= 16384]
        if extended_results:
            avg_speedup = sum(r["speedup"] for r in extended_results) / len(
                extended_results
            )

            if avg_speedup > 1.3:
                print(
                    f"✅ Extended fused kernels (8K-16K) show {avg_speedup:.2f}x average speedup"
                )
                print("   The extension is working well and should be kept!")
            else:
                print(
                    f"⚠️  Extended fused kernels (8K-16K) show only {avg_speedup:.2f}x average speedup"
                )
                print("   Consider further optimization or limiting to 8K")


if __name__ == "__main__":
    main()
