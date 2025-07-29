#!/usr/bin/env python3
"""Verify the 8K performance fix."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def benchmark_sequence(seq_len, batch_size=1, num_runs=5, warmup=3):
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
            print(f"  {config_name}: {e}")
        finally:
            # Restore
            module._triton_available = original_triton
            module._fused_kernels_available = original_fused

    return results


def main():
    print("Verifying 8K Performance Fix")
    print("=" * 80)

    # Test the critical range around 8K
    seq_lengths = [4096, 6144, 8192, 10240, 12288, 16384]

    results_table = []

    print(
        f"{'Seq Length':<12} {'PyTorch (ms)':<15} {'Fused (ms)':<15} {'Speedup':<10} {'Status':<15}"
    )
    print("-" * 80)

    for seq_len in seq_lengths:
        # Use fewer runs for longer sequences
        num_runs = 3 if seq_len > 8192 else 5

        results = benchmark_sequence(seq_len, num_runs=num_runs, warmup=2)

        if isinstance(results.get("PyTorch Baseline"), float) and isinstance(
            results.get("Fused Kernel"), float
        ):
            baseline = results["PyTorch Baseline"]
            fused = results["Fused Kernel"]
            speedup = baseline / fused

            # Determine status
            if speedup < 1.2:
                status = "⚠️  Low speedup"
            elif speedup > 3.0:
                status = "🚀 Excellent"
            else:
                status = "✅ Good"

            print(
                f"{seq_len:<12} {baseline:<15.2f} {fused:<15.2f} {speedup:<10.2f}x {status:<15}"
            )

            results_table.append(
                {
                    "seq_len": seq_len,
                    "baseline": baseline,
                    "fused": fused,
                    "speedup": speedup,
                }
            )
        else:
            # Handle errors
            baseline_result = results.get("PyTorch Baseline", "Error")
            _ = results.get("Fused Kernel", "Error")

            if isinstance(baseline_result, float):
                print(
                    f"{seq_len:<12} {baseline_result:<15.2f} {'Error':<15} {'N/A':<10} ❌ Error"
                )
            else:
                print(f"{seq_len:<12} {'Error':<15} {'Error':<15} {'N/A':<10} ❌ Error")

    # Analysis
    if len(results_table) >= 3:
        print("\n\nPERFORMANCE ANALYSIS")
        print("=" * 80)

        # Check for 8K anomaly
        speedups = {r["seq_len"]: r["speedup"] for r in results_table}

        if 4096 in speedups and 8192 in speedups and 12288 in speedups:
            speedup_4k = speedups[4096]
            speedup_8k = speedups[8192]
            speedup_12k = speedups[12288]

            print("\nKey Performance Points:")
            print(f"  4K:  {speedup_4k:.2f}x speedup")
            print(f"  8K:  {speedup_8k:.2f}x speedup")
            print(f"  12K: {speedup_12k:.2f}x speedup")

            # Check if 8K is still an anomaly
            avg_neighbor = (speedup_4k + speedup_12k) / 2
            deviation = abs(speedup_8k - avg_neighbor) / avg_neighbor * 100

            print("\n8K Performance Analysis:")
            print(f"  Expected (avg of 4K & 12K): {avg_neighbor:.2f}x")
            print(f"  Actual: {speedup_8k:.2f}x")
            print(f"  Deviation: {deviation:.1f}%")

            if deviation > 30:
                print("  Status: ⚠️  ANOMALY STILL PRESENT")
                print("\nPossible causes:")
                print("  - Shared memory constraints still limiting performance")
                print("  - Grid alignment issues not fully resolved")
                print("  - Cache thrashing at this specific size")
            else:
                print("  Status: ✅ ANOMALY FIXED")
                print("\nThe performance curve is now smooth!")

        # Plot performance curve
        print("\n\nPerformance Curve:")
        print("-" * 50)

        max_speedup = max(r["speedup"] for r in results_table)
        for r in results_table:
            seq_k = r["seq_len"] // 1024
            bar_length = int(r["speedup"] / max_speedup * 40)
            bar = "█" * bar_length
            print(f"{seq_k:>3}K: {bar} {r['speedup']:.2f}x")


if __name__ == "__main__":
    main()
