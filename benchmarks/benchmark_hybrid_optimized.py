# \!/usr/bin/env python3
"""
Benchmark the optimized hybrid ring dilated attention against the original.
Tests single GPU performance improvements from proper V2 integration.
"""

import torch
import time
import gc
from typing import Dict, List
import json

# Import both implementations
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid as HybridOriginal,
)
from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized import (
    RingDilatedAttentionHybridOptimized as HybridOptimized,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_implementation(
    model_class,
    model_name: str,
    seq_lengths: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    segment_lengths: list = None,
    dilation_rates: list = None,
    warmup_iterations: int = 2,
    benchmark_iterations: int = 5,
) -> List[Dict]:
    """Benchmark a specific implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if segment_lengths is None:
        segment_lengths = [512, 1024]
    if dilation_rates is None:
        dilation_rates = [1, 2]

    results = []

    for seq_len in seq_lengths:
        # Ensure divisibility
        max_seg = max(segment_lengths)
        if seq_len % max_seg != 0:
            seq_len = ((seq_len // max_seg) + 1) * max_seg

        try:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create model
            if model_name == "improved":
                model = model_class(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    enable_memory_pool=True,
                    lightweight_pool=False,
                )
            else:
                # Hybrid models
                model = model_class(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    ring_size=1,  # Single GPU
                    device=device,
                    dtype=torch.float32,
                    enable_memory_pool=True,
                    use_flash_attention=True,
                    use_pattern_cache=True,
                )

            # Create inputs
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Warmup
            for _ in range(warmup_iterations):
                with torch.no_grad():
                    _ = model(q, k, v)
                torch.cuda.synchronize()

            # Benchmark
            torch.cuda.reset_peak_memory_stats()
            times = []

            for _ in range(benchmark_iterations):
                torch.cuda.synchronize()
                start = time.time()

                with torch.no_grad():
                    output = model(q, k, v)

                torch.cuda.synchronize()
                times.append(time.time() - start)

            # Get stats
            avg_time = sum(times) / len(times)
            peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            memory_per_token = peak_memory * 1024 / seq_len  # KB per token
            throughput = seq_len * batch_size / avg_time

            result = {
                "model": model_name,
                "seq_len": seq_len,
                "time_ms": avg_time * 1000,
                "memory_mb": peak_memory,
                "memory_per_token_kb": memory_per_token,
                "throughput_tokens_per_sec": throughput,
            }

            results.append(result)

            print(
                f"{model_name} @ {seq_len:,}: {avg_time * 1000:.1f}ms, "
                f"{peak_memory:.0f}MB, {throughput:,.0f} tok/s"
            )

            # Clean up
            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{model_name} @ {seq_len:,}: OOM")
            break
        except Exception as e:
            print(f"{model_name} @ {seq_len:,}: Error - {e}")
            break

    return results


def main():
    """Run comparison benchmarks."""

    print("=== Hybrid Ring Attention Optimization Benchmark ===")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Test configurations
    test_sequences = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

    print("\n" + "=" * 60)
    print("Benchmarking Original Hybrid Implementation")
    print("=" * 60)

    original_results = benchmark_implementation(
        HybridOriginal, "hybrid_original", test_sequences
    )

    print("\n" + "=" * 60)
    print("Benchmarking Optimized Hybrid Implementation")
    print("=" * 60)

    optimized_results = benchmark_implementation(
        HybridOptimized, "hybrid_optimized", test_sequences
    )

    print("\n" + "=" * 60)
    print("Benchmarking Improved Dilated Attention (Reference)")
    print("=" * 60)

    improved_results = benchmark_implementation(
        ImprovedDilatedAttention, "improved", test_sequences
    )

    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Seq Length':>10}  < /dev/null |  {'Original':>12} | {'Optimized':>12} | {'Improved':>12} | {'Speedup':>8}"
    )
    print("-" * 70)

    for seq_len in test_sequences:
        orig = next((r for r in original_results if r["seq_len"] == seq_len), None)
        opt = next((r for r in optimized_results if r["seq_len"] == seq_len), None)
        imp = next((r for r in improved_results if r["seq_len"] == seq_len), None)

        if orig and opt:
            speedup = orig["time_ms"] / opt["time_ms"]
            print(
                f"{seq_len:>10,} | {orig['time_ms']:>10.1f}ms | "
                f"{opt['time_ms']:>10.1f}ms | "
                f"{imp['time_ms'] if imp else 'N/A':>10} | "
                f"{speedup:>6.2f}x"
            )

    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Seq Length':>10} | {'Original':>15} | {'Optimized':>15} | {'Improved':>15}"
    )
    print("-" * 60)

    for seq_len in test_sequences:
        orig = next((r for r in original_results if r["seq_len"] == seq_len), None)
        opt = next((r for r in optimized_results if r["seq_len"] == seq_len), None)
        imp = next((r for r in improved_results if r["seq_len"] == seq_len), None)

        if orig and opt:
            print(
                f"{seq_len:>10,} | {orig['memory_per_token_kb']:>13.1f}KB | "
                f"{opt['memory_per_token_kb']:>13.1f}KB | "
                f"{imp['memory_per_token_kb'] if imp else 0:>13.1f}KB"
            )

    # Calculate average improvements
    speedups = []
    memory_reductions = []

    for seq_len in test_sequences:
        orig = next((r for r in original_results if r["seq_len"] == seq_len), None)
        opt = next((r for r in optimized_results if r["seq_len"] == seq_len), None)

        if orig and opt:
            speedups.append(orig["time_ms"] / opt["time_ms"])
            memory_reductions.append(1 - opt["memory_mb"] / orig["memory_mb"])

    if speedups:
        print(f"\nAverage speedup: {sum(speedups) / len(speedups):.2f}x")
    if memory_reductions:
        print(
            f"Average memory reduction: {sum(memory_reductions) / len(memory_reductions) * 100:.1f}%"
        )

    print("\nKey Optimizations Applied:")
    print("✓ Properly initialized memory pools")
    print("✓ Pattern pre-computation and caching")
    print("✓ Batch segment processing")
    print("✓ Optimized memory access patterns")
    print("✓ Single-GPU fast path")
    print("✓ Direct use of optimize_attention_computation")

    # Save results
    results = {
        "original": original_results,
        "optimized": optimized_results,
        "improved": improved_results,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    with open("hybrid_optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to hybrid_optimization_results.json")


if __name__ == "__main__":
    main()
