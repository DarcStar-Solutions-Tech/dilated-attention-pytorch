#!/usr/bin/env python3
"""
Benchmark Hilbert optimization with different dilation rates.

This tests whether Hilbert ordering helps more with dilated patterns,
where the natural access pattern is already non-contiguous.
"""

import torch
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)


@dataclass
class DilationBenchmarkResult:
    """Results from benchmarking Hilbert with different dilation rates."""

    implementation: str  # "standard" or "hilbert"
    sequence_length: int
    dilation_rate: int
    segment_length: int
    forward_time_ms: float
    memory_mb: float
    speedup: float = 1.0


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_with_dilation(
    variant: str,
    seq_len: int,
    dilation_rates: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 64,
    sparsity_ratio: float = 0.05,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> List[DilationBenchmarkResult]:
    """Benchmark a specific implementation with various dilation rates."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"\n{variant.upper()} Implementation:")
    print("-" * 70)
    print(f"{'Dilation':>10} {'Segment':>10} {'Time (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 70)

    for dilation_rate in dilation_rates:
        clear_gpu_memory()

        # Calculate appropriate segment length for this dilation rate
        # Ensure seq_len is divisible by (segment_length * dilation_rate)
        base_segment = seq_len // (4 * dilation_rate)  # Use 4 segments
        segment_length = base_segment * dilation_rate

        # Adjust segment length to ensure divisibility
        while seq_len % segment_length != 0:
            segment_length -= dilation_rate

        if segment_length <= 0:
            print(f"{dilation_rate:10d} - Skipped (invalid segment length)")
            continue

        try:
            # Create model with dilation
            if variant == "hilbert":
                model = create_block_sparse_attention(
                    variant="hilbert",
                    segment_lengths=[segment_length],
                    dilation_rates=[dilation_rate],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=sparsity_ratio,
                        block_size=block_size,
                    ),
                    hilbert_block_level=True,
                    hilbert_within_blocks=False,
                ).to(device)
            else:
                model = create_block_sparse_attention(
                    variant="base",
                    segment_lengths=[segment_length],
                    dilation_rates=[dilation_rate],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=sparsity_ratio,
                        block_size=block_size,
                    ),
                ).to(device)

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Measure memory before
            mem_before = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

            # Warmup
            for _ in range(warmup_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Benchmark
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(benchmark_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    output = model(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            elapsed_time = (time.perf_counter() - start_time) / benchmark_iters * 1000

            # Measure memory
            mem_after = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )
            memory_used = mem_after - mem_before

            print(
                f"{dilation_rate:10d} {segment_length:10d} {elapsed_time:12.2f} {memory_used:12.1f}"
            )

            results.append(
                DilationBenchmarkResult(
                    implementation=variant,
                    sequence_length=seq_len,
                    dilation_rate=dilation_rate,
                    segment_length=segment_length,
                    forward_time_ms=elapsed_time,
                    memory_mb=memory_used,
                )
            )

            # Cleanup
            del model, q, k, v, output

        except Exception as e:
            print(f"{dilation_rate:10d} - Error: {str(e)[:40]}")

    return results


def analyze_dilation_impact(
    seq_lengths: List[int],
    max_dilation: int = 8,
) -> Dict[int, Dict[str, float]]:
    """Analyze how Hilbert performance changes with dilation rate."""

    dilation_rates = [1, 2, 4, 8][:max_dilation]
    analysis = {}

    for seq_len in seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'=' * 80}")

        # Benchmark both implementations
        standard_results = benchmark_with_dilation("standard", seq_len, dilation_rates)
        hilbert_results = benchmark_with_dilation("hilbert", seq_len, dilation_rates)

        # Calculate speedups
        print(f"\n{'Speedup Analysis':^70}")
        print("-" * 70)
        print(
            f"{'Dilation':>10} {'Standard (ms)':>15} {'Hilbert (ms)':>15} {'Speedup':>10}"
        )
        print("-" * 70)

        seq_analysis = {}

        for std, hlb in zip(standard_results, hilbert_results):
            if std.dilation_rate == hlb.dilation_rate:
                speedup = std.forward_time_ms / hlb.forward_time_ms
                print(
                    f"{std.dilation_rate:10d} {std.forward_time_ms:15.2f} {hlb.forward_time_ms:15.2f} {speedup:10.2f}x"
                )

                seq_analysis[std.dilation_rate] = {
                    "standard_ms": std.forward_time_ms,
                    "hilbert_ms": hlb.forward_time_ms,
                    "speedup": speedup,
                }

        analysis[seq_len] = seq_analysis

    return analysis


def main():
    """Run comprehensive dilation rate benchmarks."""

    print("=" * 80)
    print("Hilbert Optimization with Dilation Rates Benchmark")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Test configurations
    seq_lengths = [4096, 8192, 16384]

    # Run analysis
    analysis = analyze_dilation_impact(seq_lengths)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Speedup by Dilation Rate")
    print("=" * 80)

    # Calculate average speedup by dilation rate
    dilation_speedups = {}

    for seq_len, seq_data in analysis.items():
        for dilation, data in seq_data.items():
            if dilation not in dilation_speedups:
                dilation_speedups[dilation] = []
            dilation_speedups[dilation].append(data["speedup"])

    print(f"\n{'Dilation Rate':>15} {'Avg Speedup':>15} {'Interpretation':>30}")
    print("-" * 60)

    for dilation in sorted(dilation_speedups.keys()):
        avg_speedup = sum(dilation_speedups[dilation]) / len(
            dilation_speedups[dilation]
        )

        if avg_speedup > 1.0:
            interpretation = "Hilbert FASTER ✓"
        elif avg_speedup > 0.9:
            interpretation = "Nearly equal"
        else:
            interpretation = "Hilbert slower ✗"

        print(f"{dilation:15d} {avg_speedup:15.2f}x {interpretation:>30}")

    # Detailed analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nKey Findings:")
    print("1. Impact of Dilation on Hilbert Performance:")

    # Check if performance improves with dilation
    if len(dilation_speedups) > 1:
        d1_speedup = sum(dilation_speedups.get(1, [0])) / len(
            dilation_speedups.get(1, [1])
        )
        max_dilation = max(dilation_speedups.keys())
        max_speedup = sum(dilation_speedups.get(max_dilation, [0])) / len(
            dilation_speedups.get(max_dilation, [1])
        )

        improvement = (max_speedup - d1_speedup) / d1_speedup * 100

        if improvement > 10:
            print(
                f"   ✓ Hilbert performs {improvement:.1f}% better with dilation={max_dilation} vs dilation=1"
            )
            print("   → Hilbert ordering helps more with dilated patterns!")
        elif improvement > 0:
            print(
                f"   ~ Slight improvement ({improvement:.1f}%) with higher dilation rates"
            )
        else:
            print(f"   ✗ No improvement with dilation (change: {improvement:.1f}%)")

    print("\n2. Why Dilation Might Help Hilbert:")
    print("   - Dilation creates naturally scattered access patterns")
    print("   - Hilbert reordering can group scattered accesses")
    print("   - Cache benefits may outweigh reordering overhead")

    print("\n3. Recommendations:")
    if any(avg > 1.0 for avg in [sum(v) / len(v) for v in dilation_speedups.values()]):
        print("   ✓ Consider Hilbert optimization for dilated attention")
        print("   ✓ Especially beneficial for dilation rates > 2")
    else:
        print("   ✗ Hilbert still not beneficial even with dilation")
        print("   ✗ Stick with standard implementation")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"hilbert_dilation_benchmark_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "analysis": analysis,
                "summary": {
                    f"dilation_{k}": {"avg_speedup": sum(v) / len(v)}
                    for k, v in dilation_speedups.items()
                },
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
