#!/usr/bin/env python3
"""
Benchmark dilation-aware Hilbert implementation vs standard and original Hilbert.

This tests whether grouping blocks by dilation pattern before applying Hilbert
ordering improves performance.
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
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_dilation_aware import (
    create_dilation_aware_hilbert_attention,
)


@dataclass
class DilationAwareBenchmarkResult:
    """Results from benchmarking dilation-aware Hilbert."""

    implementation: str  # "standard", "hilbert_v1", or "dilation_aware"
    sequence_length: int
    dilation_rate: int
    forward_time_ms: float
    memory_mb: float
    speedup: float = 1.0
    access_groups: int = 0


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_implementation(
    implementation: str,
    seq_len: int,
    dilation_rates: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    block_size: int = 64,
    sparsity_ratio: float = 0.1,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> List[DilationAwareBenchmarkResult]:
    """Benchmark a specific implementation with various dilation rates."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"\n{implementation.upper()} Implementation:")
    print("-" * 70)

    for dilation_rate in dilation_rates:
        clear_gpu_memory()

        # Calculate segment length
        segment_length = seq_len // 4  # Use 4 segments

        try:
            # Create model based on implementation
            if implementation == "standard":
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
                access_groups = 0

            elif implementation == "hilbert_v1":
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
                access_groups = 0

            elif implementation == "dilation_aware":
                model = create_dilation_aware_hilbert_attention(
                    segment_lengths=[segment_length],
                    dilation_rates=[dilation_rate],
                    sparsity_ratio=sparsity_ratio,
                    block_size=block_size,
                ).to(device)
                # Get number of access groups
                num_blocks = seq_len // block_size
                access_groups = len(
                    model.hilbert_ordering.get_dilation_access_groups(
                        num_blocks, dilation_rate
                    )
                )

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

            # Warmup
            for _ in range(warmup_iters):
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _ = model(q, k, v)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            # Measure memory
            mem_before = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

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
                f"  Dilation {dilation_rate:2d}: {elapsed_time:7.2f}ms, {memory_used:6.1f}MB",
                end="",
            )
            if implementation == "dilation_aware":
                print(f" ({access_groups} groups)")
            else:
                print()

            results.append(
                DilationAwareBenchmarkResult(
                    implementation=implementation,
                    sequence_length=seq_len,
                    dilation_rate=dilation_rate,
                    forward_time_ms=elapsed_time,
                    memory_mb=memory_used,
                    access_groups=access_groups,
                )
            )

            del model, q, k, v, output

        except Exception as e:
            print(f"  Dilation {dilation_rate:2d}: Error - {str(e)[:40]}")

    return results


def analyze_results(
    standard_results: List[DilationAwareBenchmarkResult],
    hilbert_v1_results: List[DilationAwareBenchmarkResult],
    dilation_aware_results: List[DilationAwareBenchmarkResult],
) -> Dict[int, Dict[str, float]]:
    """Analyze and compare results."""

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    analysis = {}

    print(
        f"\n{'Dilation':>10} {'Standard':>12} {'Hilbert V1':>12} {'Dil-Aware':>12} {'V1 Speedup':>12} {'DA Speedup':>12}"
    )
    print("-" * 80)

    for i, std in enumerate(standard_results):
        if i < len(hilbert_v1_results) and i < len(dilation_aware_results):
            v1 = hilbert_v1_results[i]
            da = dilation_aware_results[i]

            v1_speedup = std.forward_time_ms / v1.forward_time_ms
            da_speedup = std.forward_time_ms / da.forward_time_ms

            print(
                f"{std.dilation_rate:10d} {std.forward_time_ms:11.2f}ms {v1.forward_time_ms:11.2f}ms {da.forward_time_ms:11.2f}ms {v1_speedup:11.2f}x {da_speedup:11.2f}x"
            )

            analysis[std.dilation_rate] = {
                "standard_ms": std.forward_time_ms,
                "hilbert_v1_ms": v1.forward_time_ms,
                "dilation_aware_ms": da.forward_time_ms,
                "v1_speedup": v1_speedup,
                "da_speedup": da_speedup,
                "access_groups": da.access_groups,
            }

    # Calculate averages
    avg_v1_speedup = sum(d["v1_speedup"] for d in analysis.values()) / len(analysis)
    avg_da_speedup = sum(d["da_speedup"] for d in analysis.values()) / len(analysis)

    print("-" * 80)
    print(
        f"{'Average':>10} {'-':>12} {'-':>12} {'-':>12} {avg_v1_speedup:11.2f}x {avg_da_speedup:11.2f}x"
    )

    return analysis


def main():
    """Run comprehensive dilation-aware benchmarks."""

    print("=" * 80)
    print("Dilation-Aware Hilbert Optimization Benchmark")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Test configurations
    seq_lengths = [4096, 8192, 16384]
    dilation_rates = [1, 2, 4, 8]

    all_results = {}

    for seq_len in seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'=' * 80}")

        # Benchmark all implementations
        standard_results = benchmark_implementation("standard", seq_len, dilation_rates)
        hilbert_v1_results = benchmark_implementation(
            "hilbert_v1", seq_len, dilation_rates
        )
        dilation_aware_results = benchmark_implementation(
            "dilation_aware", seq_len, dilation_rates
        )

        # Analyze results
        analysis = analyze_results(
            standard_results, hilbert_v1_results, dilation_aware_results
        )
        all_results[seq_len] = analysis

    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)

    print("\n1. Dilation-Aware vs Original Hilbert:")
    improvements = []
    for seq_len, seq_data in all_results.items():
        for dilation, data in seq_data.items():
            improvement = (
                (data["da_speedup"] - data["v1_speedup"]) / data["v1_speedup"] * 100
            )
            improvements.append((dilation, improvement))

    avg_improvement = sum(imp for _, imp in improvements) / len(improvements)
    print(f"   Average improvement: {avg_improvement:+.1f}%")

    # Best cases
    best_improvement = max(improvements, key=lambda x: x[1])
    print(
        f"   Best improvement: {best_improvement[1]:+.1f}% at dilation={best_improvement[0]}"
    )

    print("\n2. Performance vs Standard:")
    for dilation in dilation_rates:
        speedups = [
            all_results[seq_len][dilation]["da_speedup"]
            for seq_len in seq_lengths
            if dilation in all_results[seq_len]
        ]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0

        if avg_speedup > 1.0:
            verdict = "✓ FASTER than standard"
        elif avg_speedup > 0.9:
            verdict = "~ Nearly equal"
        else:
            verdict = "✗ Slower than standard"

        print(f"   Dilation {dilation}: {avg_speedup:.2f}x - {verdict}")

    print("\n3. Key Insights:")
    print("   - Dilation-aware grouping preserves dilated access patterns")
    print("   - Hilbert ordering within groups improves cache locality")
    print("   - Performance depends on group size and structure")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"dilation_aware_hilbert_benchmark_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
