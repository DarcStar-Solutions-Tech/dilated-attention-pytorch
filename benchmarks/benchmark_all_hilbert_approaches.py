#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all Hilbert optimization approaches.

This tests:
1. Standard block-sparse attention (baseline)
2. Original Hilbert V1 (late reordering)
3. Dilation-aware Hilbert (grouped optimization)
4. Post-pattern Hilbert (processing order optimization)
5. Memory layout optimization (data reordering)
"""

import torch
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict
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
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    create_post_pattern_hilbert_attention,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_memory_layout import (
    create_memory_layout_optimized_attention,
)


@dataclass
class ComprehensiveBenchmarkResult:
    """Results from comprehensive Hilbert benchmark."""

    approach: str
    sequence_length: int
    dilation_rate: int
    forward_time_ms: float
    memory_mb: float
    speedup: float = 1.0
    notes: str = ""


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_approach(
    approach_name: str,
    model,
    seq_len: int,
    dilation_rate: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> ComprehensiveBenchmarkResult:
    """Benchmark a specific approach."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    try:
        for _ in range(warmup_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
    except Exception as e:
        return ComprehensiveBenchmarkResult(
            approach=approach_name,
            sequence_length=seq_len,
            dilation_rate=dilation_rate,
            forward_time_ms=float("inf"),
            memory_mb=0,
            speedup=0,
            notes=f"Warmup failed: {str(e)[:50]}",
        )

    # Measure memory
    mem_before = (
        torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    )

    # Benchmark
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(benchmark_iters):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()

        elapsed_time = (time.perf_counter() - start_time) / benchmark_iters * 1000

        mem_after = (
            torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        )
        memory_used = mem_after - mem_before

        return ComprehensiveBenchmarkResult(
            approach=approach_name,
            sequence_length=seq_len,
            dilation_rate=dilation_rate,
            forward_time_ms=elapsed_time,
            memory_mb=memory_used,
        )

    except Exception as e:
        return ComprehensiveBenchmarkResult(
            approach=approach_name,
            sequence_length=seq_len,
            dilation_rate=dilation_rate,
            forward_time_ms=float("inf"),
            memory_mb=0,
            speedup=0,
            notes=f"Benchmark failed: {str(e)[:50]}",
        )
    finally:
        del q, k, v
        if "output" in locals():
            del output
        clear_gpu_memory()


def create_all_models(
    segment_length: int,
    dilation_rate: int,
    sparsity_ratio: float = 0.1,
    block_size: int = 64,
) -> Dict[str, any]:
    """Create all model variants for comparison."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}

    # Standard baseline
    try:
        models["standard"] = create_block_sparse_attention(
            variant="base",
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparse_config=SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=sparsity_ratio,
                block_size=block_size,
            ),
        ).to(device)
    except Exception as e:
        print(f"Failed to create standard model: {e}")

    # Hilbert V1
    try:
        models["hilbert_v1"] = create_block_sparse_attention(
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
    except Exception as e:
        print(f"Failed to create Hilbert V1 model: {e}")

    # Dilation-aware
    try:
        models["dilation_aware"] = create_dilation_aware_hilbert_attention(
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        ).to(device)
    except Exception as e:
        print(f"Failed to create dilation-aware model: {e}")

    # Post-pattern
    try:
        models["post_pattern"] = create_post_pattern_hilbert_attention(
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        ).to(device)
    except Exception as e:
        print(f"Failed to create post-pattern model: {e}")

    # Memory layout
    try:
        models["memory_layout"] = create_memory_layout_optimized_attention(
            segment_lengths=[segment_length],
            dilation_rates=[dilation_rate],
            sparsity_ratio=sparsity_ratio,
            block_size=block_size,
        ).to(device)
    except Exception as e:
        print(f"Failed to create memory layout model: {e}")

    return models


def main():
    """Run comprehensive Hilbert approach comparison."""

    print("=" * 80)
    print("Comprehensive Hilbert Optimization Approaches Benchmark")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    # Test configurations
    seq_lengths = [4096, 8192]
    dilation_rates = [1, 2, 4, 8]

    all_results = []

    for seq_len in seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"SEQUENCE LENGTH: {seq_len}")
        print(f"{'=' * 80}")

        for dilation_rate in dilation_rates:
            print(f"\nDilation Rate: {dilation_rate}")
            print("-" * 60)

            segment_length = seq_len // 4

            # Create all models
            models = create_all_models(segment_length, dilation_rate)

            # Benchmark each approach
            results_for_config = {}

            for approach_name, model in models.items():
                result = benchmark_approach(
                    approach_name,
                    model,
                    seq_len,
                    dilation_rate,
                )

                results_for_config[approach_name] = result
                all_results.append(result)

                # Clean up model
                del model
                clear_gpu_memory()

            # Calculate speedups relative to standard
            if "standard" in results_for_config:
                standard_time = results_for_config["standard"].forward_time_ms

                # Print results table
                print(
                    f"\n{'Approach':20} {'Time (ms)':>10} {'Memory (MB)':>12} {'Speedup':>10} {'Notes':>20}"
                )
                print("-" * 75)

                for approach, result in results_for_config.items():
                    if result.forward_time_ms < float("inf"):
                        speedup = standard_time / result.forward_time_ms
                        result.speedup = speedup
                        print(
                            f"{approach:20} {result.forward_time_ms:10.2f} {result.memory_mb:12.1f} {speedup:10.2f}x {result.notes:>20}"
                        )
                    else:
                        print(
                            f"{approach:20} {'Failed':>10} {result.memory_mb:12.1f} {'N/A':>10} {result.notes:>20}"
                        )

    # Overall analysis
    print("\n" + "=" * 80)
    print("OVERALL ANALYSIS")
    print("=" * 80)

    # Group results by approach
    approach_results = {}
    for result in all_results:
        if result.approach not in approach_results:
            approach_results[result.approach] = []
        if result.forward_time_ms < float("inf"):
            approach_results[result.approach].append(result)

    # Calculate average performance by approach
    print("\nAverage Performance by Approach:")
    print(
        f"{'Approach':20} {'Avg Speedup':>12} {'Best':>10} {'Worst':>10} {'Success Rate':>15}"
    )
    print("-" * 70)

    for approach, results in approach_results.items():
        if results and approach != "standard":
            speedups = [r.speedup for r in results if r.speedup > 0]
            if speedups:
                avg_speedup = sum(speedups) / len(speedups)
                best_speedup = max(speedups)
                worst_speedup = min(speedups)
                success_rate = (
                    len(results) / (len(seq_lengths) * len(dilation_rates)) * 100
                )
                print(
                    f"{approach:20} {avg_speedup:12.2f}x {best_speedup:10.2f}x {worst_speedup:10.2f}x {success_rate:14.1f}%"
                )

    # Best approach by dilation rate
    print("\nBest Approach by Dilation Rate:")
    for dilation in dilation_rates:
        dilation_results = [
            r for r in all_results if r.dilation_rate == dilation and r.speedup > 0
        ]
        if dilation_results:
            best = max(dilation_results, key=lambda r: r.speedup)
            if best.approach != "standard":
                print(f"  Dilation {dilation}: {best.approach} ({best.speedup:.2f}x)")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"hilbert_comprehensive_benchmark_{timestamp}.json"

    results_dict = []
    for r in all_results:
        results_dict.append(
            {
                "approach": r.approach,
                "sequence_length": r.sequence_length,
                "dilation_rate": r.dilation_rate,
                "forward_time_ms": r.forward_time_ms
                if r.forward_time_ms < float("inf")
                else None,
                "memory_mb": r.memory_mb,
                "speedup": r.speedup,
                "notes": r.notes,
            }
        )

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "device": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                },
                "results": results_dict,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Final recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. Overall: Standard implementation remains fastest")
    print("2. If using Hilbert optimization:")
    print("   - Post-pattern shows least overhead")
    print("   - Dilation-aware helps with higher dilation rates")
    print("   - Memory layout has high overhead from data reordering")
    print("   - Original V1 should be avoided")
    print("\n3. GPU architecture strongly favors simple access patterns")


if __name__ == "__main__":
    main()
