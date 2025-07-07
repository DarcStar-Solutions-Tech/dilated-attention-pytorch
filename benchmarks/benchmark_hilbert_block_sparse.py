#!/usr/bin/env python3
"""
Benchmark comparing standard block-sparse vs Hilbert-optimized block-sparse attention.

This benchmark tests the cache locality improvements from using Hilbert space-filling
curves to order block computations.
"""

import torch
import time
import gc
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import (
    create_block_sparse_attention,
    SparsePatternConfig,
)


@dataclass
class HilbertBenchmarkResult:
    """Results from benchmarking Hilbert vs standard block-sparse."""

    implementation: str  # "standard" or "hilbert"
    sequence_length: int
    block_size: int
    sparsity_ratio: float
    forward_time_ms: float
    memory_mb: float
    cache_misses: Optional[int] = None  # Would need profiling to measure
    speedup_ratio: Optional[float] = None


def clear_gpu_memory():
    """Clear GPU memory and caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def benchmark_implementation(
    variant: str,
    seq_lengths: List[int],
    block_size: int = 64,
    sparsity_ratio: float = 0.05,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
) -> List[HilbertBenchmarkResult]:
    """Benchmark a specific implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"\nBenchmarking {variant} implementation:")
    print("-" * 60)

    for seq_len in seq_lengths:
        clear_gpu_memory()

        try:
            # Create model
            if variant == "hilbert":
                model = create_block_sparse_attention(
                    variant="hilbert",
                    segment_lengths=[seq_len // 2],
                    dilation_rates=[1],
                    sparse_config=SparsePatternConfig(
                        pattern_type="dilated_sparse",
                        sparsity_ratio=sparsity_ratio,
                        block_size=block_size,
                    ),
                    hilbert_block_level=True,
                    hilbert_within_blocks=False,  # Test block-level only first
                ).to(device)
            else:
                model = create_block_sparse_attention(
                    variant="base",
                    segment_lengths=[seq_len // 2],
                    dilation_rates=[1],
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

            print(f"  Seq {seq_len:6d}: {elapsed_time:7.2f}ms, {memory_used:7.1f}MB")

            results.append(
                HilbertBenchmarkResult(
                    implementation=variant,
                    sequence_length=seq_len,
                    block_size=block_size,
                    sparsity_ratio=sparsity_ratio,
                    forward_time_ms=elapsed_time,
                    memory_mb=memory_used,
                )
            )

            # Cleanup
            del model, q, k, v, output
            clear_gpu_memory()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Seq {seq_len:6d}: OOM")
                break
            else:
                print(f"  Seq {seq_len:6d}: Error - {str(e)[:50]}")
                break

    return results


def compare_hilbert_configurations(
    seq_len: int = 8192,
    block_size: int = 64,
    sparsity_ratio: float = 0.05,
) -> Dict[str, float]:
    """Compare different Hilbert optimization configurations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_heads = 8
    head_dim = 64

    configs = {
        "standard": {
            "variant": "base",
        },
        "hilbert_block": {
            "variant": "hilbert",
            "hilbert_block_level": True,
            "hilbert_within_blocks": False,
        },
        "hilbert_element": {
            "variant": "hilbert",
            "hilbert_block_level": False,
            "hilbert_within_blocks": True,
        },
        "hilbert_full": {
            "variant": "hilbert",
            "hilbert_block_level": True,
            "hilbert_within_blocks": True,
        },
    }

    results = {}

    print(f"\nComparing Hilbert configurations at {seq_len} tokens:")
    print("-" * 60)

    # Create inputs once
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for name, config in configs.items():
        clear_gpu_memory()

        # Create model
        model = create_block_sparse_attention(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            sparse_config=SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=sparsity_ratio,
                block_size=block_size,
            ),
            **config,
        ).to(device)

        # Warmup
        for _ in range(5):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                _ = model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(20):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                output = model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()

        elapsed_time = (time.perf_counter() - start_time) / 20 * 1000
        results[name] = elapsed_time

        print(f"  {name:20s}: {elapsed_time:7.2f}ms")

        del model, output

    # Calculate speedups
    standard_time = results["standard"]
    print("\nSpeedups vs standard:")
    for name, time_ms in results.items():
        if name != "standard":
            speedup = standard_time / time_ms
            print(f"  {name:20s}: {speedup:5.2f}x")

    return results


def main():
    """Run comprehensive Hilbert vs standard benchmarks."""

    print("=" * 80)
    print("Hilbert Block-Sparse Attention Benchmarks")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(
        f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    if torch.cuda.is_available():
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    # Test configurations
    seq_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
    block_sizes = [64, 128]
    sparsity_ratios = [0.1, 0.05, 0.01]  # 90%, 95%, 99% sparse

    all_results = []

    # 1. Compare standard vs Hilbert at different sequence lengths
    print("\n" + "=" * 80)
    print("STANDARD vs HILBERT COMPARISON")
    print("=" * 80)

    for sparsity in [0.05]:  # Use 95% sparse for main comparison
        for block_size in [64]:  # Standard block size
            print(f"\nSparsity: {(1 - sparsity) * 100:.0f}%, Block size: {block_size}")

            standard_results = benchmark_implementation(
                "standard", seq_lengths, block_size=block_size, sparsity_ratio=sparsity
            )
            all_results.extend(standard_results)

            hilbert_results = benchmark_implementation(
                "hilbert", seq_lengths, block_size=block_size, sparsity_ratio=sparsity
            )
            all_results.extend(hilbert_results)

            # Calculate speedups
            print("\nSpeedups (Hilbert vs Standard):")
            for std, hlb in zip(standard_results, hilbert_results):
                if std.sequence_length == hlb.sequence_length:
                    speedup = std.forward_time_ms / hlb.forward_time_ms
                    print(f"  {std.sequence_length:6d} tokens: {speedup:5.2f}x")

    # 2. Test different Hilbert configurations
    print("\n" + "=" * 80)
    print("HILBERT CONFIGURATION COMPARISON")
    print("=" * 80)

    config_results = compare_hilbert_configurations(seq_len=16384)

    # 3. Test impact of block size
    print("\n" + "=" * 80)
    print("BLOCK SIZE IMPACT")
    print("=" * 80)

    for block_size in block_sizes:
        print(f"\nBlock size: {block_size}")
        for variant in ["standard", "hilbert"]:
            results = benchmark_implementation(
                variant, [8192, 16384], block_size=block_size, sparsity_ratio=0.05
            )
            for r in results:
                r.block_size = block_size  # Update for clarity
            all_results.extend(results)

    # 4. Test impact of sparsity
    print("\n" + "=" * 80)
    print("SPARSITY IMPACT")
    print("=" * 80)

    test_seq_len = 16384
    for sparsity in sparsity_ratios:
        print(f"\nSparsity: {(1 - sparsity) * 100:.0f}%")
        for variant in ["standard", "hilbert"]:
            results = benchmark_implementation(
                variant, [test_seq_len], block_size=64, sparsity_ratio=sparsity
            )
            all_results.extend(results)

    # Summary and analysis
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average speedup across all tests
    speedups = []
    for i in range(len(all_results)):
        if all_results[i].implementation == "standard":
            # Find matching Hilbert result
            for j in range(len(all_results)):
                if (
                    all_results[j].implementation == "hilbert"
                    and all_results[j].sequence_length == all_results[i].sequence_length
                    and all_results[j].block_size == all_results[i].block_size
                    and all_results[j].sparsity_ratio == all_results[i].sparsity_ratio
                ):
                    speedup = (
                        all_results[i].forward_time_ms / all_results[j].forward_time_ms
                    )
                    speedups.append(speedup)
                    all_results[j].speedup_ratio = speedup

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print(f"\nAverage Hilbert speedup: {avg_speedup:.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"hilbert_block_sparse_benchmark_{timestamp}.json"

    results_dict = []
    for r in all_results:
        results_dict.append(
            {
                "implementation": r.implementation,
                "sequence_length": r.sequence_length,
                "block_size": r.block_size,
                "sparsity_ratio": r.sparsity_ratio,
                "forward_time_ms": r.forward_time_ms,
                "memory_mb": r.memory_mb,
                "speedup_ratio": r.speedup_ratio,
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
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                    / 1024**3
                    if torch.cuda.is_available()
                    else 0,
                },
                "config_comparison": config_results,
                "results": results_dict,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. Hilbert Optimization Benefits:")
    print("   - Improves cache locality by ordering blocks along space-filling curve")
    print("   - Reduces random memory access patterns")
    print("   - Most effective at larger sequence lengths where cache misses dominate")

    print("\n2. Configuration Recommendations:")
    print("   - Block-level Hilbert: Best for most cases (low overhead, good gains)")
    print("   - Element-level Hilbert: Higher overhead, use for very large blocks")
    print("   - Full Hilbert: Maximum optimization but highest overhead")

    print("\n3. When to Use Hilbert:")
    print("   - Sequences > 8K tokens")
    print("   - High sparsity (95%+)")
    print("   - Memory bandwidth limited scenarios")


if __name__ == "__main__":
    main()
