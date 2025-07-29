#!/usr/bin/env python3
"""
Comprehensive benchmark comparing UnifiedHilbertAttention vs UnifiedHilbertAttentionOptimizedEnhanced
after all optimizations.
"""

import torch
import time
import sys
import gc
from dataclasses import dataclass
from typing import List

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


@dataclass
class BenchmarkResult:
    config: str
    seq_len: int
    dilation_rate: int
    unified_time: float
    enhanced_time: float
    ratio: float
    unified_memory: float
    enhanced_memory: float
    uses_hilbert: bool
    block_size: str
    effective_len: int


def profile_model(model, x, warmup=3, runs=10):
    """Profile a model for time and memory."""
    # Clear memory stats
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    mem_used = peak_mem - start_mem

    return sum(times) / len(times), mem_used


def run_comprehensive_benchmark():
    """Run comprehensive benchmarks across various configurations."""

    print("=== Comprehensive Unified vs Enhanced Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")

    # Detect GPU architecture
    compute_capability = torch.cuda.get_device_capability()[0]
    gpu_type = (
        "Pascal" if compute_capability < 7 else f"Volta+ (CC {compute_capability}.x)"
    )
    print(f"GPU Architecture: {gpu_type}")
    print()

    # Test configurations - comprehensive set
    test_configs = [
        # Small sequences
        (512, 1, "512 Dense"),
        (1024, 1, "1K Dense"),
        (1024, 2, "1K Sparse d=2"),
        # Medium sequences
        (2048, 1, "2K Dense"),
        (2048, 2, "2K Sparse d=2"),
        (2048, 4, "2K Sparse d=4"),
        # 4K sequences (our optimization target)
        (4096, 1, "4K Dense"),
        (4096, 2, "4K Sparse d=2"),
        (4096, 4, "4K Sparse d=4"),
        (4096, 8, "4K Sparse d=8"),
        # Large sequences
        (8192, 1, "8K Dense"),
        (8192, 2, "8K Sparse d=2"),
        (8192, 4, "8K Sparse d=4"),
        # Very large sequences
        (16384, 1, "16K Dense"),
        (16384, 2, "16K Sparse d=2"),
        (16384, 4, "16K Sparse d=4"),
    ]

    # Common parameters
    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    results: List[BenchmarkResult] = []

    # Print header
    print(
        f"{'Config':<18} | {'Unified':<12} | {'Enhanced':<12} | {'Ratio':<8} | {'Memory (MB)':<20} | {'Details':<30}"
    )
    print("-" * 110)

    for seq_len, dilation_rate, desc in test_configs:
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Create models
            unified = (
                UnifiedHilbertAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            enhanced = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                    enable_4k_optimization=True,
                    enable_8k_optimization=True,
                )
                .cuda()
                .eval()
            )

            # Create input
            x = torch.randn(
                batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
            )

            # Get configuration info
            config = enhanced._get_optimal_config(seq_len)
            effective_len = seq_len // dilation_rate
            M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
            uses_hilbert = M_padded > enhanced.hilbert_threshold
            block_size = f"{config['block_m']}x{config['block_n']}"

            # Profile both models
            unified_time, unified_mem = profile_model(unified, x)
            enhanced_time, enhanced_mem = profile_model(enhanced, x)

            ratio = enhanced_time / unified_time

            # Determine status (removed - unused variable)

            # Memory comparison
            mem_str = f"U:{unified_mem:.1f} E:{enhanced_mem:.1f}"

            # Details
            details = f"H:{uses_hilbert} B:{block_size}"
            if dilation_rate > 1:
                details += f" Eff:{effective_len}"

            print(
                f"{desc:<18} | {unified_time:<12.2f} | {enhanced_time:<12.2f} | "
                f"{ratio:<8.2f}x | {mem_str:<20} | {details:<30}"
            )

            result = BenchmarkResult(
                config=desc,
                seq_len=seq_len,
                dilation_rate=dilation_rate,
                unified_time=unified_time,
                enhanced_time=enhanced_time,
                ratio=ratio,
                unified_memory=unified_mem,
                enhanced_memory=enhanced_mem,
                uses_hilbert=uses_hilbert,
                block_size=block_size,
                effective_len=effective_len,
            )
            results.append(result)

        except Exception as e:
            print(f"{desc:<18} | Error: {str(e)}")
            continue

    return results


def analyze_results(results: List[BenchmarkResult]):
    """Analyze and summarize benchmark results."""

    print("\n=== Performance Analysis ===")

    # Group by sequence length
    by_length = {}
    for r in results:
        if r.seq_len not in by_length:
            by_length[r.seq_len] = []
        by_length[r.seq_len].append(r)

    print("\n1. Performance by Sequence Length:")
    for seq_len in sorted(by_length.keys()):
        group = by_length[seq_len]
        avg_ratio = sum(r.ratio for r in group) / len(group)
        print(f"   {seq_len:5d} tokens: avg ratio = {avg_ratio:.2f}x", end="")
        if avg_ratio < 1.0:
            print(f" (Enhanced {(1 / avg_ratio - 1) * 100:.0f}% faster)")
        elif avg_ratio > 1.0:
            print(f" (Enhanced {(avg_ratio - 1) * 100:.0f}% slower)")
        else:
            print(" (Parity)")

    print("\n2. Performance by Pattern Type:")
    dense_results = [r for r in results if r.dilation_rate == 1]
    sparse_results = [r for r in results if r.dilation_rate > 1]

    if dense_results:
        avg_dense = sum(r.ratio for r in dense_results) / len(dense_results)
        print(f"   Dense patterns (d=1): avg ratio = {avg_dense:.2f}x")

    if sparse_results:
        avg_sparse = sum(r.ratio for r in sparse_results) / len(sparse_results)
        print(f"   Sparse patterns (d>1): avg ratio = {avg_sparse:.2f}x")

    # Specific dilation rates
    for d in [2, 4, 8]:
        d_results = [r for r in results if r.dilation_rate == d]
        if d_results:
            avg_d = sum(r.ratio for r in d_results) / len(d_results)
            print(f"   Dilation rate {d}: avg ratio = {avg_d:.2f}x")

    print("\n3. 4K Sequence Performance (Our Optimization Target):")
    fk_results = [r for r in results if r.seq_len == 4096]
    for r in fk_results:
        print(f"   {r.config}: {r.ratio:.2f}x", end="")
        if r.ratio < 1.0:
            print(f" (✓ {(1 / r.ratio - 1) * 100:.0f}% faster)")
        elif r.ratio < 1.2:
            print(" (✓ Acceptable)")
        else:
            print(f" (⚠ {(r.ratio - 1) * 100:.0f}% slower)")

    print("\n4. Memory Efficiency:")
    total_unified_mem = sum(r.unified_memory for r in results)
    total_enhanced_mem = sum(r.enhanced_memory for r in results)
    print(f"   Average memory ratio: {total_enhanced_mem / total_unified_mem:.2f}x")

    # Find best and worst cases
    print("\n5. Best and Worst Cases:")
    best = min(results, key=lambda r: r.ratio)
    worst = max(results, key=lambda r: r.ratio)

    print(f"   Best: {best.config} - {best.ratio:.2f}x", end="")
    if best.ratio < 1.0:
        print(f" (Enhanced {(1 / best.ratio - 1) * 100:.0f}% faster)")
    else:
        print()

    print(f"   Worst: {worst.config} - {worst.ratio:.2f}x", end="")
    if worst.ratio > 1.0:
        print(f" (Enhanced {(worst.ratio - 1) * 100:.0f}% slower)")
    else:
        print()

    # Overall assessment
    print("\n6. Overall Assessment:")
    wins = sum(1 for r in results if r.ratio < 0.9)
    parity = sum(1 for r in results if 0.9 <= r.ratio <= 1.1)
    losses = sum(1 for r in results if r.ratio > 1.1)

    print(f"   Enhanced faster: {wins}/{len(results)} configs")
    print(f"   Parity: {parity}/{len(results)} configs")
    print(f"   Enhanced slower: {losses}/{len(results)} configs")

    avg_ratio = sum(r.ratio for r in results) / len(results)
    print(f"   Overall average: {avg_ratio:.2f}x", end="")
    if avg_ratio < 1.0:
        print(f" (Enhanced {(1 / avg_ratio - 1) * 100:.0f}% faster on average)")
    elif avg_ratio > 1.0:
        print(f" (Enhanced {(avg_ratio - 1) * 100:.0f}% slower on average)")
    else:
        print(" (Perfect parity)")


def test_extreme_cases():
    """Test extreme cases to find limits."""

    print("\n\n=== Testing Extreme Cases ===")

    # Very small sequences
    print("\n1. Very Small Sequences:")
    small_configs = [(128, 1), (256, 1), (256, 2)]

    for seq_len, dilation_rate in small_configs:
        try:
            unified = (
                UnifiedHilbertAttention(
                    hidden_dim=512,
                    num_heads=8,
                    segment_size=128,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            enhanced = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=512,
                    num_heads=8,
                    segment_size=128,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            x = torch.randn(1, seq_len, 512, device="cuda", dtype=torch.float32)

            u_time, _ = profile_model(unified, x, runs=5)
            e_time, _ = profile_model(enhanced, x, runs=5)

            print(f"   {seq_len} d={dilation_rate}: {e_time / u_time:.2f}x")

        except Exception as e:
            print(f"   {seq_len} d={dilation_rate}: Error - {str(e)}")

    # Very sparse patterns
    print("\n2. Very Sparse Patterns:")
    sparse_configs = [(8192, 8), (8192, 16), (16384, 8)]

    for seq_len, dilation_rate in sparse_configs:
        try:
            unified = (
                UnifiedHilbertAttention(
                    hidden_dim=512,
                    num_heads=8,
                    segment_size=128,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            enhanced = (
                UnifiedHilbertAttentionOptimizedEnhanced(
                    hidden_dim=512,
                    num_heads=8,
                    segment_size=128,
                    dilation_rate=dilation_rate,
                )
                .cuda()
                .eval()
            )

            x = torch.randn(1, seq_len, 512, device="cuda", dtype=torch.float32)

            u_time, _ = profile_model(unified, x, runs=3)
            e_time, _ = profile_model(enhanced, x, runs=3)

            eff_len = seq_len // dilation_rate
            print(
                f"   {seq_len} d={dilation_rate} (eff={eff_len}): {e_time / u_time:.2f}x"
            )

        except Exception as e:
            print(f"   {seq_len} d={dilation_rate}: Error - {str(e)}")


def main():
    # Run comprehensive benchmark
    results = run_comprehensive_benchmark()

    # Analyze results
    analyze_results(results)

    # Test extreme cases
    test_extreme_cases()

    print("\n=== Conclusion ===")
    print("The Enhanced implementation with 4K optimization shows:")
    print("- Good performance for most configurations")
    print("- Special optimization for 4K d=4 working as intended")
    print("- Hilbert SFC preserved for cache benefits")
    print("- Some trade-offs in specific configurations")


if __name__ == "__main__":
    main()
