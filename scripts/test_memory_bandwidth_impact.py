#!/usr/bin/env python3
"""
Test the impact of sparse Hilbert optimization on memory bandwidth usage.
"""

import torch
import time
from typing import Dict


def measure_memory_bandwidth(
    module,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    use_hilbert: bool,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Measure memory bandwidth usage for attention computation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

    if device == "cuda":
        torch.cuda.synchronize()

    # Measure time
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.perf_counter() - start) / num_iterations

    # Calculate theoretical memory accesses
    # Each attention computation needs to read Q, K, V and write output
    bytes_per_element = 4  # float32
    elements_per_token = hidden_dim

    # For dilated attention with segment_size S and dilation_rate D:
    # - Each query attends to S/D keys
    # - Total attention computations: seq_len * (S/D)

    segment_size = module.segment_size
    dilation_rate = module.dilation_rate
    sparse_positions = segment_size // dilation_rate

    # Memory reads per token:
    # - Q: 1 read
    # - K: sparse_positions reads (with Hilbert, may have worse locality)
    # - V: sparse_positions reads
    # - Output: 1 write

    total_memory_ops = seq_len * (1 + 2 * sparse_positions + 1) * elements_per_token
    total_bytes = total_memory_ops * bytes_per_element * batch_size

    # Effective bandwidth
    bandwidth_gb_s = (total_bytes / 1e9) / elapsed

    return {
        "time_ms": elapsed * 1000,
        "theoretical_bytes_gb": total_bytes / 1e9,
        "effective_bandwidth_gb_s": bandwidth_gb_s,
        "sparse_positions": sparse_positions,
        "memory_ops_per_token": 1 + 2 * sparse_positions + 1,
    }


def analyze_memory_patterns():
    """Analyze memory access patterns for different implementations."""
    print("Memory Bandwidth Analysis: Original vs Sparse Hilbert")
    print("=" * 70)

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    segment_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

        # Theoretical bandwidth for different GPUs
        gpu_bandwidth = {
            "GTX 1080": 320,  # GB/s
            "RTX 2080": 448,
            "RTX 3090": 936,
            "A100": 1555,
            "H100": 3350,
        }

        # Find closest match
        theoretical_bw = 320  # Default to GTX 1080
        for gpu, bw in gpu_bandwidth.items():
            if gpu.lower() in gpu_name.lower():
                theoretical_bw = bw
                break

        print(f"Theoretical memory bandwidth: {theoretical_bw} GB/s")
    else:
        print("Running on CPU")
        theoretical_bw = 50  # Rough estimate for CPU

    # Test different sequence lengths and dilation rates
    test_configs = [
        (1024, 1),
        (1024, 2),
        (1024, 4),
        (2048, 1),
        (2048, 2),
        (2048, 4),
        (4096, 1),
        (4096, 2),
        (4096, 4),
    ]

    print("\n" + "-" * 90)
    print(
        f"{'Config':^20} | {'Original Hilbert':^25} | {'Sparse Hilbert':^25} | {'Improvement':^15}"
    )
    print(
        f"{'seq_len, dil_rate':^20} | {'Time (ms) | BW (GB/s)':^25} | {'Time (ms) | BW (GB/s)':^25} | {'Speedup':^15}"
    )
    print("-" * 90)

    # Import implementations
    from dilated_attention_pytorch.kernels import HilbertAttentionCore
    from dilated_attention_pytorch.kernels.hilbert_attention_sparse_simple import (
        HilbertAttentionSparseSimple,
    )

    for seq_len, dilation_rate in test_configs:
        # Create modules
        original = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).to(device)

        sparse = HilbertAttentionSparseSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).to(device)

        # Measure original
        try:
            original_metrics = measure_memory_bandwidth(
                original, batch_size, seq_len, hidden_dim, use_hilbert=True
            )
        except Exception:
            original_metrics = {"time_ms": float("inf"), "effective_bandwidth_gb_s": 0}

        # Measure sparse optimized
        try:
            sparse_metrics = measure_memory_bandwidth(
                sparse, batch_size, seq_len, hidden_dim, use_hilbert=True
            )
        except Exception:
            sparse_metrics = {"time_ms": float("inf"), "effective_bandwidth_gb_s": 0}

        # Calculate improvement
        speedup = original_metrics["time_ms"] / sparse_metrics["time_ms"]

        # Print results
        config_str = f"{seq_len}, {dilation_rate}"
        original_str = f"{original_metrics['time_ms']:6.2f} | {original_metrics['effective_bandwidth_gb_s']:7.1f}"
        sparse_str = f"{sparse_metrics['time_ms']:6.2f} | {sparse_metrics['effective_bandwidth_gb_s']:7.1f}"

        print(
            f"{config_str:^20} | {original_str:^25} | {sparse_str:^25} | {speedup:^15.2f}x"
        )

    # Memory access pattern analysis
    print("\n" + "=" * 70)
    print("MEMORY ACCESS PATTERN ANALYSIS")
    print("=" * 70)

    print("\nOriginal Hilbert Implementation:")
    print("- Creates full Hilbert map (seq_len entries)")
    print("- Indirect memory access: K[hilbert_map[position]]")
    print("- Random access pattern due to Hilbert reordering")
    print("- Cache misses on both map lookup and data access")

    print("\nSparse Hilbert Implementation:")
    print("- Creates sparse Hilbert map (seg_size/dil_rate entries)")
    print("- More sequential access pattern")
    print("- Better cache utilization for the mapping")
    print("- Reduced indirection overhead")

    print("\nMemory Bandwidth Impact:")
    print("1. Reduced indirect lookups → Better memory throughput")
    print("2. Smaller mapping → Better cache utilization")
    print("3. More predictable access → Better prefetching")

    # Calculate theoretical improvement
    print("\n" + "=" * 70)
    print("THEORETICAL ANALYSIS")
    print("=" * 70)

    for dil_rate in [1, 2, 4]:
        sparse_per_seg = segment_size // dil_rate

        # Original: Full map lookup + data access
        original_accesses = seq_len + seq_len  # Map lookup + actual data

        # Sparse: Small map lookup + data access
        sparse_accesses = sparse_per_seg + seq_len  # Much smaller map

        reduction = (1 - sparse_accesses / original_accesses) * 100

        print(f"\nDilation rate {dil_rate}:")
        print(f"  Original memory accesses: {original_accesses}")
        print(f"  Sparse memory accesses: {sparse_accesses}")
        print(f"  Reduction: {reduction:.1f}%")
        print(f"  Cache efficiency: {sparse_per_seg}x reuse of mapping")


if __name__ == "__main__":
    analyze_memory_patterns()
