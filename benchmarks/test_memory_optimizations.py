#!/usr/bin/env python3
"""
Test memory optimizations for Hilbert Attention kernels.
"""

import torch
import torch.nn as nn
import time
import gc
import numpy as np
from typing import Dict

# Import kernels
from dilated_attention_pytorch.kernels import (
    HilbertAttentionCore,
    UnifiedHilbertAttention,
)


def measure_memory_and_time(
    module: nn.Module, x: torch.Tensor, num_warmup: int = 3, num_runs: int = 10
) -> Dict[str, float]:
    """Measure memory usage and runtime."""
    device = x.device

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    # Measure peak memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = module(x)

    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1e6  # MB
    else:
        peak_memory = 0

    # Measure runtime
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "memory_mb": peak_memory,
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    }


def test_memory_optimizations():
    """Test different memory optimization levels."""
    print("=== Memory Optimization Benchmark ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {total_memory:.1f} GB\n")

    # Test configurations
    configs = [
        # (batch_size, seq_len, hidden_dim, num_heads)
        (2, 512, 768, 12),
        (2, 1024, 768, 12),
        (2, 2048, 768, 12),
        (1, 4096, 768, 12),
    ]

    segment_size = 128
    dilation_rate = 2

    for batch_size, seq_len, hidden_dim, num_heads in configs:
        print(
            f"\nConfiguration: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}, heads={num_heads}"
        )
        print("-" * 80)

        # Skip if too large
        estimated_memory = (batch_size * seq_len * hidden_dim * 4 * 6) / 1e9
        if estimated_memory > total_memory * 0.7:
            print(
                f"SKIPPED - Estimated {estimated_memory:.1f} GB exceeds available memory"
            )
            continue

        # Create input
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device=device, dtype=torch.float32
        )

        # Test original kernel
        try:
            print("\n1. Original HilbertAttentionCore:")
            core = HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                use_custom_backward=False,
            ).to(device)

            results_core = measure_memory_and_time(core, x)
            print(f"   Memory: {results_core['memory_mb']:.1f} MB")
            print(
                f"   Time: {results_core['time_mean']:.2f} ± {results_core['time_std']:.2f} ms"
            )

        except Exception as e:
            print(f"   FAILED: {str(e)}")
            results_core = {"memory_mb": float("inf"), "time_mean": float("inf")}

        # Test memory-optimized versions
        optimization_levels = [
            (0, "No optimization"),
            (1, "Moderate optimization"),
            (2, "Aggressive optimization"),
        ]

        for level, desc in optimization_levels:
            try:
                print(f"\n2. Memory-Optimized (Level {level} - {desc}):")
                # Map optimization levels to memory modes
                memory_mode_map = {0: "standard", 1: "optimized", 2: "aggressive"}
                mem_opt = UnifiedHilbertAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    segment_size=segment_size,
                    dilation_rate=dilation_rate,
                    memory_mode=memory_mode_map[level],
                ).to(device)

                results_opt = measure_memory_and_time(mem_opt, x)
                print(f"   Memory: {results_opt['memory_mb']:.1f} MB")
                print(
                    f"   Time: {results_opt['time_mean']:.2f} ± {results_opt['time_std']:.2f} ms"
                )

                # Calculate improvements
                if results_core["memory_mb"] != float("inf"):
                    mem_reduction = (
                        (results_core["memory_mb"] - results_opt["memory_mb"])
                        / results_core["memory_mb"]
                        * 100
                    )
                    speedup = results_core["time_mean"] / results_opt["time_mean"]
                    print(f"   Memory reduction: {mem_reduction:.1f}%")
                    print(f"   Speed: {speedup:.2f}x")

            except Exception as e:
                print(f"   FAILED: {str(e)}")

        # Force cleanup between configs
        del x
        if "core" in locals():
            del core
        if "mem_opt" in locals():
            del mem_opt
        torch.cuda.empty_cache()
        gc.collect()


def test_dilated_access_pattern():
    """Test memory access pattern optimization for dilated attention."""
    print("\n\n=== Dilated Access Pattern Optimization ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    # Test configuration
    batch_size = 2
    seq_len = 1024
    hidden_dim = 768
    num_heads = 12
    segment_size = 128

    print("Testing different dilation rates:")
    print("-" * 60)
    print("Dilation | Original (MB) | Optimized (MB) | Reduction")
    print("-" * 60)

    for dilation_rate in [1, 2, 4, 8]:
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Original
        core = HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        ).to(device)

        results_core = measure_memory_and_time(core, x)

        # Optimized
        mem_opt = UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            memory_mode="optimized",
        ).to(device)

        results_opt = measure_memory_and_time(mem_opt, x)

        reduction = (
            (results_core["memory_mb"] - results_opt["memory_mb"])
            / results_core["memory_mb"]
            * 100
        )

        print(
            f"{dilation_rate:8d} | {results_core['memory_mb']:13.1f} | {results_opt['memory_mb']:14.1f} | {reduction:8.1f}%"
        )

        del x, core, mem_opt
        torch.cuda.empty_cache()
        gc.collect()


def test_block_size_impact():
    """Test impact of different block sizes on memory usage."""
    print("\n\n=== Block Size Impact on Memory ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")

    # Test configuration
    batch_size = 2
    seq_len = 1024
    hidden_dim = 768
    num_heads = 12

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    print("Testing memory usage with different optimization levels:")
    print("-" * 50)

    # Get block sizes for each level
    for level in [0, 1, 2]:
        memory_mode_map = {0: "standard", 1: "optimized", 2: "aggressive"}
        module = UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=2,
            memory_mode=memory_mode_map[level],
        ).to(device)

        # Get config to check block sizes
        config = module.get_config()
        block_sizes = f"memory_mode={config['memory_mode']}"
        results = measure_memory_and_time(module, x)

        print(f"\nLevel {level}: Block sizes = {block_sizes}")
        print(f"  Memory: {results['memory_mb']:.1f} MB")
        print(f"  Time: {results['time_mean']:.2f} ms")

        del module
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=" * 80)
    print("Memory Optimization Tests for Hilbert Attention")
    print("=" * 80)

    test_memory_optimizations()
    test_dilated_access_pattern()
    test_block_size_impact()

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)
