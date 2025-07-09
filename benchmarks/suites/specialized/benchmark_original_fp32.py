#!/usr/bin/env python3
"""
FP32 benchmark script for original DilatedAttention implementation.
"""

import torch
import time
import json
from datetime import datetime
from typing import Dict, List
import numpy as np

from dilated_attention_pytorch import DilatedAttention


def benchmark_config(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    device: torch.device,
    dtype: torch.dtype,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> Dict:
    """Benchmark a single configuration."""

    # Create model
    model = DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        attention_dropout=0.0,
    ).to(device)

    # Create inputs with specified dtype
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(q, k, v)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        mem_before = 0

    # Time the forward passes
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(q, k, v)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    # Get peak memory
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        mem_delta = peak_memory - mem_before
    else:
        peak_memory = 0
        mem_delta = 0

    times_ms = [t * 1000 for t in times]

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "dtype": str(dtype),
        "segments": len(segment_lengths),
        "mean_ms": round(np.mean(times_ms), 2),
        "std_ms": round(np.std(times_ms), 2),
        "peak_mem_mb": round(peak_memory, 2),
        "mem_delta_mb": round(mem_delta, 2),
        "tokens_per_sec": int((batch_size * seq_len) * 1000 / np.mean(times_ms)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print("\n=== BENCHMARKING DILATEDATTENTION: FP32 vs FP16 ===\n")

    # Test configurations
    configs = [
        # (batch_size, seq_len, num_heads, head_dim, segment_lengths, dilation_rates)
        (2, 1024, 8, 64, [256, 512], [1, 2]),
        (2, 2048, 8, 64, [512, 1024], [1, 2]),
        (2, 4096, 8, 64, [1024, 2048], [1, 2]),
        (1, 8192, 8, 64, [2048, 4096], [1, 2]),
        # Different configurations
        (2, 4096, 4, 64, [1024, 2048], [1, 2]),
        (2, 4096, 16, 64, [1024, 2048], [1, 2]),
    ]

    results_fp32 = []
    results_fp16 = []

    print("Testing FP32...")
    for (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        segment_lengths,
        dilation_rates,
    ) in configs:
        print(
            f"  seq_len={seq_len}, batch={batch_size}, heads={num_heads}... ",
            end="",
            flush=True,
        )

        try:
            result = benchmark_config(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                segment_lengths,
                dilation_rates,
                device,
                torch.float32,
            )
            results_fp32.append(result)
            print(f"✓ {result['mean_ms']:.1f}ms, {result['peak_mem_mb']:.0f}MB")

        except Exception as e:
            print(f"✗ Failed: {str(e)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTesting FP16...")
    for (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        segment_lengths,
        dilation_rates,
    ) in configs:
        print(
            f"  seq_len={seq_len}, batch={batch_size}, heads={num_heads}... ",
            end="",
            flush=True,
        )

        try:
            result = benchmark_config(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                segment_lengths,
                dilation_rates,
                device,
                torch.float16,
            )
            results_fp16.append(result)
            print(f"✓ {result['mean_ms']:.1f}ms, {result['peak_mem_mb']:.0f}MB")

        except Exception as e:
            print(f"✗ Failed: {str(e)}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output = {
        "implementation": "DilatedAttention (Original)",
        "timestamp": timestamp,
        "device": str(device),
        "results_fp32": results_fp32,
        "results_fp16": results_fp16,
    }

    filename = f"benchmarks/original_dilated_fp32_vs_fp16_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")

    # Print comparison table
    print("\n=== FP32 vs FP16 COMPARISON ===")
    print(
        f"{'Config':^30} | {'FP32 Time':^12} | {'FP32 Mem':^10} | {'FP16 Time':^12} | {'FP16 Mem':^10} | {'Speedup':^8}"
    )
    print("-" * 100)

    for fp32, fp16 in zip(results_fp32, results_fp16):
        if "error" not in fp32 and "error" not in fp16:
            config = (
                f"seq={fp32['seq_len']},b={fp32['batch_size']},h={fp32['num_heads']}"
            )
            speedup = fp32["mean_ms"] / fp16["mean_ms"]
            _ = (
                fp32["peak_mem_mb"] / fp16["peak_mem_mb"]
                if fp16["peak_mem_mb"] > 0
                else 0
            )

            print(
                f"{config:^30} | {fp32['mean_ms']:>10.1f}ms | {fp32['peak_mem_mb']:>8.0f}MB | "
                f"{fp16['mean_ms']:>10.1f}ms | {fp16['peak_mem_mb']:>8.0f}MB | {speedup:>6.2f}x"
            )

    # Summary statistics
    if results_fp32 and results_fp16:
        print("\n=== SUMMARY ===")

        # Average speedup
        speedups = [
            fp32["mean_ms"] / fp16["mean_ms"]
            for fp32, fp16 in zip(results_fp32, results_fp16)
            if "error" not in fp32 and "error" not in fp16
        ]
        if speedups:
            print(f"Average FP16 Speedup: {np.mean(speedups):.2f}x")

        # Memory savings
        mem_ratios = [
            fp32["peak_mem_mb"] / fp16["peak_mem_mb"]
            for fp32, fp16 in zip(results_fp32, results_fp16)
            if "error" not in fp32 and "error" not in fp16 and fp16["peak_mem_mb"] > 0
        ]
        if mem_ratios:
            print(f"Average Memory Reduction: {np.mean(mem_ratios):.2f}x")

        # Throughput comparison
        fp32_throughput = sum(
            r["tokens_per_sec"] for r in results_fp32 if "error" not in r
        )
        fp16_throughput = sum(
            r["tokens_per_sec"] for r in results_fp16 if "error" not in r
        )
        print("\nTotal Throughput:")
        print(f"  FP32: {fp32_throughput:,} tokens/sec")
        print(f"  FP16: {fp16_throughput:,} tokens/sec")
        print(f"  Improvement: {fp16_throughput / fp32_throughput:.2f}x")


if __name__ == "__main__":
    main()
