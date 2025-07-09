#!/usr/bin/env python3
"""
Quick benchmark script for original DilatedAttention implementation.
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

    # Create inputs with float16 for efficiency
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
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

    print("\n=== BENCHMARKING ORIGINAL DILATEDATTENTION ===\n")

    # Test configurations - focusing on realistic use cases
    configs = [
        # (batch_size, seq_len, num_heads, head_dim, segment_lengths, dilation_rates)
        (2, 1024, 8, 64, [256, 512], [1, 2]),
        (2, 2048, 8, 64, [512, 1024], [1, 2]),
        (2, 4096, 8, 64, [1024, 2048], [1, 2]),
        (1, 8192, 8, 64, [2048, 4096], [1, 2]),
        (1, 16384, 8, 64, [4096, 8192], [1, 2]),
        # Different head counts
        (2, 4096, 4, 64, [1024, 2048], [1, 2]),
        (2, 4096, 16, 64, [1024, 2048], [1, 2]),
        # More segments
        (2, 4096, 8, 64, [512, 1024, 2048], [1, 2, 4]),
        # Extreme dilation
        (1, 8192, 8, 64, [512, 1024, 2048, 4096], [1, 4, 16, 64]),
    ]

    results = []

    for (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        segment_lengths,
        dilation_rates,
    ) in configs:
        print(
            f"Testing seq_len={seq_len}, batch={batch_size}, heads={num_heads}... ",
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
            )
            results.append(result)
            print(f"✓ {result['mean_ms']:.1f}ms, {result['peak_mem_mb']:.0f}MB")

        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            results.append(
                {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "num_heads": num_heads,
                    "error": str(e),
                }
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output = {
        "implementation": "DilatedAttention (Original)",
        "timestamp": timestamp,
        "device": str(device),
        "results": results,
    }

    filename = f"benchmarks/original_dilated_quick_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")

    # Print summary table
    print("\n=== PERFORMANCE SUMMARY ===")
    print(
        f"{'Seq Len':>8} {'Batch':>6} {'Heads':>6} {'Time (ms)':>10} {'Memory (MB)':>12} {'Tokens/sec':>12}"
    )
    print("-" * 70)

    for r in results:
        if "error" not in r:
            print(
                f"{r['seq_len']:>8} {r['batch_size']:>6} {r['num_heads']:>6} "
                f"{r['mean_ms']:>10.1f} {r['peak_mem_mb']:>12.1f} "
                f"{r['tokens_per_sec']:>12,}"
            )


if __name__ == "__main__":
    main()
