#!/usr/bin/env python3
"""
Benchmark script for original DilatedAttention implementation.
"""

import torch
import time
import json
from datetime import datetime
from typing import Dict, List
import numpy as np

from dilated_attention_pytorch import DilatedAttention


def get_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def benchmark_dilated_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: List[int],
    dilation_rates: List[int],
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict:
    """Benchmark a single configuration of DilatedAttention."""

    # Create model
    model = DilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        attention_dropout=0.0,
    ).to(device)

    # Create inputs
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Measure memory before
    mem_before = get_memory_usage()

    # Time the forward passes
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(q, k, v)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        times.append(end - start)

    # Measure memory after
    mem_after = get_memory_usage()

    # Calculate statistics
    times_ms = [t * 1000 for t in times]

    return {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
            "total_params": sum(p.numel() for p in model.parameters()),
        },
        "timing": {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "median_ms": np.median(times_ms),
        },
        "memory": {
            "before_mb": mem_before,
            "after_mb": mem_after,
            "delta_mb": mem_after - mem_before,
        },
        "throughput": {
            "sequences_per_second": 1000 / np.mean(times_ms),
            "tokens_per_second": (batch_size * seq_len) * 1000 / np.mean(times_ms),
        },
    }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")

    results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "device": str(device),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "implementation": "DilatedAttention (Original)",
        },
        "benchmarks": [],
    }

    # Test configurations
    configs = [
        # (batch_size, seq_len, num_heads, head_dim, segment_lengths, dilation_rates)
        # Small sequences
        (2, 1024, 8, 64, [256, 512], [1, 2]),
        (2, 2048, 8, 64, [512, 1024], [1, 2]),
        (2, 4096, 8, 64, [1024, 2048], [1, 2]),
        # Medium sequences
        (2, 8192, 8, 64, [2048, 4096], [1, 2]),
        (2, 16384, 8, 64, [4096, 8192], [1, 2]),
        # Large sequences (if memory allows)
        (1, 32768, 8, 64, [8192, 16384], [1, 2]),
        (1, 65536, 8, 64, [16384, 32768], [1, 2]),
        # Different head configurations
        (2, 4096, 4, 64, [1024, 2048], [1, 2]),
        (2, 4096, 16, 64, [1024, 2048], [1, 2]),
        (2, 4096, 32, 32, [1024, 2048], [1, 2]),
        # Different segment/dilation configurations
        (2, 4096, 8, 64, [512, 1024, 2048], [1, 2, 4]),
        (2, 4096, 8, 64, [256, 512, 1024, 2048], [1, 2, 4, 8]),
        # Extreme dilation
        (2, 8192, 8, 64, [512, 1024, 2048, 4096], [1, 4, 16, 64]),
    ]

    print(f"\nRunning {len(configs)} benchmark configurations...")

    for i, (
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        segment_lengths,
        dilation_rates,
    ) in enumerate(configs):
        print(
            f"\n[{i + 1}/{len(configs)}] Config: batch={batch_size}, seq_len={seq_len}, "
            f"heads={num_heads}, dim={head_dim}, segments={segment_lengths}, dilations={dilation_rates}"
        )

        try:
            result = benchmark_dilated_attention(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                segment_lengths,
                dilation_rates,
                device,
            )
            results["benchmarks"].append(result)

            print(
                f"  ✓ Time: {result['timing']['mean_ms']:.2f}±{result['timing']['std_ms']:.2f} ms"
            )
            print(f"  ✓ Memory: {result['memory']['delta_mb']:.2f} MB")
            print(
                f"  ✓ Throughput: {result['throughput']['tokens_per_second']:.0f} tokens/sec"
            )

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            results["benchmarks"].append(
                {
                    "config": {
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "num_heads": num_heads,
                        "head_dim": head_dim,
                        "segment_lengths": segment_lengths,
                        "dilation_rates": dilation_rates,
                    },
                    "error": str(e),
                }
            )

        # Clear cache between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"benchmarks/original_dilated_attention_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {filename}")

    # Print summary
    successful_runs = [r for r in results["benchmarks"] if "error" not in r]
    if successful_runs:
        print("\n=== SUMMARY ===")
        print(f"Successful runs: {len(successful_runs)}/{len(results['benchmarks'])}")

        # Find best/worst performers
        by_throughput = sorted(
            successful_runs,
            key=lambda x: x["throughput"]["tokens_per_second"],
            reverse=True,
        )

        print("\nTop 3 configurations by throughput:")
        for i, r in enumerate(by_throughput[:3]):
            c = r["config"]
            print(
                f"  {i + 1}. seq_len={c['seq_len']}, heads={c['num_heads']}: "
                f"{r['throughput']['tokens_per_second']:.0f} tokens/sec"
            )

        print("\nBottom 3 configurations by throughput:")
        for i, r in enumerate(by_throughput[-3:]):
            c = r["config"]
            print(
                f"  {i + 1}. seq_len={c['seq_len']}, heads={c['num_heads']}: "
                f"{r['throughput']['tokens_per_second']:.0f} tokens/sec"
            )

        # Memory usage
        by_memory = sorted(
            successful_runs, key=lambda x: x["memory"]["delta_mb"], reverse=True
        )
        print("\nHighest memory usage:")
        for i, r in enumerate(by_memory[:3]):
            c = r["config"]
            print(
                f"  {i + 1}. seq_len={c['seq_len']}, heads={c['num_heads']}: "
                f"{r['memory']['delta_mb']:.2f} MB"
            )


if __name__ == "__main__":
    run_benchmarks()
