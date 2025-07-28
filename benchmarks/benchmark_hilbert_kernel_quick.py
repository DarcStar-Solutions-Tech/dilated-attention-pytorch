#!/usr/bin/env python3
"""
Quick benchmark focused on Hilbert kernel performance in dilated attention context.
"""

import time
import torch
import numpy as np
from typing import Dict

# Import the Hilbert kernel
from dilated_attention_pytorch.kernels import HilbertAttentionCore


def create_dilated_mask(
    seq_len: int, segment_size: int, dilation_rate: int
) -> torch.Tensor:
    """Create a dilated attention mask."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    for i in range(seq_len):
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        # Within segment, only attend to dilated positions
        for j in range(seg_start, seg_end):
            if (j - seg_start) % dilation_rate == 0:
                mask[i, j] = True

    return mask


def benchmark_kernel(
    hidden_dim: int,
    num_heads: int,
    seq_len: int,
    segment_size: int,
    dilation_rate: int,
    batch_size: int = 2,
    use_hilbert: bool = True,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark a single configuration."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module
    module = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time forward pass
    times = []
    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

        if device == "cuda":
            torch.cuda.synchronize()

        times.append((time.perf_counter() - start) * 1000)

    # Memory usage
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        memory_mb = 0

    # Calculate attention density
    mask = create_dilated_mask(seq_len, segment_size, dilation_rate)
    attention_density = mask.sum().item() / (seq_len * seq_len)

    return {
        "mean_time_ms": np.mean(times),
        "std_time_ms": np.std(times),
        "memory_mb": memory_mb,
        "attention_density": attention_density,
        "effective_computations": mask.sum().item(),
        "throughput_tokens_sec": (batch_size * seq_len) / (np.mean(times) / 1000),
    }


def main():
    print("=" * 80)
    print("Hilbert Kernel Performance in Dilated Attention Context")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {compute_cap}")
    else:
        print("Running on CPU")

    # Fixed parameters
    hidden_dim = 768
    num_heads = 12
    batch_size = 2

    # Test configurations
    configs = [
        # (seq_len, segment_size, dilation_rate)
        (1024, 256, 1),  # Dense within segments
        (1024, 256, 2),  # 50% sparse
        (1024, 256, 4),  # 75% sparse
        (2048, 512, 1),  # Larger segments
        (2048, 512, 2),
        (2048, 512, 4),
        (4096, 1024, 1),  # Even larger
        (4096, 1024, 2),
        (4096, 1024, 4),
    ]

    print(
        f"\nTesting with hidden_dim={hidden_dim}, num_heads={num_heads}, batch_size={batch_size}"
    )
    print("-" * 80)
    print(
        f"{'Config':^30} | {'Hilbert OFF':^20} | {'Hilbert ON':^20} | {'Speedup':^10} | {'Density':^10}"
    )
    print("-" * 80)

    results = []

    for seq_len, segment_size, dilation_rate in configs:
        config_str = f"seq={seq_len}, seg={segment_size}, dil={dilation_rate}"

        # Benchmark without Hilbert
        try:
            metrics_off = benchmark_kernel(
                hidden_dim,
                num_heads,
                seq_len,
                segment_size,
                dilation_rate,
                batch_size,
                use_hilbert=False,
                num_iterations=10,
            )
        except Exception as e:
            print(f"{config_str:30} | {'FAILED':^20} | {str(e)}")
            continue

        # Benchmark with Hilbert
        try:
            metrics_on = benchmark_kernel(
                hidden_dim,
                num_heads,
                seq_len,
                segment_size,
                dilation_rate,
                batch_size,
                use_hilbert=True,
                num_iterations=10,
            )
        except Exception as e:
            print(
                f"{config_str:30} | {metrics_off['mean_time_ms']:^20.2f} | {'FAILED':^20} | {str(e)}"
            )
            continue

        # Calculate speedup
        speedup = metrics_off["mean_time_ms"] / metrics_on["mean_time_ms"]

        # Print results
        print(
            f"{config_str:30} | "
            f"{metrics_off['mean_time_ms']:8.2f}ms "
            f"({metrics_off['memory_mb']:6.1f}MB) | "
            f"{metrics_on['mean_time_ms']:8.2f}ms "
            f"({metrics_on['memory_mb']:6.1f}MB) | "
            f"{speedup:10.2f}x | "
            f"{metrics_on['attention_density']:10.1%}"
        )

        results.append(
            {
                "seq_len": seq_len,
                "segment_size": segment_size,
                "dilation_rate": dilation_rate,
                "metrics_off": metrics_off,
                "metrics_on": metrics_on,
                "speedup": speedup,
            }
        )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group by dilation rate
    print("\nPerformance by Dilation Rate:")
    print("-" * 40)

    for dil_rate in [1, 2, 4]:
        dil_results = [r for r in results if r["dilation_rate"] == dil_rate]
        if dil_results:
            avg_speedup = np.mean([r["speedup"] for r in dil_results])
            print(f"  Dilation rate {dil_rate}: {avg_speedup:.2f}x average speedup")

    # Memory efficiency
    print("\nMemory Efficiency:")
    print("-" * 40)

    for r in results:
        mem_reduction = (
            (r["metrics_off"]["memory_mb"] - r["metrics_on"]["memory_mb"])
            / r["metrics_off"]["memory_mb"]
            * 100
        )
        if mem_reduction > 0:
            print(
                f"  seq={r['seq_len']}, dil={r['dilation_rate']}: {mem_reduction:.1f}% memory reduction"
            )

    # Computational savings
    print("\nComputational Savings from Dilated Attention:")
    print("-" * 40)

    for r in results:
        full_computations = r["seq_len"] * r["seq_len"]
        actual_computations = r["metrics_on"]["effective_computations"]
        savings = (1 - actual_computations / full_computations) * 100
        print(
            f"  seq={r['seq_len']}, seg={r['segment_size']}, dil={r['dilation_rate']}: "
            f"{savings:.1f}% computation reduction "
            f"({actual_computations:,} vs {full_computations:,} ops)"
        )

    # Best configurations
    print("\nOptimal Configurations:")
    print("-" * 40)

    if results:
        # Best overall speedup
        best_speedup = max(results, key=lambda x: x["speedup"])
        print(
            f"  Best speedup: seq={best_speedup['seq_len']}, "
            f"seg={best_speedup['segment_size']}, "
            f"dil={best_speedup['dilation_rate']} "
            f"({best_speedup['speedup']:.2f}x)"
        )

        # Best for each sequence length
        for seq_len in sorted(set(r["seq_len"] for r in results)):
            seq_results = [r for r in results if r["seq_len"] == seq_len]
            if seq_results:
                best = max(seq_results, key=lambda x: x["speedup"])
                print(
                    f"  Best for seq={seq_len}: "
                    f"seg={best['segment_size']}, "
                    f"dil={best['dilation_rate']} "
                    f"({best['speedup']:.2f}x speedup)"
                )


if __name__ == "__main__":
    main()
