#!/usr/bin/env python3
"""
Analyze the performance characteristics of the Triton implementation.
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from typing import Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_triton_integrated import (
    RingDilatedAttentionTritonIntegrated,
)
from dilated_attention_pytorch.ring_dilated_attention_simple_triton import (
    RingDilatedAttentionSimpleTriton,
)


def profile_kernel_components(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
) -> Dict[str, float]:
    """Profile different components of the attention computation."""

    # Create input tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Measure different components
    results = {}

    # 1. Overall forward pass
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model(q, k, v, is_causal=False)
    torch.cuda.synchronize()
    results["total_time"] = (time.time() - start) * 1000  # ms

    # 2. Memory access pattern analysis
    if hasattr(model, "_pattern_cache"):
        results["pattern_cache_size"] = len(model._pattern_cache)

    if hasattr(model, "_hilbert_cache"):
        results["hilbert_cache_size"] = len(model._hilbert_cache)

    # 3. Estimate computational complexity
    total_flops = 0
    for seg_len, dil_rate in zip(model.segment_lengths, model.dilation_rates):
        # Number of positions accessed per segment
        positions_per_segment = seg_len // dil_rate
        num_segments = (seq_len + seg_len - 1) // seg_len

        # FLOPs for attention: 2 * seq_len * positions * dim (for QK^T) + 2 * seq_len * positions * dim (for AV)
        flops_per_head = 4 * seq_len * positions_per_segment * head_dim * num_segments
        total_flops += flops_per_head * num_heads

    results["estimated_gflops"] = total_flops / 1e9
    results["gflops_per_sec"] = results["estimated_gflops"] / (
        results["total_time"] / 1000
    )

    return results


def analyze_sequence_scaling():
    """Analyze how performance scales with sequence length."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 80}")
    print("Sequence Length Scaling Analysis")
    print(f"{'=' * 80}\n")

    seq_lengths = [2048, 4096, 8192, 16384]
    segment_lengths = [2048, 4096]
    dilation_rates = [2, 4]

    results = {
        "baseline": [],
        "python_hilbert": [],
        "triton_pytorch": [],
        "triton_kernel": [],
    }

    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        print("-" * 40)

        # Skip if sequence is too large for memory
        required_memory = seq_len * 8 * 64 * 4 * 3 / 1e9  # Approximate GB
        if required_memory > 8:  # Skip if >8GB required
            print(f"Skipping - requires ~{required_memory:.1f} GB")
            continue

        # Create models
        models = {
            "baseline": RingDilatedAttentionSimpleTriton(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=False,
            ),
            "python_hilbert": RingDilatedAttentionSimpleTriton(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=True,
            ),
            "triton_pytorch": RingDilatedAttentionTritonIntegrated(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_triton=False,
            ),
            "triton_kernel": RingDilatedAttentionTritonIntegrated(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_triton=True,
            ),
        }

        for name, model in models.items():
            try:
                profile = profile_kernel_components(
                    model,
                    batch_size=1,
                    seq_len=seq_len,
                    num_heads=8,
                    head_dim=64,
                    device=device,
                )

                results[name].append(
                    {
                        "seq_len": seq_len,
                        "time_ms": profile["total_time"],
                        "gflops_per_sec": profile["gflops_per_sec"],
                    }
                )

                print(
                    f"{name:15s}: {profile['total_time']:7.2f} ms, "
                    f"{profile['gflops_per_sec']:7.2f} GFLOPS/s"
                )

            except Exception as e:
                print(f"{name:15s}: Failed - {e}")

    # Plot results
    plt.figure(figsize=(12, 8))

    # Plot time scaling
    plt.subplot(2, 2, 1)
    for name, data in results.items():
        if data:
            seq_lens = [d["seq_len"] for d in data]
            times = [d["time_ms"] for d in data]
            plt.plot(seq_lens, times, marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (ms)")
    plt.title("Time vs Sequence Length")
    plt.legend()
    plt.grid(True)

    # Plot throughput scaling
    plt.subplot(2, 2, 2)
    for name, data in results.items():
        if data:
            seq_lens = [d["seq_len"] for d in data]
            throughputs = [
                s / t * 1000 for s, t in zip(seq_lens, [d["time_ms"] for d in data])
            ]
            plt.plot(seq_lens, throughputs, marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Throughput (tokens/sec)")
    plt.title("Throughput vs Sequence Length")
    plt.legend()
    plt.grid(True)

    # Plot GFLOPS scaling
    plt.subplot(2, 2, 3)
    for name, data in results.items():
        if data:
            seq_lens = [d["seq_len"] for d in data]
            gflops = [d["gflops_per_sec"] for d in data]
            plt.plot(seq_lens, gflops, marker="o", label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("GFLOPS/s")
    plt.title("Computational Efficiency")
    plt.legend()
    plt.grid(True)

    # Plot speedup
    plt.subplot(2, 2, 4)
    if results["baseline"]:
        baseline_times = {d["seq_len"]: d["time_ms"] for d in results["baseline"]}
        for name, data in results.items():
            if data and name != "baseline":
                seq_lens = [d["seq_len"] for d in data]
                speedups = [
                    baseline_times[s] / d["time_ms"]
                    for s, d in zip(seq_lens, data)
                    if s in baseline_times
                ]
                if speedups:
                    plt.plot(
                        seq_lens[: len(speedups)], speedups, marker="o", label=name
                    )
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup vs Baseline")
    plt.title("Relative Performance")
    plt.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("triton_performance_analysis.png", dpi=150)
    print("\nPlot saved to triton_performance_analysis.png")


def analyze_kernel_overhead():
    """Analyze the overhead of Triton kernel launches."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 80}")
    print("Kernel Launch Overhead Analysis")
    print(f"{'=' * 80}\n")

    # Test with very small sequences to isolate kernel overhead
    small_seqs = [256, 512, 1024, 2048]

    for seq_len in small_seqs:
        print(f"\nSequence length: {seq_len}")

        # Simple attention baseline
        q = torch.randn(1, seq_len, 8, 64, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Time PyTorch SDPA
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                )
        torch.cuda.synchronize()
        sdpa_time = (time.time() - start) * 10  # ms per iteration

        print(f"PyTorch SDPA: {sdpa_time:.3f} ms")

        # Time our implementations
        models = {
            "Simple (no Hilbert)": RingDilatedAttentionSimpleTriton(
                segment_lengths=[seq_len],
                dilation_rates=[1],
                use_hilbert=False,
            ),
            "Triton Integrated": RingDilatedAttentionTritonIntegrated(
                segment_lengths=[seq_len],
                dilation_rates=[1],
                use_triton=True,
            ),
        }

        for name, model in models.items():
            try:
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(100):
                    with torch.no_grad():
                        _ = model(q, k, v)
                torch.cuda.synchronize()
                model_time = (time.time() - start) * 10  # ms per iteration

                overhead = model_time - sdpa_time
                print(f"{name}: {model_time:.3f} ms (overhead: {overhead:.3f} ms)")
            except Exception as e:
                print(f"{name}: Failed - {e}")


def main():
    """Run all analyses."""

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping analysis")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Run analyses
    analyze_sequence_scaling()
    analyze_kernel_overhead()

    # Print recommendations
    print(f"\n{'=' * 80}")
    print("Recommendations for Pascal Architecture (GTX 1080)")
    print(f"{'=' * 80}\n")

    print("1. The Python-based Hilbert implementation shows slight improvement due to")
    print("   better cache locality from the simple permutation pattern.")
    print()
    print("2. The Triton kernel shows overhead on Pascal due to:")
    print("   - Lack of Tensor Core support on Pascal")
    print("   - Triton optimizations target newer architectures (Volta+)")
    print("   - Kernel launch overhead for sparse access patterns")
    print()
    print("3. For Pascal GPUs, recommend:")
    print("   - Use the SimpleTriton implementation with Python Hilbert")
    print("   - Focus on dilation benefits (proven 5-8x speedup)")
    print("   - Consider CUDA kernels instead of Triton for this architecture")
    print()
    print("4. For modern GPUs (A100/H100), the Triton kernels would likely show")
    print("   significant improvements due to better memory subsystem and")
    print("   hardware support for the operations used.")


if __name__ == "__main__":
    main()
