#!/usr/bin/env python3
"""
Focused test of sparse Hilbert optimization with key sequence lengths and dilation rates.
"""

import torch
import time
import gc
import matplotlib.pyplot as plt
import numpy as np


def quick_benchmark(module, seq_len, batch_size, hidden_dim, use_hilbert=True):
    """Quick benchmark with minimal iterations."""
    device = module.qkv_proj.weight.device

    try:
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

        # Single warmup
        with torch.no_grad():
            _ = module(x, use_hilbert=use_hilbert)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time 3 iterations
        start = time.perf_counter()
        for _ in range(3):
            with torch.no_grad():
                _ = module(x, use_hilbert=use_hilbert)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) / 3 * 1000  # ms

        if device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
            torch.cuda.reset_peak_memory_stats()
        else:
            memory_mb = 0

        return elapsed, memory_mb, True

    except Exception:
        return float("inf"), 0, False
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    print("Sparse Hilbert Scaling - Focused Test")
    print("=" * 100)

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 1
    segment_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Key test points
    test_configs = [
        # (seq_len, dilation_rate)
        (2048, 1),
        (2048, 4),
        (2048, 8),
        (4096, 1),
        (4096, 4),
        (4096, 8),
        (8192, 1),
        (8192, 4),
        (8192, 8),
        (8192, 16),
        (16384, 2),
        (16384, 4),
        (16384, 8),
        (16384, 16),
        (32768, 4),
        (32768, 8),
        (32768, 16),
        (32768, 32),
    ]

    # Import implementations
    from dilated_attention_pytorch.kernels import HilbertAttentionCore
    from dilated_attention_pytorch.kernels.hilbert_attention_sparse_simple import (
        HilbertAttentionSparseSimple,
    )

    print(
        f"\nConfiguration: hidden_dim={hidden_dim}, num_heads={num_heads}, segment_size={segment_size}"
    )
    print("-" * 100)
    print(
        f"{'Seq Len':>8} | {'Dil':>4} | {'Sparse/Seg':>10} | {'Original':>10} | {'Sparse':>10} | {'No Hilbert':>10} | {'Speedup':>8} | {'vs Base':>8} | {'Memory':>8}"
    )
    print("-" * 100)

    results = []

    for seq_len, dil_rate in test_configs:
        sparse_per_seg = segment_size // dil_rate

        # Create modules
        original = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dil_rate,
            )
            .to(device)
            .eval()
        )

        sparse = (
            HilbertAttentionSparseSimple(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dil_rate,
            )
            .to(device)
            .eval()
        )

        # Benchmark
        orig_time, orig_mem, _ = quick_benchmark(
            original, seq_len, batch_size, hidden_dim, True
        )
        sparse_time, sparse_mem, _ = quick_benchmark(
            sparse, seq_len, batch_size, hidden_dim, True
        )
        base_time, base_mem, _ = quick_benchmark(
            original, seq_len, batch_size, hidden_dim, False
        )

        # Calculate improvements
        speedup = orig_time / sparse_time if sparse_time > 0 else 0
        vs_baseline = base_time / sparse_time if sparse_time > 0 else 0

        # Store results
        results.append(
            {
                "seq_len": seq_len,
                "dil_rate": dil_rate,
                "sparse_per_seg": sparse_per_seg,
                "orig_time": orig_time,
                "sparse_time": sparse_time,
                "base_time": base_time,
                "speedup": speedup,
                "vs_baseline": vs_baseline,
                "memory": max(orig_mem, sparse_mem),
            }
        )

        # Print
        print(
            f"{seq_len:>8} | {dil_rate:>4} | {sparse_per_seg:>10} | "
            f"{orig_time:>8.1f}ms | {sparse_time:>8.1f}ms | {base_time:>8.1f}ms | "
            f"{speedup:>6.2f}x | {vs_baseline:>6.2f}x | {max(orig_mem, sparse_mem):>6.0f}MB"
        )

        # Cleanup
        del original, sparse
        gc.collect()
        torch.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Best configurations
    best_speedups = sorted(results, key=lambda x: x["speedup"], reverse=True)[:5]
    print("\nTop 5 Speedups (Original vs Sparse):")
    for r in best_speedups:
        print(
            f"  seq={r['seq_len']:>6}, dil={r['dil_rate']:>2}: {r['speedup']:>6.2f}x "
            f"({r['orig_time']:.1f}ms → {r['sparse_time']:.1f}ms)"
        )

    # Best vs baseline
    best_vs_base = sorted(results, key=lambda x: x["vs_baseline"], reverse=True)[:5]
    print("\nTop 5 Improvements vs No Hilbert:")
    for r in best_vs_base:
        print(
            f"  seq={r['seq_len']:>6}, dil={r['dil_rate']:>2}: {r['vs_baseline']:>6.2f}x faster than baseline"
        )

    # Scaling analysis
    print("\nScaling Patterns:")

    # By sequence length
    for seq in [4096, 8192, 16384]:
        seq_results = [r for r in results if r["seq_len"] == seq]
        if seq_results:
            avg_speedup = np.mean([r["speedup"] for r in seq_results])
            print(f"  seq={seq}: avg speedup {avg_speedup:.2f}x")

    # By dilation rate
    for dil in [4, 8, 16]:
        dil_results = [r for r in results if r["dil_rate"] == dil]
        if dil_results:
            avg_speedup = np.mean([r["speedup"] for r in dil_results])
            print(f"  dil={dil}: avg speedup {avg_speedup:.2f}x")

    # Memory efficiency
    print("\nMemory Efficiency:")
    print("  Hilbert map sizes:")
    print("    Original: O(sequence_length)")
    print("    Sparse:   O(segment_size / dilation_rate)")

    # Example calculation
    seq = 32768
    dil = 16
    print(f"\n  Example: seq={seq}, dil={dil}")
    print(f"    Original map: {seq:,} entries")
    print(f"    Sparse map:   {segment_size // dil} entries")
    print(f"    Reduction:    {seq / (segment_size // dil):.0f}x smaller")

    # Create simple plot
    create_simple_plot(results)


def create_simple_plot(results):
    """Create a simple visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Speedup by sequence length
    seq_lens = sorted(set(r["seq_len"] for r in results))
    avg_speedups = []

    for seq in seq_lens:
        seq_results = [r["speedup"] for r in results if r["seq_len"] == seq]
        avg_speedups.append(np.mean(seq_results))

    ax1.bar(range(len(seq_lens)), avg_speedups, color="green")
    ax1.set_xticks(range(len(seq_lens)))
    ax1.set_xticklabels([str(s) for s in seq_lens], rotation=45)
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Average Speedup")
    ax1.set_title("Sparse Hilbert Speedup by Sequence Length")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, v in enumerate(avg_speedups):
        ax1.text(i, v + 0.1, f"{v:.1f}x", ha="center")

    # Plot 2: Performance comparison for seq=8192
    seq_8k_results = [r for r in results if r["seq_len"] == 8192]
    if seq_8k_results:
        dil_rates = [r["dil_rate"] for r in seq_8k_results]
        orig_times = [r["orig_time"] for r in seq_8k_results]
        sparse_times = [r["sparse_time"] for r in seq_8k_results]
        base_times = [r["base_time"] for r in seq_8k_results]

        x = np.arange(len(dil_rates))
        width = 0.25

        ax2.bar(x - width, orig_times, width, label="Original Hilbert", color="red")
        ax2.bar(x, sparse_times, width, label="Sparse Hilbert", color="green")
        ax2.bar(x + width, base_times, width, label="No Hilbert", color="blue")

        ax2.set_xlabel("Dilation Rate")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("Performance Comparison (seq_len=8192)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(dil_rates)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("sparse_hilbert_focused_results.png", dpi=150)
    print("\nPlot saved to: sparse_hilbert_focused_results.png")


if __name__ == "__main__":
    main()
