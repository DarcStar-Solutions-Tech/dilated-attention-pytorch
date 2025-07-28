#!/usr/bin/env python3
"""
Comprehensive benchmark for Triton kernels.

Tests performance across:
- Different sequence lengths
- Different head dimensions
- Different batch sizes
- Hilbert vs standard attention
- Triton vs PyTorch implementations
- Float32 vs Float16
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Benchmark configuration
WARMUP_ITERS = 10
MEASURE_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def time_cuda_kernel(func, *args, warmup=WARMUP_ITERS, iters=MEASURE_ITERS):
    """Time a CUDA kernel with proper synchronization."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Measure
    torch.cuda.synchronize()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        func(*args)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.mean(times), np.std(times)


def benchmark_hilbert_attention_core():
    """Benchmark HilbertAttentionCore with various configurations."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0),
        "benchmarks": [],
    }

    print("=== Benchmarking HilbertAttentionCore ===\n")

    # Test configurations
    configs = [
        # (batch_size, seq_len, hidden_dim, num_heads, segment_size)
        (1, 128, 256, 8, 32),
        (1, 256, 256, 8, 64),
        (1, 512, 256, 8, 128),
        (1, 1024, 256, 8, 128),
        (1, 2048, 256, 8, 256),
        (4, 512, 256, 8, 128),
        (8, 512, 256, 8, 128),
        (1, 512, 512, 16, 128),
        (1, 512, 768, 12, 128),  # BERT-base like
        (1, 512, 1024, 16, 128),  # BERT-large like
    ]

    for batch, seq_len, hidden_dim, num_heads, segment_size in configs:
        print(
            f"\nConfig: batch={batch}, seq_len={seq_len}, hidden={hidden_dim}, heads={num_heads}"
        )

        # Create module
        module = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                use_custom_backward=False,  # For fair comparison
            )
            .cuda()
            .eval()
        )

        # Create input
        x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

        # Benchmark with Hilbert
        def forward_hilbert():
            with torch.no_grad():
                return module(x, use_hilbert=True)

        time_hilbert, std_hilbert = time_cuda_kernel(forward_hilbert)

        # Benchmark without Hilbert
        def forward_standard():
            with torch.no_grad():
                return module(x, use_hilbert=False)

        time_standard, std_standard = time_cuda_kernel(forward_standard)

        # Calculate throughput (tokens/sec)
        total_tokens = batch * seq_len
        throughput_hilbert = (total_tokens / time_hilbert) * 1000  # ms to sec
        throughput_standard = (total_tokens / time_standard) * 1000

        print(
            f"  Hilbert:  {time_hilbert:.2f} ± {std_hilbert:.2f} ms ({throughput_hilbert:.0f} tokens/sec)"
        )
        print(
            f"  Standard: {time_standard:.2f} ± {std_standard:.2f} ms ({throughput_standard:.0f} tokens/sec)"
        )
        print(f"  Speedup:  {time_standard / time_hilbert:.2f}x")

        results["benchmarks"].append(
            {
                "config": {
                    "batch_size": batch,
                    "seq_len": seq_len,
                    "hidden_dim": hidden_dim,
                    "num_heads": num_heads,
                    "segment_size": segment_size,
                },
                "hilbert": {
                    "time_ms": time_hilbert,
                    "std_ms": std_hilbert,
                    "throughput": throughput_hilbert,
                },
                "standard": {
                    "time_ms": time_standard,
                    "std_ms": std_standard,
                    "throughput": throughput_standard,
                },
                "speedup": time_standard / time_hilbert,
            }
        )

    return results


def benchmark_float16_performance():
    """Benchmark float16 vs float32 performance."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    print("\n=== Benchmarking Float16 vs Float32 ===\n")

    results = []
    seq_lengths = [128, 256, 512, 1024]

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")

        # Float32 module
        module_fp32 = (
            HilbertAttentionCore(
                hidden_dim=256, num_heads=8, segment_size=min(128, seq_len)
            )
            .cuda()
            .eval()
        )

        # Float16 module
        module_fp16 = (
            HilbertAttentionCore(
                hidden_dim=256, num_heads=8, segment_size=min(128, seq_len)
            )
            .cuda()
            .half()
            .eval()
        )

        # Inputs
        x_fp32 = torch.randn(4, seq_len, 256, device="cuda")
        x_fp16 = x_fp32.half()

        # Benchmark FP32
        def forward_fp32():
            with torch.no_grad():
                return module_fp32(x_fp32, use_hilbert=True)

        time_fp32, std_fp32 = time_cuda_kernel(forward_fp32)

        # Benchmark FP16
        def forward_fp16():
            with torch.no_grad():
                return module_fp16(x_fp16, use_hilbert=True)

        time_fp16, std_fp16 = time_cuda_kernel(forward_fp16)

        speedup = time_fp32 / time_fp16
        print(f"  FP32: {time_fp32:.2f} ± {std_fp32:.2f} ms")
        print(f"  FP16: {time_fp16:.2f} ± {std_fp16:.2f} ms")
        print(f"  FP16 Speedup: {speedup:.2f}x")

        results.append(
            {
                "seq_len": seq_len,
                "fp32_ms": time_fp32,
                "fp16_ms": time_fp16,
                "speedup": speedup,
            }
        )

    return results


def benchmark_vs_standard_attention():
    """Compare Triton kernel with standard PyTorch attention."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    print("\n=== Benchmarking vs Standard PyTorch Attention ===\n")

    results = []

    for seq_len in [256, 512, 1024, 2048]:
        print(f"\nSequence length: {seq_len}")

        batch_size = 4
        hidden_dim = 256
        num_heads = 8
        _ = hidden_dim // num_heads

        # Our Triton implementation
        triton_module = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=min(256, seq_len),
            )
            .cuda()
            .eval()
        )

        # Standard PyTorch MHA
        pytorch_mha = (
            nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )
            .cuda()
            .eval()
        )

        # Input
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

        # Benchmark Triton
        def forward_triton():
            with torch.no_grad():
                return triton_module(x, use_hilbert=True)

        time_triton, std_triton = time_cuda_kernel(forward_triton)

        # Benchmark PyTorch
        def forward_pytorch():
            with torch.no_grad():
                return pytorch_mha(x, x, x, need_weights=False)[0]

        time_pytorch, std_pytorch = time_cuda_kernel(forward_pytorch)

        speedup = time_pytorch / time_triton
        print(f"  Triton:  {time_triton:.2f} ± {std_triton:.2f} ms")
        print(f"  PyTorch: {time_pytorch:.2f} ± {std_pytorch:.2f} ms")
        print(f"  Triton Speedup: {speedup:.2f}x")

        results.append(
            {
                "seq_len": seq_len,
                "triton_ms": time_triton,
                "pytorch_ms": time_pytorch,
                "speedup": speedup,
            }
        )

    return results


def benchmark_memory_usage():
    """Benchmark memory usage of Triton kernels."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    print("\n=== Benchmarking Memory Usage ===\n")

    results = []

    for seq_len in [512, 1024, 2048, 4096]:
        try:
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create module and input
            module = HilbertAttentionCore(
                hidden_dim=256, num_heads=8, segment_size=min(256, seq_len)
            ).cuda()

            x = torch.randn(1, seq_len, 256, device="cuda")

            # Measure memory for forward pass
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()

            with torch.no_grad():
                _ = module(x, use_hilbert=True)

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated()

            mem_used_mb = (peak_mem - start_mem) / (1024 * 1024)

            print(f"Seq length {seq_len}: {mem_used_mb:.1f} MB")

            results.append({"seq_len": seq_len, "memory_mb": mem_used_mb})

        except torch.cuda.OutOfMemoryError:
            print(f"Seq length {seq_len}: OOM")
            results.append({"seq_len": seq_len, "memory_mb": None})

    return results


def plot_results(benchmark_results, fp16_results, vs_pytorch_results):
    """Create visualization plots."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Hilbert vs Standard performance
    ax1 = axes[0, 0]
    seq_lens = [
        r["config"]["seq_len"]
        for r in benchmark_results["benchmarks"]
        if r["config"]["batch_size"] == 1
    ]
    hilbert_times = [
        r["hilbert"]["time_ms"]
        for r in benchmark_results["benchmarks"]
        if r["config"]["batch_size"] == 1
    ]
    standard_times = [
        r["standard"]["time_ms"]
        for r in benchmark_results["benchmarks"]
        if r["config"]["batch_size"] == 1
    ]

    if seq_lens:
        ax1.plot(seq_lens[:5], hilbert_times[:5], "b-o", label="Hilbert")
        ax1.plot(seq_lens[:5], standard_times[:5], "r-s", label="Standard")
        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Hilbert vs Standard Attention")
        ax1.legend()
        ax1.grid(True)

    # Plot 2: FP16 speedup
    ax2 = axes[0, 1]
    if fp16_results:
        seq_lens = [r["seq_len"] for r in fp16_results]
        speedups = [r["speedup"] for r in fp16_results]
        ax2.bar(range(len(seq_lens)), speedups)
        ax2.set_xticks(range(len(seq_lens)))
        ax2.set_xticklabels(seq_lens)
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("FP16 Speedup")
        ax2.set_title("FP16 vs FP32 Speedup")
        ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5)
        ax2.grid(True, axis="y")

    # Plot 3: Triton vs PyTorch
    ax3 = axes[1, 0]
    if vs_pytorch_results:
        seq_lens = [r["seq_len"] for r in vs_pytorch_results]
        triton_times = [r["triton_ms"] for r in vs_pytorch_results]
        pytorch_times = [r["pytorch_ms"] for r in vs_pytorch_results]

        x = np.arange(len(seq_lens))
        width = 0.35

        ax3.bar(x - width / 2, triton_times, width, label="Triton")
        ax3.bar(x + width / 2, pytorch_times, width, label="PyTorch")
        ax3.set_xticks(x)
        ax3.set_xticklabels(seq_lens)
        ax3.set_xlabel("Sequence Length")
        ax3.set_ylabel("Time (ms)")
        ax3.set_title("Triton vs PyTorch MHA")
        ax3.legend()
        ax3.grid(True, axis="y")

    # Plot 4: Throughput scaling
    ax4 = axes[1, 1]
    batch_sizes = [1, 4, 8]
    throughputs = []
    for bs in batch_sizes:
        tp = [
            r["hilbert"]["throughput"]
            for r in benchmark_results["benchmarks"]
            if r["config"]["batch_size"] == bs and r["config"]["seq_len"] == 512
        ]
        if tp:
            throughputs.append(tp[0])

    if throughputs:
        ax4.plot(batch_sizes[: len(throughputs)], throughputs, "g-o")
        ax4.set_xlabel("Batch Size")
        ax4.set_ylabel("Throughput (tokens/sec)")
        ax4.set_title("Throughput Scaling with Batch Size")
        ax4.grid(True)

    plt.tight_layout()
    plt.savefig("triton_benchmark_results.png", dpi=150)
    print("\nPlots saved to triton_benchmark_results.png")


def main():
    """Run all benchmarks."""
    print("=== Triton Kernel Performance Benchmarks ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    try:
        import triton

        print(f"Triton: {triton.__version__}")
    except ImportError:
        print("Triton not available")
        return

    # Run benchmarks
    benchmark_results = benchmark_hilbert_attention_core()
    fp16_results = benchmark_float16_performance()
    vs_pytorch_results = benchmark_vs_standard_attention()
    memory_results = benchmark_memory_usage()

    # Save results
    all_results = {
        "main_benchmarks": benchmark_results,
        "fp16_benchmarks": fp16_results,
        "vs_pytorch": vs_pytorch_results,
        "memory_usage": memory_results,
    }

    with open("triton_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nResults saved to triton_benchmark_results.json")

    # Create plots
    try:
        plot_results(benchmark_results, fp16_results, vs_pytorch_results)
    except Exception as e:
        print(f"Failed to create plots: {e}")

    # Summary
    print("\n=== Summary ===")

    # Average speedup
    speedups = [r["speedup"] for r in benchmark_results["benchmarks"]]
    avg_speedup = np.mean(speedups) if speedups else 0
    print(f"Average Hilbert speedup: {avg_speedup:.2f}x")

    # FP16 speedup
    if fp16_results:
        fp16_speedups = [r["speedup"] for r in fp16_results]
        avg_fp16_speedup = np.mean(fp16_speedups)
        print(f"Average FP16 speedup: {avg_fp16_speedup:.2f}x")

    # vs PyTorch
    if vs_pytorch_results:
        pytorch_speedups = [r["speedup"] for r in vs_pytorch_results]
        avg_pytorch_speedup = np.mean(pytorch_speedups)
        print(f"Average speedup vs PyTorch MHA: {avg_pytorch_speedup:.2f}x")


if __name__ == "__main__":
    main()
