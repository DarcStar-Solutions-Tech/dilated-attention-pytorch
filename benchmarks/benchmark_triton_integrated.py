#!/usr/bin/env python3
"""
Benchmark the Triton integrated Ring Dilated Attention implementation.

This script compares:
1. Baseline (no Hilbert ordering)
2. Python-based Hilbert implementation
3. Triton kernel Hilbert implementation
"""

import torch
import torch.nn as nn
import argparse
import gc
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


def get_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
        }
    return {"allocated": 0, "reserved": 0, "max_allocated": 0}


def benchmark_attention(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    warmup_iters: int = 5,
    test_iters: int = 20,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """Benchmark a single attention model."""

    # Create input tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    # Get initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = get_memory_info()

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(q, k, v, is_causal=False)

    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Benchmark forward pass
    start_event.record()
    with torch.no_grad():
        for _ in range(test_iters):
            output = model(q, k, v, is_causal=False)
    end_event.record()

    torch.cuda.synchronize()

    # Get metrics
    forward_time = start_event.elapsed_time(end_event) / test_iters  # ms
    peak_mem = get_memory_info()

    # Calculate throughput
    total_tokens = batch_size * seq_len
    throughput = (total_tokens / forward_time) * 1000  # tokens/sec

    return {
        "forward_time_ms": forward_time,
        "throughput_tokens_sec": throughput,
        "memory_gb": peak_mem["max_allocated"] - initial_mem["allocated"],
        "output_shape": output.shape,
    }


def compare_implementations(args):
    """Compare different implementations."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: Running on CPU, performance will be significantly slower")

    print(f"\n{'=' * 80}")
    print("Benchmarking Triton Integrated Ring Dilated Attention")
    print(f"{'=' * 80}")
    print(
        f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Sequence Length: {args.seq_len}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Num Heads: {args.num_heads}")
    print(f"Head Dim: {args.head_dim}")
    print(f"Segment Lengths: {args.segment_lengths}")
    print(f"Dilation Rates: {args.dilation_rates}")
    print(f"{'=' * 80}\n")

    # Create models
    models = {}

    # 1. Baseline (no Hilbert)
    models["Baseline (No Hilbert)"] = RingDilatedAttentionSimpleTriton(
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        use_hilbert=False,
    )

    # 2. Python Hilbert (SimpleTriton)
    models["Python Hilbert"] = RingDilatedAttentionSimpleTriton(
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        use_hilbert=True,
    )

    # 3. Triton Integrated (with fallback to PyTorch)
    models["Triton Integrated (PyTorch)"] = RingDilatedAttentionTritonIntegrated(
        segment_lengths=args.segment_lengths,
        dilation_rates=args.dilation_rates,
        dropout=0.0,
        device=device,
        dtype=torch.float32,
        use_triton=False,  # Use PyTorch implementation
    )

    # 4. Triton Integrated (with Triton kernels)
    try:
        models["Triton Integrated (Triton)"] = RingDilatedAttentionTritonIntegrated(
            segment_lengths=args.segment_lengths,
            dilation_rates=args.dilation_rates,
            dropout=0.0,
            device=device,
            dtype=torch.float32,
            use_triton=True,  # Use Triton kernels
        )
    except Exception as e:
        print(f"Warning: Could not create Triton kernel version: {e}")

    # Benchmark each model
    results = {}
    baseline_throughput = None

    for name, model in models.items():
        print(f"\n{'-' * 60}")
        print(f"Testing: {name}")
        print(f"{'-' * 60}")

        try:
            result = benchmark_attention(
                model,
                args.batch_size,
                args.seq_len,
                args.num_heads,
                args.head_dim,
                warmup_iters=args.warmup_iters,
                test_iters=args.test_iters,
                device=device,
            )

            results[name] = result

            # Calculate speedup
            if baseline_throughput is None:
                baseline_throughput = result["throughput_tokens_sec"]
                speedup = 1.0
            else:
                speedup = result["throughput_tokens_sec"] / baseline_throughput

            print(f"Forward Time: {result['forward_time_ms']:.2f} ms")
            print(f"Throughput: {result['throughput_tokens_sec']:,.0f} tokens/sec")
            print(f"Memory Used: {result['memory_gb']:.2f} GB")
            print(f"Speedup vs Baseline: {speedup:.2f}x")

        except Exception as e:
            print(f"ERROR: Failed to benchmark {name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(
        f"{'Implementation':<30} {'Time (ms)':<12} {'Throughput':<15} {'Memory (GB)':<12} {'Speedup':<10}"
    )
    print(f"{'-' * 80}")

    baseline_time = None
    for name, result in results.items():
        if baseline_time is None:
            baseline_time = result["forward_time_ms"]
            speedup = 1.0
        else:
            speedup = baseline_time / result["forward_time_ms"]

        print(
            f"{name:<30} {result['forward_time_ms']:<12.2f} "
            f"{result['throughput_tokens_sec']:<15,.0f} "
            f"{result['memory_gb']:<12.2f} {speedup:<10.2f}x"
        )

    print(f"\n{'=' * 80}")
    print("Analysis:")
    print(f"{'=' * 80}")

    if "Python Hilbert" in results and "Baseline (No Hilbert)" in results:
        overhead = (
            results["Python Hilbert"]["forward_time_ms"]
            - results["Baseline (No Hilbert)"]["forward_time_ms"]
        )
        overhead_pct = (
            overhead / results["Baseline (No Hilbert)"]["forward_time_ms"]
        ) * 100
        print(f"Python Hilbert Overhead: {overhead:.2f} ms ({overhead_pct:.1f}%)")

    if "Triton Integrated (Triton)" in results and "Baseline (No Hilbert)" in results:
        improvement = (
            results["Baseline (No Hilbert)"]["forward_time_ms"]
            - results["Triton Integrated (Triton)"]["forward_time_ms"]
        )
        improvement_pct = (
            improvement / results["Baseline (No Hilbert)"]["forward_time_ms"]
        ) * 100
        print(
            f"Triton Kernel Improvement: {improvement:.2f} ms ({improvement_pct:.1f}% faster)"
        )

    # Test different sequence lengths
    if args.test_scaling:
        print(f"\n{'=' * 80}")
        print("Sequence Length Scaling")
        print(f"{'=' * 80}")

        seq_lengths = [4096, 8192, 16384, 32768]

        for seq_len in seq_lengths:
            if seq_len > args.seq_len:
                break

            print(f"\nSequence Length: {seq_len}")
            print(f"{'-' * 40}")

            # Only test the best performing model
            best_model_name = min(
                results.keys(), key=lambda k: results[k]["forward_time_ms"]
            )
            model = models[best_model_name]

            try:
                result = benchmark_attention(
                    model,
                    args.batch_size,
                    seq_len,
                    args.num_heads,
                    args.head_dim,
                    warmup_iters=3,
                    test_iters=10,
                    device=device,
                )

                print(
                    f"{best_model_name}: {result['forward_time_ms']:.2f} ms, "
                    f"{result['throughput_tokens_sec']:,.0f} tokens/sec, "
                    f"{result['memory_gb']:.2f} GB"
                )

            except Exception as e:
                print(f"Failed at seq_len={seq_len}: {e}")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Triton Integrated Implementation"
    )
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs="+",
        default=[2048, 4096],
        help="Segment lengths for dilated attention",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Dilation rates for dilated attention",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test_iters", type=int, default=20, help="Number of test iterations"
    )
    parser.add_argument(
        "--test_scaling", action="store_true", help="Test sequence length scaling"
    )

    args = parser.parse_args()

    # Validate arguments
    assert len(args.segment_lengths) == len(args.dilation_rates), (
        "segment_lengths and dilation_rates must have the same length"
    )

    compare_implementations(args)


if __name__ == "__main__":
    main()
