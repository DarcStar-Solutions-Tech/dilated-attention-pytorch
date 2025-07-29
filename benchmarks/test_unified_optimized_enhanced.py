#!/usr/bin/env python3
"""
Benchmark comparing UnifiedHilbertAttentionOptimizedEnhanced vs other implementations.
"""

import torch
import time
import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_model(model, x, use_hilbert, warmup=3, iterations=10):
    """Benchmark a model with proper warmup and timing."""
    # Move model to eval mode
    model.eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x, use_hilbert=use_hilbert)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time with CUDA events for accuracy
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(x, use_hilbert=use_hilbert)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / iterations
    else:
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                _ = model(x, use_hilbert=use_hilbert)
        elapsed_time = (time.perf_counter() - start) / iterations * 1000

    return elapsed_time


def create_models(hidden_dim, num_heads, segment_size, dilation_rate, device, dtype):
    """Create all model variants with same configuration."""
    models = {}

    # Original UnifiedHilbertAttention for reference
    models["Original"] = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=1024,
        )
        .to(device)
        .to(dtype)
    )

    # Unified implementation
    models["Unified"] = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=1024,
        )
        .to(device)
        .to(dtype)
    )

    # Unified Optimized
    models["Unified Optimized"] = (
        UnifiedHilbertAttentionOptimized(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=1024,
        )
        .to(device)
        .to(dtype)
    )

    # NEW: Unified Optimized Enhanced
    models["Unified Opt Enhanced"] = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=1024,
            enable_8k_optimization=True,
            enable_multi_row=True,
        )
        .to(device)
        .to(dtype)
    )

    return models


def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU (will be slower)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print()

    # Test configurations
    configs = [
        # (seq_len, batch_size, hidden_dim, num_heads, segment_size, dilation_rate, description)
        (512, 4, 768, 12, 128, 1, "Small sequence"),
        (1024, 2, 768, 12, 128, 1, "1K sequence"),
        (2048, 2, 768, 12, 128, 1, "2K sequence"),
        (4096, 1, 768, 12, 128, 1, "4K sequence"),
        (8192, 1, 768, 12, 128, 1, "8K sequence (special optimization)"),
        (16384, 1, 768, 12, 128, 1, "16K sequence"),
        # Sparse patterns
        (2048, 2, 768, 12, 128, 2, "2K sparse (dilation=2)"),
        (4096, 1, 768, 12, 128, 4, "4K sparse (dilation=4)"),
        (8192, 1, 768, 12, 128, 8, "8K sparse (dilation=8)"),
    ]

    print("=" * 100)
    print("UNIFIED OPTIMIZED ENHANCED IMPLEMENTATION COMPARISON")
    print("=" * 100)
    print()

    results = []

    for (
        seq_len,
        batch_size,
        hidden_dim,
        num_heads,
        segment_size,
        dilation_rate,
        desc,
    ) in configs:
        print(f"\n{desc}:")
        print(f"  Sequence length: {seq_len}, Batch size: {batch_size}")
        print(f"  Hidden dim: {hidden_dim}, Heads: {num_heads}")
        print(f"  Segment size: {segment_size}, Dilation rate: {dilation_rate}")
        print()

        # Create input
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)

        # Create models
        models = create_models(
            hidden_dim, num_heads, segment_size, dilation_rate, device, dtype
        )

        # Benchmark each model
        times = {}
        for name, model in models.items():
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                time_ms = benchmark_model(model, x, use_hilbert=True)
                times[name] = time_ms
                print(f"  {name:20s}: {time_ms:8.2f} ms")
            except Exception as e:
                print(f"  {name:20s}: Failed - {str(e)[:50]}")
                times[name] = float("inf")

        # Calculate speedups
        if times["Unified Optimized"] != float("inf"):
            print("\n  Speedup vs Unified Optimized:")
            for name in ["Original", "Unified", "Unified Opt Enhanced"]:
                if name in times and times[name] != float("inf"):
                    speedup = times["Unified Optimized"] / times[name]
                    if speedup > 1:
                        print(f"    {name:20s}: {speedup:6.2f}x faster")
                    else:
                        print(f"    {name:20s}: {1 / speedup:6.2f}x slower")

        # Check if Enhanced is better
        if times["Unified Opt Enhanced"] != float("inf") and times[
            "Unified Optimized"
        ] != float("inf"):
            if times["Unified Opt Enhanced"] < times["Unified Optimized"]:
                improvement = (
                    times["Unified Optimized"] / times["Unified Opt Enhanced"] - 1
                ) * 100
                print(
                    f"\n  ✓ Enhanced is {improvement:.1f}% faster than Unified Optimized"
                )

        results.append(
            {
                "config": desc,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dilation_rate": dilation_rate,
                "times": times,
            }
        )

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    # Dense attention summary
    print("\nDense Attention (dilation_rate=1):")
    print(
        f"{'Seq Len':>8} | {'Original':>10} | {'Unified':>10} | {'Unified Opt':>12} | {'Enhanced':>12} | {'Best':>15}"
    )
    print("-" * 85)

    for result in results:
        if result["dilation_rate"] == 1:
            times = result["times"]
            best_name = min(
                times.items(),
                key=lambda x: x[1] if x[1] != float("inf") else float("inf"),
            )[0]

            print(f"{result['seq_len']:8d} | ", end="")
            print(f"{times.get('Original', float('inf')):9.2f}ms | ", end="")
            print(f"{times.get('Unified', float('inf')):9.2f}ms | ", end="")
            print(f"{times.get('Unified Optimized', float('inf')):11.2f}ms | ", end="")
            print(
                f"{times.get('Unified Opt Enhanced', float('inf')):11.2f}ms | ", end=""
            )
            print(f"{best_name:>15}")

    # Sparse attention summary
    print("\nSparse Attention (dilation_rate>1):")
    print(
        f"{'Seq Len':>8} | {'Dilation':>8} | {'Unified Opt':>12} | {'Enhanced':>12} | {'Speedup':>10}"
    )
    print("-" * 65)

    for result in results:
        if result["dilation_rate"] > 1:
            times = result["times"]
            if times["Unified Optimized"] != float("inf") and times[
                "Unified Opt Enhanced"
            ] != float("inf"):
                speedup = times["Unified Optimized"] / times["Unified Opt Enhanced"]
                print(
                    f"{result['seq_len']:8d} | {result['dilation_rate']:8d} | ", end=""
                )
                print(f"{times['Unified Optimized']:11.2f}ms | ", end="")
                print(f"{times['Unified Opt Enhanced']:11.2f}ms | ", end="")
                print(f"{speedup:9.2f}x")

    # Key findings
    print("\n" + "=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)

    # Check 8K optimization
    for result in results:
        if result["seq_len"] == 8192 and result["dilation_rate"] == 1:
            times = result["times"]
            if "Unified Opt Enhanced" in times and "Unified Optimized" in times:
                if times["Unified Opt Enhanced"] < times["Unified Optimized"]:
                    improvement = (
                        times["Unified Optimized"] / times["Unified Opt Enhanced"] - 1
                    ) * 100
                    print(
                        f"\n8K Optimization: Enhanced is {improvement:.1f}% faster than Unified Optimized"
                    )

    # Overall performance improvement
    total_improvement = 0
    count = 0
    for result in results:
        times = result["times"]
        if times["Unified Optimized"] != float("inf") and times[
            "Unified Opt Enhanced"
        ] != float("inf"):
            if times["Unified Opt Enhanced"] < times["Unified Optimized"]:
                improvement = (
                    times["Unified Optimized"] / times["Unified Opt Enhanced"] - 1
                ) * 100
                total_improvement += improvement
                count += 1

    if count > 0:
        avg_improvement = total_improvement / count
        print(
            f"\nAverage improvement: Enhanced is {avg_improvement:.1f}% faster across {count} configs"
        )


if __name__ == "__main__":
    main()
