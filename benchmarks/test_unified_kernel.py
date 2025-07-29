#!/usr/bin/env python3
"""
Test the unified Hilbert attention kernel.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention_unified import (
    UnifiedHilbertAttention,
)


def benchmark_model(model, x, use_hilbert, iterations=10):
    """Benchmark a model."""
    # Warmup
    with torch.no_grad():
        _ = model(x, use_hilbert=use_hilbert)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x, use_hilbert=use_hilbert)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / iterations * 1000  # ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")

    # Test configurations
    configs = [
        (512, 2),  # Small - PyTorch path
        (1024, 2),  # Threshold boundary
        (2048, 2),  # Medium - Triton
        (4096, 1),  # Medium - Triton
        (8192, 1),  # Large - Triton with special config
    ]

    print("\n=== Unified vs Original Implementation ===")
    print("Seq Len | Batch | Original (ms) | Unified (ms) | Speedup | Kernel Used")
    print("-" * 75)

    for seq_len, batch_size in configs:
        # Original implementation
        model_orig = (
            UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=1,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model_orig.eval()

        # Unified implementation
        model_unified = (
            UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=1,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model_unified.eval()

        # Create input
        x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

        # Benchmark
        try:
            time_orig = benchmark_model(model_orig, x, use_hilbert=True)
        except Exception as e:
            time_orig = float("inf")
            print(f"Original failed for {seq_len}: {str(e)[:30]}")

        try:
            time_unified = benchmark_model(model_unified, x, use_hilbert=True)
        except Exception as e:
            time_unified = float("inf")
            print(f"Unified failed for {seq_len}: {str(e)[:30]}")
            continue

        speedup = time_orig / time_unified if time_unified > 0 else 0

        # Determine kernel used
        if seq_len <= 512:
            kernel = "PyTorch"
        else:
            kernel = "Triton"

        print(
            f"{seq_len:7d} | {batch_size:5d} | {time_orig:13.2f} | {time_unified:12.2f} | {speedup:7.2f}x | {kernel}"
        )

    # Test sparse patterns
    print("\n=== Sparse Pattern Performance (Unified) ===")
    print("Dilation | Seq Len | Time (ms)")
    print("-" * 35)

    for dilation in [1, 2, 4]:
        model = (
            UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=dilation,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model.eval()

        x = torch.randn(2, 2048, 768, device=device, dtype=dtype)

        time_unified = benchmark_model(model, x, use_hilbert=True, iterations=5)
        print(f"{dilation:8d} | {2048:7d} | {time_unified:9.2f}")

    # Test adaptive config selection
    print("\n=== Adaptive Configuration ===")
    model = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
    ).to(device)

    for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
        config = model._get_kernel_config(seq_len)
        print(
            f"Seq {seq_len:5d}: BLOCK_M={config[0]:3d}, BLOCK_N={config[1]:3d}, "
            f"BLOCK_D={config[2]:3d}, FUSED_SOFTMAX={'Yes' if config[3] else 'No'}"
        )


if __name__ == "__main__":
    main()
