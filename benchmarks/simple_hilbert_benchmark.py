#!/usr/bin/env python3
"""
Simple benchmark to test Hilbert attention performance.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention


def benchmark_forward(model, x, use_hilbert, iterations=10):
    """Benchmark forward pass only."""
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
    # Use float32 to avoid dtype issues
    dtype = torch.float32

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Test basic configurations
    print("\n=== Hilbert Attention Performance (float32) ===")
    print("Seq Len | Batch | Standard (ms) | Hilbert (ms) | Speedup | Implementation")
    print("-" * 75)

    configs = [
        (512, 2),  # Below threshold
        (1024, 2),  # At threshold
        (2048, 2),  # Fused kernel range
        (4096, 1),  # Fused kernel range
        (8192, 1),  # Fused kernel range
    ]

    for seq_len, batch_size in configs:
        # Create model
        model = (
            HilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=1,
                hilbert_threshold=1024,
            )
            .to(device)
            .to(dtype)
        )
        model.eval()

        # Create input
        x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

        # Benchmark standard attention
        try:
            time_standard = benchmark_forward(model, x, use_hilbert=False)
        except Exception as e:
            time_standard = float("inf")
            print(f"Standard failed for {seq_len}: {e}")
            continue

        # Benchmark Hilbert attention
        try:
            time_hilbert = benchmark_forward(model, x, use_hilbert=True)
        except Exception as e:
            time_hilbert = float("inf")
            print(f"Hilbert failed for {seq_len}: {e}")
            continue

        speedup = time_standard / time_hilbert if time_hilbert > 0 else 0

        # Determine implementation
        impl = "PyTorch"
        if seq_len > model.hilbert_threshold:
            if 2048 <= seq_len <= 16384 and model._fused_kernels_available:
                impl = "Fused"
            elif model._triton_available:
                impl = "Triton"

        print(
            f"{seq_len:7d} | {batch_size:5d} | {time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x | {impl}"
        )

    # Test sparse patterns
    print("\n=== Sparse Pattern Performance (seq_len=2048) ===")
    print("Dilation | Standard (ms) | Hilbert (ms) | Speedup")
    print("-" * 50)

    for dilation in [1, 2, 4]:
        model = (
            HilbertAttention(
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

        time_standard = benchmark_forward(model, x, use_hilbert=False, iterations=5)
        time_hilbert = benchmark_forward(model, x, use_hilbert=True, iterations=5)
        speedup = time_standard / time_hilbert

        print(
            f"{dilation:8d} | {time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x"
        )

    # Summary
    print("\n=== Summary ===")
    print("- Hilbert threshold: 1024 (sequences > 1024 use Hilbert reordering)")
    print("- Fused kernels: Active for sequences 2048-16384")
    print("- The implementation automatically selects the best backend")


if __name__ == "__main__":
    main()
