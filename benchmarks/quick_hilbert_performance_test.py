#!/usr/bin/env python3
"""
Quick Hilbert attention performance test.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention


def quick_benchmark(model, x, use_hilbert, iterations=5):
    """Quick benchmark."""
    # Warmup
    with torch.no_grad():
        _ = model(x, use_hilbert=use_hilbert)

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        with torch.no_grad():
            _ = model(x, use_hilbert=use_hilbert)

    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iterations * 1000  # ms


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Key sequence lengths
    seq_lengths = [1024, 2048, 4096, 8192]

    print("\n=== Hilbert Attention Performance ===")
    print("Seq Len | Standard (ms) | Hilbert (ms) | Speedup | Notes")
    print("-" * 70)

    for seq_len in seq_lengths:
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

        # Reduce batch size for larger sequences
        batch_size = 1 if seq_len > 4096 else 2
        x = torch.randn(batch_size, seq_len, 768, device=device, dtype=dtype)

        # Benchmark
        try:
            time_standard = quick_benchmark(model, x, use_hilbert=False)
            time_hilbert = quick_benchmark(model, x, use_hilbert=True)
            speedup = time_standard / time_hilbert

            # Determine what's being used
            notes = ""
            if seq_len <= 1024:
                notes = "Below threshold"
            elif 2048 <= seq_len <= 16384 and model._fused_kernels_available:
                notes = "Fused kernel"
            elif model._triton_available:
                notes = "Triton kernel"
            else:
                notes = "PyTorch fallback"

            print(
                f"{seq_len:7d} | {time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x | {notes}"
            )
        except Exception as e:
            print(f"{seq_len:7d} | Error: {str(e)[:40]}")

    # Test sparse patterns at 4K
    print("\n=== Sparse Patterns at 4096 ===")
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

        x = torch.randn(2, 4096, 768, device=device, dtype=dtype)

        time_standard = quick_benchmark(model, x, use_hilbert=False, iterations=3)
        time_hilbert = quick_benchmark(model, x, use_hilbert=True, iterations=3)
        speedup = time_standard / time_hilbert

        print(
            f"{dilation:8d} | {time_standard:13.2f} | {time_hilbert:12.2f} | {speedup:7.2f}x"
        )


if __name__ == "__main__":
    main()
