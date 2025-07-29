#!/usr/bin/env python3
"""Quick benchmark for UnifiedHilbertAttention kernel."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def benchmark():
    """Run quick benchmark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768, num_heads=12, segment_size=128, dilation_rate=1, dropout=0.0
    ).to(device)

    # Test configurations
    configs = [
        (2, 512),  # batch_size, seq_len
        (2, 1024),
        (2, 2048),
    ]

    print("\nBenchmark Results:")
    print("=" * 60)
    print(f"{'Config':<20} {'Standard (ms)':<15} {'Hilbert (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for batch_size, seq_len in configs:
        x = torch.randn(batch_size, seq_len, 768, device=device)

        # Warmup
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
            _ = module(x, use_hilbert=True)

        # Benchmark standard
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) / 5 * 1000

        # Benchmark Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(5):
                _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) / 5 * 1000

        speedup = std_time / hilbert_time
        print(
            f"B={batch_size}, L={seq_len:<6} {std_time:>12.2f} {hilbert_time:>15.2f} {speedup:>9.2f}x"
        )

    print("\nBackend:", "Triton" if module._triton_available else "PyTorch")


if __name__ == "__main__":
    benchmark()
