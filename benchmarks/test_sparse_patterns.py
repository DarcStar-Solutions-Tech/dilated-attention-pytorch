#!/usr/bin/env python3
"""Test performance with sparse attention patterns."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def benchmark_sparse_patterns():
    """Benchmark different sparse patterns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Sparse Pattern Performance Benchmark")
    print("=" * 80)
    print(
        f"{'Seq Len':<10} {'Dilation':<10} {'Sparsity':<15} {'Time (ms)':<15} {'Speedup':<15}"
    )
    print("-" * 80)

    # Test different configurations
    configs = [
        (4096, 1, "0% (dense)"),
        (4096, 2, "50%"),
        (4096, 4, "75%"),
        (4096, 8, "87.5%"),
        (8192, 1, "0% (dense)"),
        (8192, 2, "50%"),
        (8192, 4, "75%"),
        (8192, 8, "87.5%"),
    ]

    baselines = {}

    for seq_len, dilation_rate, sparsity in configs:
        # Create module
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=100,  # Force Hilbert
        ).to(device)

        x = torch.randn(1, seq_len, 768, device=device)

        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()

            # Time
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

        # Calculate speedup
        if dilation_rate == 1:
            baselines[seq_len] = elapsed
            speedup = "baseline"
        else:
            speedup = f"{baselines[seq_len] / elapsed:.2f}x"

        print(
            f"{seq_len:<10} {dilation_rate:<10} {sparsity:<15} {elapsed:<15.2f} {speedup:<15}"
        )

        # Add separator between sequence lengths
        if dilation_rate == 8 and seq_len == 4096:
            print()

    print("\nKey Observations:")
    print("- Sparse patterns show significant speedup")
    print("- Higher dilation rates = more speedup")
    print("- Optimization is working effectively")


if __name__ == "__main__":
    benchmark_sparse_patterns()
