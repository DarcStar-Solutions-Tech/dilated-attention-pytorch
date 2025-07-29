#!/usr/bin/env python3
"""Test the impact of position optimization."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_sparse_attention_performance():
    """Test performance with sparse attention patterns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing Position Optimization Impact")
    print("=" * 60)

    # Test configurations with different dilation rates
    configs = [
        # (seq_len, dilation_rate, description)
        (4096, 1, "Dense attention"),
        (4096, 2, "Dilation rate 2 (50% sparse)"),
        (4096, 4, "Dilation rate 4 (75% sparse)"),
        (4096, 8, "Dilation rate 8 (87.5% sparse)"),
        (8192, 1, "Dense attention (8K)"),
        (8192, 4, "Dilation rate 4 (8K)"),
    ]

    print(f"{'Config':<30} {'Time (ms)':<15} {'Speedup vs Dense':<20}")
    print("-" * 65)

    for seq_len, dilation_rate, desc in configs:
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

            # Time
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

        # Store dense baseline for comparison
        if dilation_rate == 1:
            if seq_len == 4096:
                dense_4k = elapsed
            elif seq_len == 8192:
                dense_8k = elapsed
            speedup_str = "baseline"
        else:
            if seq_len == 4096:
                speedup = dense_4k / elapsed
            else:
                speedup = dense_8k / elapsed
            speedup_str = f"{speedup:.2f}x"

        print(f"{desc:<30} {elapsed:<15.2f} {speedup_str:<20}")

    print("\nExpected behavior:")
    print("- Sparse patterns should be faster due to fewer positions processed")
    print("- Higher dilation rates = fewer active positions = faster execution")


if __name__ == "__main__":
    test_sparse_attention_performance()
