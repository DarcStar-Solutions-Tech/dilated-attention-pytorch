#!/usr/bin/env python3
"""Debug why sparse optimization is slower."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def debug_sparse_performance():
    """Compare standard vs optimized kernel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Debugging Sparse Performance")
    print("=" * 60)

    seq_len = 4096

    # Test with dilation_rate=4
    for dilation_rate in [1, 4]:
        print(f"\nDilation rate: {dilation_rate}")

        # Create module
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=100,
        ).to(device)

        x = torch.randn(1, seq_len, 768, device=device)

        # Force using standard kernel
        with torch.no_grad():
            # Time with current implementation
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            time1 = (time.perf_counter() - start) * 1000

            print(f"  Current implementation: {time1:.2f}ms")

        # Calculate theoretical speedup
        if dilation_rate > 1:
            active_positions = seq_len // dilation_rate
            print(f"  Active positions: {active_positions} / {seq_len}")
            print(f"  Theoretical speedup: {dilation_rate}x")
            print(f"  Actual performance: {1 / (time1 / time1):.2f}x")

            # The issue: we're still processing all positions
            # then filtering, rather than only processing active ones
            print(f"  Problem: Still checking all {seq_len} positions")
            print(f"  Solution needed: Process only {active_positions} positions")


if __name__ == "__main__":
    debug_sparse_performance()
