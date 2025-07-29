#!/usr/bin/env python3
"""Test the fixed sparse attention optimization."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def test_sparse_optimization():
    """Test if sparse optimization compiles and runs correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing Fixed Sparse Optimization")
    print("=" * 60)

    # Test different dilation rates
    configs = [
        (2048, 128, 1, "Dense (baseline)"),
        (2048, 128, 2, "Sparse dilation=2"),
        (2048, 128, 4, "Sparse dilation=4"),
    ]

    for seq_len, segment_size, dilation_rate, desc in configs:
        print(f"\n{desc}:")
        print(f"  seq_len={seq_len}, segment={segment_size}, dilation={dilation_rate}")

        try:
            # Create module
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                dropout=0.0,
                hilbert_threshold=100,  # Force Hilbert
            ).to(device)

            x = torch.randn(1, seq_len, 768, device=device)

            # Test forward pass
            with torch.no_grad():
                # Warmup
                _ = module(x, use_hilbert=True)
                torch.cuda.synchronize()

                # Time
                start = time.perf_counter()
                output = module(x, use_hilbert=True)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000

            print(f"  Forward pass: {elapsed:.2f}ms")
            print(f"  Output shape: {output.shape}")
            print("  Success ✓")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_sparse_optimization()
