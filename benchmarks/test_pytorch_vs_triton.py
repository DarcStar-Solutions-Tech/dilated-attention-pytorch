#!/usr/bin/env python3
"""Compare PyTorch vs Triton backend performance."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,  # Test dense attention first
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    print("Comparing PyTorch vs Triton backend")
    print("=" * 60)

    for seq_len in [2048, 4096]:
        x = torch.randn(1, seq_len, 768, device=device)

        print(f"\nSequence length: {seq_len}")

        # Force PyTorch backend
        original_triton = module._triton_available
        module._triton_available = False

        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=True)

            # Time PyTorch with Hilbert
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) * 1000

        # Re-enable Triton
        module._triton_available = original_triton

        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=True)

            # Time Triton with Hilbert
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) * 1000

        print(f"  PyTorch backend: {pytorch_time:.2f}ms")
        print(f"  Triton backend: {triton_time:.2f}ms")
        print(f"  Triton speedup: {pytorch_time / triton_time:.2f}x")

        # Also test without Hilbert
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            no_hilbert_time = (time.perf_counter() - start) * 1000

        print(f"  No Hilbert: {no_hilbert_time:.2f}ms")
        print(f"  Hilbert overhead (Triton): {triton_time / no_hilbert_time:.2f}x")


if __name__ == "__main__":
    main()
