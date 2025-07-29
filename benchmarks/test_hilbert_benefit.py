#!/usr/bin/env python3
"""Quick test to show Hilbert ordering behavior."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module with sparse pattern
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,  # Sparse pattern
        dropout=0.0,
    ).to(device)

    print("Testing Hilbert ordering with dilated attention")
    print(f"Dilation rate: {module.dilation_rate}")
    print(f"Backend: {'Triton' if module._triton_available else 'PyTorch'}")
    print()

    # Test a few sequence lengths
    for seq_len in [1024, 2048, 4096]:
        x = torch.randn(1, seq_len, 768, device=device)

        # Warmup
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
            _ = module(x, use_hilbert=True)

        # Time standard
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) * 1000

        # Time Hilbert
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) * 1000

        speedup = std_time / hilbert_time
        print(
            f"Seq={seq_len}: Standard={std_time:.1f}ms, Hilbert={hilbert_time:.1f}ms, Speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    main()
