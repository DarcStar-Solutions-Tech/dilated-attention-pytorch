#!/usr/bin/env python3
"""Test Hilbert threshold behavior."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module with threshold of 1024
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    print(f"Testing Hilbert threshold behavior (threshold={module.hilbert_threshold})")
    print("=" * 60)

    # Test different sequence lengths
    for seq_len in [512, 1024, 1536, 2048, 4096, 8192]:
        x = torch.randn(1, seq_len, 768, device=device)

        # Check what mapping is used
        padded_len = seq_len if seq_len % 128 == 0 else seq_len + (128 - seq_len % 128)
        will_use_hilbert = padded_len > module.hilbert_threshold

        print(f"\nSeq={seq_len} (padded={padded_len}):")
        print(f"  Will use Hilbert: {will_use_hilbert}")

        # Get the mapping to see what it looks like
        mapping = module._get_hilbert_mapping(padded_len, device)

        # Check if it's identity
        is_identity = torch.equal(
            mapping, torch.arange(padded_len, device=device, dtype=torch.int32)
        )
        print(f"  Is identity mapping: {is_identity}")

        # Benchmark
        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=True)

            # Time standard
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            std_time = (time.perf_counter() - start) * 1000

            # Time with Hilbert flag (will respect threshold)
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            hilbert_time = (time.perf_counter() - start) * 1000

        print(f"  Standard: {std_time:.2f}ms")
        print(f"  Hilbert: {hilbert_time:.2f}ms")
        print(f"  Speedup: {std_time / hilbert_time:.2f}x")


if __name__ == "__main__":
    main()
