#!/usr/bin/env python3
"""Clean backend comparison test."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def benchmark_config(module, x, backend, use_hilbert):
    """Benchmark a specific configuration."""
    original_triton = module._triton_available

    # Set backend
    if backend == "pytorch":
        module._triton_available = False
    else:
        module._triton_available = original_triton

    with torch.no_grad():
        # Warmup
        _ = module(x, use_hilbert=use_hilbert)

        # Time
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = module(x, use_hilbert=use_hilbert)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

    # Restore
    module._triton_available = original_triton

    return elapsed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Backend Performance Comparison")
    print("=" * 80)
    print(f"{'Seq Len':<10} {'Config':<30} {'Time (ms)':<15} {'Speedup':<15}")
    print("-" * 80)

    for seq_len in [1024, 2048, 4096, 8192]:
        # Create module with high threshold so we test actual Hilbert
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=128,
            dilation_rate=1,
            dropout=0.0,
            hilbert_threshold=100,  # Low threshold to force Hilbert
        ).to(device)

        x = torch.randn(1, seq_len, 768, device=device)

        # Test configurations
        configs = [
            ("PyTorch + No Hilbert", "pytorch", False),
            ("PyTorch + Hilbert", "pytorch", True),
            ("Triton + No Hilbert", "triton", False),
            ("Triton + Hilbert", "triton", True),
        ]

        baseline = None
        for config_name, backend, use_hilbert in configs:
            time_ms = benchmark_config(module, x, backend, use_hilbert)

            if baseline is None:
                baseline = time_ms
                speedup = "1.00x (baseline)"
            else:
                speedup = f"{baseline / time_ms:.2f}x"

            print(f"{seq_len:<10} {config_name:<30} {time_ms:<15.2f} {speedup:<15}")

        print()  # Empty line between sequence lengths


if __name__ == "__main__":
    main()
