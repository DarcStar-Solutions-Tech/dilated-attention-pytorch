#!/usr/bin/env python3
"""Find the crossover point where Hilbert ordering becomes beneficial."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def benchmark_sequence_length(module, seq_len, batch_size=1, num_iterations=3):
    """Benchmark a specific sequence length."""
    device = module.qkv_proj.weight.device
    x = torch.randn(batch_size, seq_len, module.hidden_dim, device=device)

    # Warmup
    with torch.no_grad():
        _ = module(x, use_hilbert=False)
        _ = module(x, use_hilbert=True)

    # Benchmark standard
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = module(x, use_hilbert=False)
    torch.cuda.synchronize()
    std_time = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark Hilbert
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = module(x, use_hilbert=True)
    torch.cuda.synchronize()
    hilbert_time = (time.perf_counter() - start) / num_iterations * 1000

    return std_time, hilbert_time


def find_crossover_point():
    """Find sequence length where Hilbert ordering becomes beneficial."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test with different configurations
    configs = [
        {"segment_size": 128, "dilation_rate": 1, "desc": "Dense attention"},
        {"segment_size": 128, "dilation_rate": 4, "desc": "Sparse attention (4x)"},
        {"segment_size": 256, "dilation_rate": 8, "desc": "Very sparse (8x)"},
    ]

    for config in configs:
        print(
            f"\n{config['desc']} (segment={config['segment_size']}, dilation={config['dilation_rate']})"
        )
        print("=" * 70)
        print(
            f"{'Seq Length':<12} {'Standard (ms)':<15} {'Hilbert (ms)':<15} {'Speedup':<10} {'Winner':<10}"
        )
        print("-" * 70)

        module = HilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=config["segment_size"],
            dilation_rate=config["dilation_rate"],
            dropout=0.0,
        ).to(device)

        # Test sequence lengths from 512 to 16K
        seq_lengths = [512, 1024, 2048, 4096, 8192]
        if device == "cuda":
            seq_lengths.extend([16384])

        crossover_found = False
        for seq_len in seq_lengths:
            try:
                # Use smaller batch size for longer sequences
                batch_size = 1 if seq_len >= 8192 else 2

                std_time, hilbert_time = benchmark_sequence_length(
                    module, seq_len, batch_size, num_iterations=3
                )

                speedup = std_time / hilbert_time
                winner = "Hilbert" if speedup > 1.0 else "Standard"

                print(
                    f"{seq_len:<12} {std_time:<15.2f} {hilbert_time:<15.2f} {speedup:<10.2f} {winner:<10}"
                )

                if speedup > 1.0 and not crossover_found:
                    crossover_found = True
                    print(f"  --> Crossover point found at {seq_len} tokens!")

            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:<12} {'OOM':<15} {'OOM':<15} {'N/A':<10} {'N/A':<10}")
                torch.cuda.empty_cache()
                break

        if not crossover_found:
            print(f"  --> No crossover found up to {seq_lengths[-1]} tokens")


if __name__ == "__main__":
    find_crossover_point()
