#!/usr/bin/env python3
"""
Benchmark different attention backends to show performance improvements.
"""

import torch
import time
import numpy as np
from typing import Dict

from dilated_attention_pytorch import RingDilatedAttentionV2Flash
from dilated_attention_pytorch.utils import flash_attention_forward


def benchmark_backend(
    backend: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark a specific attention backend."""

    # Create tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = flash_attention_forward(q, k, v, backend=backend)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = flash_attention_forward(q, k, v, backend=backend)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def main():
    """Run attention backend benchmarks."""
    print("Attention Backend Benchmark")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA required for benchmarking")
        return

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Test configurations
    configs = [
        {"batch_size": 1, "seq_len": 512, "num_heads": 8, "head_dim": 64},
        {"batch_size": 2, "seq_len": 1024, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_len": 2048, "num_heads": 8, "head_dim": 64},
        {"batch_size": 1, "seq_len": 4096, "num_heads": 8, "head_dim": 64},
    ]

    # Backends to test
    backends = ["standard", "sdpa", "xformers"]

    for config in configs:
        print(
            f"\n\nConfiguration: batch={config['batch_size']}, seq_len={config['seq_len']}, "
            f"heads={config['num_heads']}, head_dim={config['head_dim']}"
        )
        print("-" * 60)

        results = {}
        for backend in backends:
            try:
                stats = benchmark_backend(backend=backend, device=device, **config)
                results[backend] = stats
                print(
                    f"{backend:12s}: {stats['mean_ms']:8.2f} ± {stats['std_ms']:5.2f} ms"
                )
            except Exception as e:
                print(f"{backend:12s}: Failed - {e}")

        # Calculate speedups
        if "standard" in results:
            print("\nSpeedups vs standard:")
            standard_time = results["standard"]["mean_ms"]
            for backend, stats in results.items():
                if backend != "standard":
                    speedup = standard_time / stats["mean_ms"]
                    print(f"  {backend:10s}: {speedup:.2f}x")

    # Test with RingDilatedAttentionV2Flash
    print("\n\n" + "=" * 80)
    print("Ring Dilated Attention Benchmark")
    print("=" * 80)

    segment_lengths = [1024, 2048]
    dilation_rates = [1, 2]

    # Create model
    model = RingDilatedAttentionV2Flash(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        device=device,
        dtype=torch.float32,
        use_flash_attention=True,
    )

    print(f"\nUsing backend: {model.flash_backend}")

    # Benchmark
    batch_size = 1
    seq_len = 4096
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(q, k, v)

    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            _ = model(q, k, v)

        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    print("\nRing Dilated Attention Performance:")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")


if __name__ == "__main__":
    main()
