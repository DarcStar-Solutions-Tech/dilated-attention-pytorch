"""
Focused benchmark to measure pattern cache overhead in Ring Attention.
"""

import time
import torch
import numpy as np
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache


def benchmark_pattern_access(iterations=1000):
    """Benchmark pattern access time for global vs local cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Configuration
    segment_lengths = [1024, 2048]
    dilation_rates = [1, 2]

    # Create models
    ring_global = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        use_pattern_cache=True,
    ).to(device)

    ring_local = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        use_pattern_cache=False,
    ).to(device)

    # Warm up caches
    dummy = torch.randn(1, 2048, 8, 32, device=device)
    _ = ring_global(dummy, dummy, dummy)
    _ = ring_local(dummy, dummy, dummy)

    print("Pattern Access Benchmark")
    print("=" * 50)

    # Benchmark local cache access
    local_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        # Access pattern from local cache
        cache_key = (1024, 2, 0, device)
        if cache_key in ring_local._dilated_indices_cache:
            _ = ring_local._dilated_indices_cache[cache_key]
        local_times.append((time.perf_counter() - start) * 1e6)  # microseconds

    # Benchmark global cache access
    global_times = []
    cache = get_global_pattern_cache()
    for _ in range(iterations):
        start = time.perf_counter()
        # Access pattern from global cache
        _ = cache.get("ring_dilated_s1024_r2_off0", target_device=device)
        global_times.append((time.perf_counter() - start) * 1e6)  # microseconds

    print(
        f"Local cache access:  {np.mean(local_times):.3f} µs (±{np.std(local_times):.3f})"
    )
    print(
        f"Global cache access: {np.mean(global_times):.3f} µs (±{np.std(global_times):.3f})"
    )
    print(f"Overhead: {np.mean(global_times) - np.mean(local_times):.3f} µs per access")

    # Test memory transfer overhead
    print("\nMemory Transfer Benchmark")
    print("=" * 50)

    # Create a pattern on CPU
    cpu_pattern = torch.arange(1024, device="cpu")

    # Benchmark CPU to GPU transfer
    transfer_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = cpu_pattern.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        transfer_times.append((time.perf_counter() - start) * 1e6)

    print(f"CPU->GPU transfer (1024 elements): {np.mean(transfer_times):.3f} µs")

    # Test with different sizes
    for size in [512, 1024, 2048, 4096, 8192]:
        cpu_tensor = torch.arange(size, device="cpu")
        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = cpu_tensor.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1e6)
        print(f"  Size {size}: {np.mean(times):.3f} µs")


def benchmark_full_forward():
    """Benchmark full forward pass with detailed timing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nFull Forward Pass Breakdown")
    print("=" * 50)

    # Create model with global cache
    clear_global_cache()
    model = RingDilatedAttentionV2(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        use_pattern_cache=True,
    ).to(device)

    # Input tensors
    batch_size = 2
    seq_len = 2048
    num_heads = 16
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Profile forward pass
    if device == "cuda":
        torch.cuda.synchronize()

    # Warm up
    for _ in range(5):
        _ = model(q, k, v)

    # Detailed timing
    times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = model(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    print(f"Total forward pass: {np.mean(times):.3f} ms (±{np.std(times):.3f})")

    # Check cache stats
    cache = get_global_pattern_cache()
    stats = cache.get_stats()
    print("\nCache statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")


if __name__ == "__main__":
    benchmark_pattern_access()
    benchmark_full_forward()
