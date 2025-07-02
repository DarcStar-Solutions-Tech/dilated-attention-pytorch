# \!/usr/bin/env python3
"""
Quick focused benchmark comparing original vs optimized hybrid implementations.
"""

import torch
import time
import gc

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid as HybridOriginal,
)
from dilated_attention_pytorch.ring_dilated_attention_hybrid_optimized import (
    RingDilatedAttentionHybridOptimized as HybridOptimized,
)


def benchmark_single_config(model_class, model_name, seq_len=4096):
    """Benchmark a single configuration."""
    device = torch.device("cuda")

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    model = model_class(
        segment_lengths=[512, 1024],
        dilation_rates=[1, 2],
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=torch.float32,
        enable_memory_pool=True,
        use_flash_attention=True,
        use_pattern_cache=True,
    )

    # Create inputs
    batch_size = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(2):
        with torch.no_grad():
            _ = model(q, k, v)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output = model(q, k, v)

        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    throughput = seq_len / avg_time

    print(f"{model_name}:")
    print(f"  Time: {avg_time * 1000:.1f} ms")
    print(f"  Memory: {peak_memory:.0f} MB")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")
    print(f"  Output shape: {output.shape}")

    # Cleanup
    del q, k, v, output, model
    gc.collect()
    torch.cuda.empty_cache()

    return avg_time, peak_memory


def main():
    print("=== Quick Hybrid Optimization Comparison ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Testing with 4096 sequence length\n")

    # Test original
    print("Testing Original Implementation...")
    try:
        orig_time, orig_mem = benchmark_single_config(HybridOriginal, "Original")
    except Exception as e:
        print(f"Original failed: {e}")
        orig_time, orig_mem = None, None

    print("\nTesting Optimized Implementation...")
    try:
        opt_time, opt_mem = benchmark_single_config(HybridOptimized, "Optimized")
    except Exception as e:
        print(f"Optimized failed: {e}")
        opt_time, opt_mem = None, None

    # Compare
    if orig_time and opt_time:
        print("\n" + "=" * 40)
        print("COMPARISON:")
        print(f"Speed improvement: {orig_time / opt_time:.2f}x faster")
        print(f"Memory improvement: {(1 - opt_mem / orig_mem) * 100:.1f}% less memory")

    # Test multi-GPU if available
    if torch.cuda.device_count() > 1:
        print("\n" + "=" * 40)
        print("Testing Multi-GPU (2 GPUs)...")

        # Create distributed model
        _ = HybridOptimized(
            segment_lengths=[512, 1024],
            dilation_rates=[1, 2],
            dropout=0.0,
            ring_size=2,
            device=torch.device("cuda:0"),
            dtype=torch.float32,
            enable_memory_pool=True,
        )

        # Note: This would need proper distributed setup
        print("Multi-GPU testing requires torchrun")


if __name__ == "__main__":
    main()
