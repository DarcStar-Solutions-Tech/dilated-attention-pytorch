#!/usr/bin/env python3
"""Investigate the performance dip at 8K sequences."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def detailed_benchmark(seq_len, num_runs=10):
    """Detailed benchmark with profiling information."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=seq_len + 1,
    ).to(device)

    x = torch.randn(1, seq_len, 768, device=device)

    # Get memory stats before
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Test both implementations
    results = {}

    for name, use_triton in [("PyTorch", False), ("Fused", True)]:
        # Configure module
        module._triton_available = use_triton
        module._fused_kernels_available = use_triton

        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = module(x, use_hilbert=False)

            torch.cuda.synchronize()

            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            # Time
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / num_runs * 1000

            results[name] = {"time": elapsed, "memory": peak_memory}

    return results


def analyze_block_configurations():
    """Analyze how block configurations affect performance."""
    print("Block Configuration Analysis")
    print("=" * 80)

    # Check what block sizes are being used
    seq_lengths = [4096, 8192, 12288, 16384]
    compute_capability = torch.cuda.get_device_capability()[0]

    print(f"GPU Compute Capability: {compute_capability}")
    print()

    for seq_len in seq_lengths:
        # Determine block configuration (matching the kernel logic)
        if compute_capability < 7:  # Pascal
            if seq_len <= 2048:
                BLOCK_M, BLOCK_N = 32, 32
            elif seq_len <= 4096:
                BLOCK_M, BLOCK_N = 64, 64
            elif seq_len <= 8192:
                BLOCK_M, BLOCK_N = 64, 64  # Same as 4K
            else:  # 8K-16K
                BLOCK_M, BLOCK_N = 64, 64  # Still same
        else:  # Volta+
            if seq_len <= 2048:
                BLOCK_M, BLOCK_N = 64, 64
            elif seq_len <= 4096:
                BLOCK_M, BLOCK_N = 128, 128
            elif seq_len <= 8192:
                BLOCK_M, BLOCK_N = 128, 128  # Same as 4K
            else:  # 8K-16K
                BLOCK_M, BLOCK_N = 128, 128  # Still same

        num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
        num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
        total_blocks = num_blocks_m * num_blocks_n

        print(f"Sequence {seq_len}:")
        print(f"  Block size: {BLOCK_M}x{BLOCK_N}")
        print(f"  Grid size: {num_blocks_m}x{num_blocks_n} = {total_blocks} blocks")
        print(f"  Blocks per SM (assuming 20 SMs): {total_blocks / 20:.1f}")
        print()


def test_memory_bandwidth():
    """Test if memory bandwidth is the limiting factor."""
    print("\nMemory Bandwidth Analysis")
    print("=" * 80)

    seq_lengths = [4096, 8192, 12288, 16384]

    for seq_len in seq_lengths:
        # Calculate data movement
        batch_size = 1
        num_heads = 12
        head_dim = 64

        # QKV tensor sizes
        qkv_size = 3 * batch_size * seq_len * num_heads * head_dim * 2  # float16

        # Attention computation data movement (simplified)
        # Each block loads Q once, K and V multiple times
        block_size = 128
        num_blocks = seq_len // block_size

        # Standard approach
        standard_loads = qkv_size + (num_blocks * qkv_size)  # More K,V loads

        # Fused approach
        fused_loads = qkv_size * 0.7  # Better cache utilization

        print(f"Sequence {seq_len}:")
        print(f"  QKV size: {qkv_size / 1024 / 1024:.1f} MB")
        print(f"  Standard loads: {standard_loads / 1024 / 1024:.1f} MB")
        print(f"  Fused loads: {fused_loads / 1024 / 1024:.1f} MB")
        print(f"  Reduction: {(1 - fused_loads / standard_loads) * 100:.1f}%")
        print()


def test_different_configurations():
    """Test performance with different batch sizes and head counts."""
    print("\nConfiguration Sensitivity Test")
    print("=" * 80)

    seq_len = 8192
    configs = [
        (1, 12),  # Original
        (2, 12),  # Larger batch
        (1, 8),  # Fewer heads
        (1, 16),  # More heads
    ]

    for batch_size, num_heads in configs:
        print(f"\nBatch={batch_size}, Heads={num_heads}:")

        device = "cuda"
        module = HilbertAttention(
            hidden_dim=768,
            num_heads=num_heads,
            segment_size=128,
            dilation_rate=1,
            dropout=0.0,
            hilbert_threshold=seq_len + 1,
        ).to(device)

        x = torch.randn(batch_size, seq_len, 768, device=device)

        # Quick benchmark
        for name, use_triton in [("PyTorch", False), ("Fused", True)]:
            module._triton_available = use_triton
            module._fused_kernels_available = use_triton

            with torch.no_grad():
                # Warmup
                for _ in range(2):
                    _ = module(x, use_hilbert=False)

                # Time
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    _ = module(x, use_hilbert=False)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / 5 * 1000

                print(f"  {name}: {elapsed:.2f}ms")


def main():
    print("Investigating 8K Performance Dip")
    print("=" * 80)

    # First, confirm the performance pattern
    print("\nPerformance Pattern Confirmation:")
    print("-" * 40)

    for seq_len in [4096, 6144, 8192, 10240, 12288]:
        results = detailed_benchmark(seq_len, num_runs=5)

        pytorch_time = results["PyTorch"]["time"]
        fused_time = results["Fused"]["time"]
        speedup = pytorch_time / fused_time

        print(
            f"Seq {seq_len:5d}: PyTorch={pytorch_time:7.2f}ms, "
            f"Fused={fused_time:7.2f}ms, Speedup={speedup:.2f}x"
        )

    print("\n" + "=" * 80)

    # Analyze block configurations
    analyze_block_configurations()

    # Test memory bandwidth
    test_memory_bandwidth()

    # Test different configurations
    test_different_configurations()

    # Hypothesis
    print("\n\nHYPOTHESIS")
    print("=" * 80)
    print("""
The 8K performance dip is likely due to:

1. **Cache Thrashing Threshold**: 8K sequences might hit a critical point where
   the working set exceeds L2 cache but isn't large enough to benefit from the
   streaming patterns that help at 12K+.

2. **Block Configuration**: The block sizes remain the same from 4K to 16K,
   but 8K might be at an awkward size where:
   - Too large for optimal cache reuse (unlike 4K)
   - Too small for streaming benefits (unlike 12K+)

3. **GPU Occupancy**: 8K might create a grid size that doesn't map well to
   the GPU's SM count, causing load imbalancing.

4. **Memory Access Pattern**: The stride pattern at 8K might conflict with
   the GPU's memory coalescing units.

Potential fixes:
- Use different block sizes specifically for 8K
- Implement adaptive tiling based on sequence length
- Consider Flash Attention style algorithm for 8K+
""")


if __name__ == "__main__":
    main()
