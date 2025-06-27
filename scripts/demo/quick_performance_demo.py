"""
Quick performance demonstration of different implementations
"""

import time

import torch

from dilated_attention_pytorch import (
    DilatedAttention,
    ImprovedDilatedAttention,
    MultiheadDilatedAttention,
)
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention


def benchmark_implementation(_name, module, inputs, warmup=3, iterations=10):
    """Quick benchmark of an implementation"""
    q, k, v = inputs

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = module(q, k, v)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iterations):
        with torch.no_grad():
            output = module(q, k, v)

    torch.cuda.synchronize()
    elapsed = (time.time() - start) / iterations * 1000  # ms

    return elapsed, output.shape


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Performance Comparison of Dilated Attention Implementations")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print()

    # Test different sequence lengths
    test_configs = [
        (2048, "Short (2K)"),
        (8192, "Medium (8K)"),
        (32768, "Long (32K)"),
        (131072, "Very Long (128K)"),
    ]

    for seq_len, desc in test_configs:
        print(f"\n{desc} Sequence Length: {seq_len:,} tokens")
        print("-" * 70)

        # Common params
        batch_size = 1
        num_heads = 8
        head_dim = 64
        embed_dim = num_heads * head_dim

        # Adjust segment lengths based on sequence
        segments = [
            min(1024, seq_len // 4),
            min(2048, seq_len // 2),
            min(4096, seq_len),
        ]
        dilation_rates = [1, 2, 4]

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        inputs = (q, k, v)

        # Multihead inputs
        q_mh = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        k_mh = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        v_mh = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        inputs_mh = (q_mh, k_mh, v_mh)

        # Test implementations
        implementations = []

        # Core implementations
        try:
            module = DilatedAttention(segments, dilation_rates, 0.0).to(device, dtype)
            time_ms, shape = benchmark_implementation(
                "DilatedAttention", module, inputs
            )
            implementations.append(
                ("DilatedAttention", time_ms, seq_len / time_ms * 1000)
            )
        except Exception as e:
            print(f"DilatedAttention failed: {e}")

        try:
            module = ImprovedDilatedAttention(segments, dilation_rates, 0.0).to(
                device, dtype
            )
            time_ms, shape = benchmark_implementation(
                "ImprovedDilatedAttention", module, inputs
            )
            implementations.append(
                ("ImprovedDilatedAttention", time_ms, seq_len / time_ms * 1000)
            )
        except Exception as e:
            print(f"ImprovedDilatedAttention failed: {e}")

        try:
            module = RingDilatedAttention(
                segments, dilation_rates, 0.0, ring_size=1
            ).to(device, dtype)
            time_ms, shape = benchmark_implementation(
                "RingDilatedAttention", module, inputs
            )
            implementations.append(
                ("RingDilatedAttention", time_ms, seq_len / time_ms * 1000)
            )
        except Exception as e:
            print(f"RingDilatedAttention failed: {e}")

        # Block sparse (only for shorter sequences)
        if seq_len <= 32768:
            try:
                sparse_config = SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=0.1,
                    block_size=min(128, seq_len // 16),
                )
                module = BlockSparseRingDilatedAttention(
                    segments, dilation_rates, sparse_config=sparse_config, ring_size=1
                ).to(device, dtype)
                time_ms, shape = benchmark_implementation(
                    "BlockSparse(10%)", module, inputs
                )
                implementations.append(
                    ("BlockSparse(10%)", time_ms, seq_len / time_ms * 1000)
                )
            except Exception as e:
                print(f"BlockSparse failed: {e}")

        # Multihead
        try:
            module = MultiheadDilatedAttention(
                embed_dim, num_heads, dilation_rates, segments, 0.0
            ).to(device, dtype)
            time_ms, shape = benchmark_implementation(
                "MultiheadDilated", module, inputs_mh
            )
            implementations.append(
                ("MultiheadDilated", time_ms, seq_len / time_ms * 1000)
            )
        except Exception as e:
            print(f"MultiheadDilated failed: {e}")

        # Sort by speed
        implementations.sort(key=lambda x: x[1])

        # Print results
        print(f"{'Implementation':<25} {'Time (ms)':<12} {'Throughput (tok/s)':<20}")
        print("-" * 70)
        for name, time_ms, throughput in implementations:
            print(f"{name:<25} {time_ms:<12.1f} {throughput:<20,.0f}")

        # Memory cleanup
        del module, q, k, v, q_mh, k_mh, v_mh
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
