#!/usr/bin/env python3
"""
Benchmark core dilated attention implementations.
"""

import time
import torch
import gc
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch import create_multihead_dilated_attention
from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


def benchmark_forward_pass(name, model, inputs, num_runs=10, warmup=3):
    """Benchmark forward pass with proper timing."""
    q, k, v = inputs
    device = q.device

    # Move model to same device
    model = model.to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(q, k, v)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Clear cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    # Time runs
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            output = model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    # Memory usage
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        mem_used = (peak_mem - start_mem) / 1024**2  # MB
    else:
        mem_used = 0

    return avg_time, mem_used, output.shape


def main():
    """Run benchmarks."""
    print("=== Dilated Attention Performance Benchmark ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    # Test configurations
    batch_size = 2
    embed_dim = 512
    num_heads = 8

    test_configs = [
        (2048, [512, 1024], [1, 2]),
        (4096, [1024, 2048], [1, 2]),
        (8192, [2048, 4096], [1, 2]),
    ]

    results = []

    for seq_len, segment_lengths, dilation_rates in test_configs:
        print(f"\n--- Sequence Length: {seq_len:,} ---")

        # Create inputs (batch, seq_len, embed_dim)
        q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Test implementations
        implementations = []

        # 1. Improved Dilated Attention
        try:
            model = create_multihead_dilated_attention(
                "improved",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
            )
            implementations.append(("Improved Dilated", model))
        except Exception as e:
            print(f"Failed to create Improved Dilated: {e}")

        # 2. Block-Sparse Ring Attention
        try:
            # For BlockSparse, we need 4D input (batch, seq_len, num_heads, head_dim)
            head_dim = embed_dim // num_heads
            q_4d = q.view(batch_size, seq_len, num_heads, head_dim)
            k_4d = k.view(batch_size, seq_len, num_heads, head_dim)
            v_4d = v.view(batch_size, seq_len, num_heads, head_dim)

            sparse_config = SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.1,  # 90% sparse
                block_size=128,
            )

            model = BlockSparseRingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                sparse_config=sparse_config,
                device=device,
            )

            # Test with 4D inputs
            name = "Block-Sparse (90%)"
            time_taken, mem_used, output_shape = benchmark_forward_pass(
                name, model, (q_4d, k_4d, v_4d), num_runs=5
            )

            print(f"{name}:")
            print(f"  Time: {time_taken * 1000:.1f}ms")
            print(f"  Memory: {mem_used:.1f}MB")
            print(
                f"  Speedup: {results[-1]['improved_time'] / time_taken:.2f}x"
                if results and "improved_time" in results[-1]
                else ""
            )

            # Don't add to implementations list since it uses different input format

        except Exception as e:
            print(f"Failed to test Block-Sparse: {e}")

        # Benchmark regular implementations
        result = {"seq_len": seq_len}

        for name, model in implementations:
            try:
                time_taken, mem_used, output_shape = benchmark_forward_pass(
                    name, model, (q, k, v)
                )

                print(f"\n{name}:")
                print(f"  Time: {time_taken * 1000:.1f}ms")
                print(f"  Memory: {mem_used:.1f}MB")
                print(f"  Output: {output_shape}")

                if name == "Improved Dilated":
                    result["improved_time"] = time_taken
                    result["improved_mem"] = mem_used

            except Exception as e:
                print(f"\n{name}: Failed - {e}")

        results.append(result)

    # Summary
    print("\n=== Summary ===")
    print("Sequence lengths tested:", [r["seq_len"] for r in results])

    # Test maximum sequence length
    if device.type == "cuda":
        print("\n=== Maximum Sequence Length Test ===")
        test_lengths = [16384, 32768, 65536, 131072]

        for seq_len in test_lengths:
            try:
                # Use appropriate segment lengths
                if seq_len <= 16384:
                    segment_lengths = [4096, 8192]
                elif seq_len <= 32768:
                    segment_lengths = [8192, 16384]
                elif seq_len <= 65536:
                    segment_lengths = [16384, 32768]
                else:
                    segment_lengths = [32768, 65536]

                q = torch.randn(1, seq_len, embed_dim, device=device, dtype=dtype)

                model = create_multihead_dilated_attention(
                    "improved",
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=[1, 2],
                    dropout=0.0,
                )

                with torch.no_grad():
                    _ = model(q, q, q)

                print(f"✓ {seq_len:,} tokens: Success")

                del q, model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ {seq_len:,} tokens: {type(e).__name__}")
                break


if __name__ == "__main__":
    main()
