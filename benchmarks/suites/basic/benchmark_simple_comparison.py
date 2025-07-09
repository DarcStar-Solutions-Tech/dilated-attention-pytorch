#!/usr/bin/env python3
"""
Simple benchmark comparing implementations.
"""

import time
import torch
import gc
from datetime import datetime


def time_attention(name, attention_fn, num_runs=5):
    """Time an attention function."""
    times = []

    # Warmup
    for _ in range(2):
        attention_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time runs
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        attention_fn()

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time * 1000  # Convert to ms


def main():
    print("=== Simple Performance Comparison ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    # Test configurations
    configs = [
        {"seq_len": 2048, "batch": 4, "heads": 8, "dim": 64},
        {"seq_len": 4096, "batch": 2, "heads": 8, "dim": 64},
        {"seq_len": 8192, "batch": 1, "heads": 8, "dim": 64},
    ]

    print("\nRunning benchmarks...")
    print("-" * 60)

    for config in configs:
        seq_len = config["seq_len"]
        batch = config["batch"]
        heads = config["heads"]
        dim = config["dim"]

        print(f"\nConfig: seq_len={seq_len}, batch={batch}, heads={heads}, dim={dim}")

        # Create tensors
        q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # 1. Standard PyTorch attention (baseline)
        def pytorch_attention():
            with torch.no_grad():
                # Reshape for torch.nn.functional
                b, s, h, d = q.shape
                q_reshaped = q.transpose(1, 2).reshape(b * h, s, d)
                k_reshaped = k.transpose(1, 2).reshape(b * h, s, d)
                v_reshaped = v.transpose(1, 2).reshape(b * h, s, d)

                scores = torch.bmm(q_reshaped, k_reshaped.transpose(-2, -1)) / (d**0.5)
                attn = torch.softmax(scores, dim=-1)
                out = torch.bmm(attn, v_reshaped)

                return out.reshape(b, h, s, d).transpose(1, 2)

        pytorch_time = time_attention("PyTorch", pytorch_attention)
        print(f"  PyTorch Attention: {pytorch_time:.1f}ms")

        # 2. Block-Sparse attention
        try:
            from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
                BlockSparseRingDilatedAttention,
                SparsePatternConfig,
            )

            sparse_config = SparsePatternConfig(
                pattern_type="local_window",
                sparsity_ratio=0.1,  # 90% sparse
                block_size=64,
                window_size=256,
            )

            # Determine segment lengths based on seq_len
            if seq_len == 2048:
                segment_lengths = [1024, 2048]
            elif seq_len == 4096:
                segment_lengths = [2048, 4096]
            else:  # 8192
                segment_lengths = [4096, 8192]

            model = BlockSparseRingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=[1, 2],
                sparse_config=sparse_config,
                device=device,
            ).eval()

            def sparse_attention():
                with torch.no_grad():
                    return model(q, k, v)

            sparse_time = time_attention("Block-Sparse", sparse_attention)
            speedup = pytorch_time / sparse_time
            print(
                f"  Block-Sparse (90%): {sparse_time:.1f}ms (Speedup: {speedup:.2f}x)"
            )

        except Exception as e:
            print(f"  Block-Sparse: Failed - {e}")

        # Memory usage
        print(f"  Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

    print("\n" + "=" * 60)
    print("Benchmark complete!")


if __name__ == "__main__":
    main()
