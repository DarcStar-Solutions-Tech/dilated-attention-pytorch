#!/usr/bin/env python3
"""
Focused benchmark for BlockSparseRingDilatedAttention with extreme sequences.
"""

import gc
import time
import torch
from dilated_attention_pytorch import BlockSparseRingDilatedAttention


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_block_sparse_scaling():
    """Test BlockSparseRingDilatedAttention with various sequence lengths."""

    print("=" * 80)
    print("BLOCK SPARSE RING ATTENTION - EXTREME SEQUENCE BENCHMARK")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Test configurations
    configs = [
        # (seq_len, sparsity_ratio, description)
        (4096, 0.9, "4K tokens, 90% sparse"),
        (8192, 0.95, "8K tokens, 95% sparse"),
        (16384, 0.95, "16K tokens, 95% sparse"),
        (32768, 0.98, "32K tokens, 98% sparse"),
        (65536, 0.99, "64K tokens, 99% sparse"),
        (131072, 0.995, "128K tokens, 99.5% sparse"),
    ]

    results = []

    for seq_len, sparsity_ratio, description in configs:
        print(f"\n{description}:")
        clear_memory()

        try:
            # Create attention module
            attention = BlockSparseRingDilatedAttention(
                segment_lengths=[seq_len // 4, seq_len // 2, seq_len],
                dilation_rates=[1, 2, 4],
                sparsity_ratio=sparsity_ratio,
                enable_memory_pool=True,
                lightweight_pool=False,  # Full pool for extreme sequences
            )

            # Create inputs (use float16 for memory efficiency)
            batch_size = 1
            num_heads = 8
            head_dim = 64
            shape = (batch_size, seq_len, num_heads, head_dim)

            q = torch.randn(shape, device=device, dtype=torch.float16)
            k = torch.randn(shape, device=device, dtype=torch.float16)
            v = torch.randn(shape, device=device, dtype=torch.float16)

            # Warmup
            _ = attention(q, k, v)

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Time forward pass
            start = time.perf_counter()
            output = attention(q, k, v)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            time_ms = (end - start) * 1000

            # Get memory usage
            if device.type == "cuda":
                memory_gb = torch.cuda.memory_allocated() / 1024**3
            else:
                memory_gb = 0

            # Calculate effective compute
            total_elements = seq_len * seq_len
            sparse_elements = total_elements * (1 - sparsity_ratio)

            print("  ✓ Success!")
            print(f"  Time: {time_ms:.1f}ms")
            print(f"  Memory: {memory_gb:.2f}GB")
            print(f"  Effective compute: {sparse_elements / 1e6:.1f}M elements")
            print(f"  Throughput: {sparse_elements / time_ms / 1e6:.1f}M elements/ms")

            results.append(
                {
                    "seq_len": seq_len,
                    "sparsity": sparsity_ratio,
                    "time_ms": time_ms,
                    "memory_gb": memory_gb,
                    "success": True,
                }
            )

            # Cleanup
            del attention, q, k, v, output
            clear_memory()

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append(
                {
                    "seq_len": seq_len,
                    "sparsity": sparsity_ratio,
                    "time_ms": None,
                    "memory_gb": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    successful = [r for r in results if r["success"]]
    if successful:
        max_seq = max(r["seq_len"] for r in successful)
        print(f"\nMaximum sequence length achieved: {max_seq:,} tokens")

        # Find optimal sparsity
        if len(successful) > 1:
            best = min(successful, key=lambda r: r["time_ms"])
            print(
                f"Best performance: {best['seq_len']:,} tokens at {best['sparsity'] * 100:.0f}% sparsity"
            )

    # Scaling analysis
    print("\nScaling characteristics:")
    for i in range(1, len(successful)):
        prev = successful[i - 1]
        curr = successful[i]
        seq_scale = curr["seq_len"] / prev["seq_len"]
        time_scale = curr["time_ms"] / prev["time_ms"]
        mem_scale = curr["memory_gb"] / prev["memory_gb"]

        print(
            f"  {prev['seq_len']} → {curr['seq_len']}: "
            f"{seq_scale:.1f}x sequence, "
            f"{time_scale:.1f}x time, "
            f"{mem_scale:.1f}x memory"
        )

    print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    test_block_sparse_scaling()
