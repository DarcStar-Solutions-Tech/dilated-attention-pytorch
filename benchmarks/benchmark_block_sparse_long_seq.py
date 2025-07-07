#!/usr/bin/env python3
"""
Benchmark block-sparse on long sequences where it provides benefits.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def estimate_memory_saved(seq_len, sparsity_ratio, num_heads=8, head_dim=64):
    """Estimate memory saved by sparsity."""
    # Full attention matrix per head: seq_len x seq_len
    full_attention_elements = seq_len * seq_len * num_heads
    sparse_attention_elements = full_attention_elements * sparsity_ratio

    # Each element is float16 (2 bytes)
    memory_saved_mb = (
        (full_attention_elements - sparse_attention_elements) * 2 / 1024**2
    )

    return memory_saved_mb


def main():
    """Test block-sparse on sequences where it matters."""
    print("Block-Sparse Long Sequence Benefits")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Test very long sequences with small batch size
    configs = [
        {"seq_len": 8192, "batch_size": 1},
        {"seq_len": 16384, "batch_size": 1},
        {"seq_len": 32768, "batch_size": 1},
    ]

    print("\nTesting where block-sparse shines: long sequences")
    print("-" * 60)

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\nSequence length: {seq_len}")

        # Skip if too large for available memory
        estimated_dense_memory_gb = (seq_len**2 * 8 * 2) / 1024**3  # Rough estimate
        if estimated_dense_memory_gb > 8:  # Assuming 8GB available
            print(f"  Skipping - dense would need ~{estimated_dense_memory_gb:.1f}GB")

            # Show what sparse could achieve
            for sparsity in [0.1, 0.05, 0.01]:
                saved_mb = estimate_memory_saved(seq_len, sparsity)
                print(
                    f"  → {int((1 - sparsity) * 100)}% sparse would save {saved_mb:.0f}MB"
                )
            continue

        # Create inputs
        num_heads = 8
        head_dim = 64

        try:
            q = torch.randn(
                batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Test 95% sparse
            print("\n  95% Sparse:")
            model = create_block_sparse_attention(
                variant="base",
                segment_lengths=[2048, 4096],
                dilation_rates=[1, 2],
                sparsity_ratio=0.05,
                block_size=128,  # Larger blocks for long sequences
            ).to(device=device, dtype=dtype)

            # Time one forward pass
            torch.cuda.synchronize()
            start = time.time()
            output = model(q, k, v)
            torch.cuda.synchronize()
            forward_time = (time.time() - start) * 1000

            print(f"    Forward time: {forward_time:.1f}ms")
            print(
                f"    Theoretical memory saved: {estimate_memory_saved(seq_len, 0.05):.0f}MB"
            )

            del model, output
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print("  Out of memory - sequence too long for dense baseline")
        except Exception as e:
            print(f"  Error: {e}")

    # Show scaling benefits
    print("\n" + "=" * 60)
    print("Theoretical Scaling Benefits:")
    print("-" * 60)

    print("\nMemory complexity:")
    print("  Dense attention: O(n²)")
    print("  Block-sparse: O(n × sparsity_ratio)")

    print("\nFor 1M token sequence:")
    dense_memory_gb = (1_000_000**2 * 8 * 2) / 1024**3
    sparse_99_memory_gb = dense_memory_gb * 0.01
    print(f"  Dense would need: {dense_memory_gb:.0f}GB")
    print(f"  99% sparse needs: {sparse_99_memory_gb:.1f}GB")
    print(f"  Reduction: {(1 - sparse_99_memory_gb / dense_memory_gb) * 100:.0f}%")

    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
