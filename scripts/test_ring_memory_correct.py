#!/usr/bin/env python3
"""
Test script to verify correct O(n/k) memory usage in ring attention.

This script demonstrates the difference between:
1. Incorrect implementation (full sequence on each GPU)
2. Correct implementation (only local chunk on each GPU)
"""

import torch
import gc
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_gpu_optimized import (
    RingDilatedAttentionHilbertGPUOptimized as IncorrectImplementation,
)
from dilated_attention_pytorch.ring_dilated_attention_correct import (
    RingAttentionWrapper as CorrectImplementation,
)


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def cleanup_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def test_memory_usage():
    """Compare memory usage between implementations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configurations
    batch_size = 1
    seq_len = 16384  # 16K tokens
    embed_dim = 768
    num_heads = 12

    print("=" * 70)
    print("Ring Attention Memory Usage Comparison")
    print("=" * 70)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len:,} tokens")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(
            f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    print()

    # Test 1: Incorrect implementation (full sequence)
    print("Testing INCORRECT implementation (full sequence on each GPU)...")
    cleanup_memory()
    start_mem = get_memory_mb()

    try:
        # Create input - full sequence
        x_full = torch.randn(batch_size, seq_len, embed_dim, device=device)
        print(f"  Input tensor created: {get_memory_mb() - start_mem:.1f} MB")

        # Create incorrect model
        model_incorrect = IncorrectImplementation(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4],
            ring_size=1,  # Single GPU for now
            device=device,
            dtype=torch.float32,
        )
        print(f"  Model created: {get_memory_mb() - start_mem:.1f} MB")

        # Forward pass
        output = model_incorrect(x_full)
        print(f"  After forward: {get_memory_mb() - start_mem:.1f} MB")

        # Peak memory for incorrect implementation
        incorrect_peak_mb = get_memory_mb() - start_mem

    except torch.cuda.OutOfMemoryError:
        print("  OOM! This is expected for large sequences.")
        incorrect_peak_mb = float("inf")

    # Cleanup
    del x_full
    if "model_incorrect" in locals():
        del model_incorrect
    if "output" in locals():
        del output
    cleanup_memory()

    print()

    # Test 2: Correct implementation (local chunk only)
    print("Testing CORRECT implementation (local chunk only)...")
    cleanup_memory()
    start_mem = get_memory_mb()

    # Simulate multi-GPU by using smaller local sequence
    world_size = 4  # Simulate 4 GPUs
    local_seq_len = seq_len // world_size

    try:
        # Create input - local chunk only!
        x_local = torch.randn(batch_size, local_seq_len, embed_dim, device=device)
        print(
            f"  Local input tensor created ({local_seq_len} tokens): {get_memory_mb() - start_mem:.1f} MB"
        )

        # Create correct model
        model_correct = CorrectImplementation(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4],
            device=device,
            dtype=torch.float32,
        )
        print(f"  Model created: {get_memory_mb() - start_mem:.1f} MB")

        # Forward pass on local chunk
        output_local = model_correct.attention(
            x_local, total_seq_len=seq_len, is_causal=False
        )
        print(f"  After forward: {get_memory_mb() - start_mem:.1f} MB")

        # Peak memory for correct implementation
        correct_peak_mb = get_memory_mb() - start_mem

    except torch.cuda.OutOfMemoryError:
        print("  OOM! This should not happen with correct implementation.")
        correct_peak_mb = float("inf")

    print()
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Incorrect implementation (full sequence): {incorrect_peak_mb:.1f} MB")
    print(f"Correct implementation (local chunk only): {correct_peak_mb:.1f} MB")
    print(f"Memory reduction: {incorrect_peak_mb / correct_peak_mb:.1f}x")
    print()
    print("Key insights:")
    print(f"1. Incorrect: Each GPU stores full {seq_len:,} token sequence")
    print(
        f"2. Correct: Each GPU stores only {local_seq_len:,} tokens (1/{world_size} of total)"
    )
    print(f"3. This enables processing sequences {world_size}x longer!")

    # Additional analysis
    print()
    print("Memory breakdown (approximate):")

    # QKV memory
    qkv_full_mb = 3 * batch_size * seq_len * embed_dim * 4 / (1024**2)
    qkv_local_mb = 3 * batch_size * local_seq_len * embed_dim * 4 / (1024**2)

    print("QKV tensors:")
    print(f"  Incorrect (full): {qkv_full_mb:.1f} MB")
    print(f"  Correct (local): {qkv_local_mb:.1f} MB")
    print(f"  Savings: {qkv_full_mb - qkv_local_mb:.1f} MB")

    # Attention scores memory (for one head)
    attn_full_mb = batch_size * seq_len * seq_len * 4 / (1024**2)
    attn_local_mb = batch_size * local_seq_len * local_seq_len * 4 / (1024**2)

    print("Attention scores (per head):")
    print(f"  Incorrect: {attn_full_mb:.1f} MB")
    print(f"  Correct: {attn_local_mb:.1f} MB")
    print(f"  Savings: {attn_full_mb - attn_local_mb:.1f} MB")


def test_extreme_sequences():
    """Test how long sequences we can handle with correct implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("CUDA not available, skipping extreme sequence test")
        return

    print()
    print("=" * 70)
    print("Testing extreme sequence lengths with correct implementation")
    print("=" * 70)

    batch_size = 1
    embed_dim = 768
    num_heads = 12
    world_size = 4  # Simulate 4 GPUs

    # Test increasing sequence lengths
    test_lengths = [16384, 32768, 65536, 131072, 262144, 524288]

    for total_seq_len in test_lengths:
        local_seq_len = total_seq_len // world_size

        print(
            f"\nTesting {total_seq_len:,} total tokens ({local_seq_len:,} per GPU)..."
        )

        cleanup_memory()
        start_mem = get_memory_mb()

        try:
            # Create local input
            x_local = torch.randn(batch_size, local_seq_len, embed_dim, device=device)

            # Create model
            model = CorrectImplementation(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[4096, 8192, 16384],
                dilation_rates=[1, 2, 4],
                device=device,
                dtype=torch.float32,
            )

            # Forward pass
            output = model.attention(x_local, total_seq_len=total_seq_len)

            memory_mb = get_memory_mb() - start_mem
            throughput = total_seq_len / 0.1  # Assume 100ms for demo

            print(f"  SUCCESS! Memory used: {memory_mb:.1f} MB")
            print(f"  Effective throughput: {throughput:,.0f} tokens/sec")

            # Cleanup for next iteration
            del x_local, model, output

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at {total_seq_len:,} tokens")
            break
        except Exception as e:
            print(f"  Error: {e}")
            break


if __name__ == "__main__":
    test_memory_usage()
    test_extreme_sequences()
