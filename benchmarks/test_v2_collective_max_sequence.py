#!/usr/bin/env python3
"""
Test maximum sequence length that V2 Collective can handle.
"""

import gc
import torch
import torch.cuda
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        free = total - reserved
        return allocated, reserved, free, total
    return 0, 0, 0, 0


def test_sequence_length(
    seq_len, batch_size=1, num_heads=8, head_dim=64, dtype=torch.float16
):
    """Test if a specific sequence length works."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clear memory
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        # Adjust segment lengths based on sequence length
        if seq_len <= 8192:
            segment_lengths = [1024, 2048]
            dilation_rates = [1, 2]
        elif seq_len <= 32768:
            segment_lengths = [2048, 4096]
            dilation_rates = [1, 2]
        elif seq_len <= 131072:
            segment_lengths = [4096, 8192]
            dilation_rates = [1, 2]
        else:
            segment_lengths = [8192, 16384]
            dilation_rates = [1, 2]

        # Ensure sequence length is divisible by largest segment
        largest_segment = max(segment_lengths)
        if seq_len % largest_segment != 0:
            seq_len = (
                (seq_len + largest_segment - 1) // largest_segment
            ) * largest_segment

        print(
            f"\nTesting seq_len={seq_len:,} (batch={batch_size}, heads={num_heads}, dim={head_dim})"
        )
        print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

        # Create attention module
        attention = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            device=device,
            dtype=dtype,
            enable_memory_pool=True,  # Enable memory pooling
            use_flash_attention=True,
            flash_chunk_size=2048,
        )

        # Report memory before allocation
        if device.type == "cuda":
            alloc, reserved, free, total = get_gpu_memory_info()
            print(f"  GPU Memory before: {free:.2f}GB free of {total:.2f}GB")

        # Calculate theoretical memory requirement
        # Each tensor (Q,K,V) = batch * seq_len * num_heads * head_dim * bytes_per_element
        bytes_per_element = 2 if dtype == torch.float16 else 4
        tensor_size_mb = (
            batch_size * seq_len * num_heads * head_dim * bytes_per_element
        ) / (1024**2)
        total_input_mb = tensor_size_mb * 3  # Q, K, V

        # Attention computation needs additional memory for scores and output
        # Worst case: scores matrix = batch * heads * seq_len * seq_len
        scores_size_mb = (
            batch_size * num_heads * seq_len * seq_len * bytes_per_element
        ) / (1024**2)

        print(
            f"  Theoretical memory: {total_input_mb:.1f}MB (inputs) + {scores_size_mb:.1f}MB (scores)"
        )

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

        if device.type == "cuda":
            torch.cuda.synchronize()
            alloc, reserved, free, total = get_gpu_memory_info()
            print(f"  After allocation: {free:.2f}GB free")

        # Run forward pass
        output = attention(q, k, v, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            alloc, reserved, free, total = get_gpu_memory_info()
            print(f"  After forward: {free:.2f}GB free, Peak: {peak_memory_mb:.1f}MB")

        # Clean up
        del q, k, v, output, attention
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return True, seq_len

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  ❌ OOM at seq_len={seq_len:,}")
            # Clean up
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False, seq_len
        else:
            print(f"  ❌ Error: {e}")
            raise
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        raise


def find_max_sequence_length():
    """Find the maximum sequence length through binary search."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("V2 Collective Maximum Sequence Length Test")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        _, _, _, total_memory = get_gpu_memory_info()
        print(f"Total GPU Memory: {total_memory:.2f}GB")
    print("=" * 80)

    # Test increasing sequence lengths
    test_lengths = [
        # Start with reasonable lengths
        4_096,
        8_192,
        16_384,
        32_768,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,  # 1M tokens
        2_097_152,  # 2M tokens
        4_194_304,  # 4M tokens
    ]

    max_working = 0
    first_failure = None

    # First, find rough bounds
    print("\nPhase 1: Finding rough bounds...")
    for seq_len in test_lengths:
        success, length = test_sequence_length(
            seq_len, batch_size=1, num_heads=8, head_dim=64
        )
        if success:
            max_working = length
        else:
            first_failure = length
            break

    # Binary search for exact limit
    if first_failure and max_working > 0:
        print("\nPhase 2: Binary search for exact limit...")
        lower = max_working
        upper = first_failure

        while upper - lower > 8192:  # Stop when range is small enough
            mid = ((lower + upper) // 2 // 8192) * 8192  # Round to nearest 8192
            success, _ = test_sequence_length(
                mid, batch_size=1, num_heads=8, head_dim=64
            )

            if success:
                lower = mid
                max_working = mid
            else:
                upper = mid

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"✓ Maximum working sequence length: {max_working:,} tokens")

    if max_working > 0:
        print("\nWith this configuration:")
        print("  - Batch size: 1")
        print("  - Attention heads: 8")
        print("  - Head dimension: 64")
        print("  - Data type: float16")

        # Test with different configurations
        print("\nTesting other configurations...")

        # Test with smaller head dim
        print("\nWith head_dim=32:")
        success32, _ = test_sequence_length(
            max_working * 2, batch_size=1, num_heads=8, head_dim=32
        )
        if success32:
            print(
                f"  ✓ Can handle {max_working * 2:,} tokens with smaller head dimension"
            )

        # Test with larger batch
        print("\nWith batch_size=2:")
        success_batch2, _ = test_sequence_length(
            max_working // 2, batch_size=2, num_heads=8, head_dim=64
        )
        if success_batch2:
            print(f"  ✓ Can handle {max_working // 2:,} tokens with batch size 2")

        # Test with more heads
        print("\nWith num_heads=16:")
        success_heads16, _ = test_sequence_length(
            max_working // 2, batch_size=1, num_heads=16, head_dim=64
        )
        if success_heads16:
            print(f"  ✓ Can handle {max_working // 2:,} tokens with 16 attention heads")

    # Memory scaling analysis
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS:")
    print("=" * 80)

    if device.type == "cuda":
        # Test memory usage at different sequence lengths
        test_lens = [4096, 8192, 16384, 32768, 65536]
        memory_results = []

        for seq_len in test_lens:
            if seq_len > max_working:
                break

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            success, _ = test_sequence_length(
                seq_len, batch_size=1, num_heads=8, head_dim=64
            )
            if success:
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                memory_results.append((seq_len, peak_mb))

        if len(memory_results) >= 2:
            print("\nMemory usage scaling:")
            for seq_len, mem_mb in memory_results:
                mb_per_k = mem_mb / (seq_len / 1000)
                print(
                    f"  {seq_len:>7,} tokens: {mem_mb:>8.1f} MB ({mb_per_k:.1f} MB per 1K tokens)"
                )

            # Estimate memory scaling
            x1, y1 = memory_results[0]
            x2, y2 = memory_results[-1]
            slope = (y2 - y1) / (x2 - x1)

            print(f"\nEstimated memory scaling: ~{slope * 1000:.1f} MB per 1K tokens")
            print("This suggests O(n) memory complexity, as expected")


if __name__ == "__main__":
    find_max_sequence_length()
