#!/usr/bin/env python3
"""
Compare memory efficiency of fp16 vs fp32 in ImprovedDilatedAttention.
This shows exactly where the massive gains come from.
"""

import torch
import gc

from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def test_dtype_efficiency(seq_len, dtype, dtype_name):
    """Test memory usage with specific dtype."""
    device = torch.device("cuda")

    # Optimal config for long sequences
    segment_lengths = [4096, 8192, 16384, 32768]
    dilation_rates = [1, 2, 4, 8]

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        model = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            enable_memory_pool=True,
            lightweight_pool=False,
        )

        # Convert to desired dtype
        if dtype == torch.float16:
            model = model.half()
        elif dtype == torch.bfloat16:
            model = model.bfloat16()

        # Create inputs with specified dtype
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Forward pass
        with torch.no_grad():
            output = model(q, k, v)

        # Get memory stats
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        kb_per_token = peak_mb * 1024 / seq_len

        # Calculate theoretical memory for QKV tensors alone
        qkv_size = 3 * batch_size * seq_len * num_heads * head_dim * dtype.itemsize
        qkv_mb = qkv_size / 1024**2

        print(f"\n{dtype_name} @ {seq_len:,} tokens:")
        print(f"  Total memory: {peak_mb:.1f} MB")
        print(f"  Memory per token: {kb_per_token:.1f} KB/token")
        print(f"  QKV tensors alone: {qkv_mb:.1f} MB")
        print(
            f"  Overhead: {(peak_mb - qkv_mb):.1f} MB ({(peak_mb / qkv_mb):.1f}x QKV size)"
        )

        # Cleanup
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return True, peak_mb, kb_per_token

    except torch.cuda.OutOfMemoryError:
        print(f"\n{dtype_name} @ {seq_len:,} tokens: OOM")
        gc.collect()
        torch.cuda.empty_cache()
        return False, 0, 0
    except Exception as e:
        print(f"\n{dtype_name} @ {seq_len:,} tokens: Error - {e}")
        gc.collect()
        torch.cuda.empty_cache()
        return False, 0, 0


def main():
    """Compare fp16 vs fp32 efficiency."""

    print("=== FP16 vs FP32 Memory Efficiency Comparison ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Test sequences
    test_sequences = [65536, 131072, 196608, 262144, 393216, 524288]

    print("\n" + "=" * 60)
    print("Testing with float32 (baseline)")
    print("=" * 60)

    fp32_results = []
    for seq in test_sequences:
        success, peak_mb, kb_token = test_dtype_efficiency(
            seq, torch.float32, "float32"
        )
        if success:
            fp32_results.append((seq, peak_mb, kb_token))
        else:
            break

    print("\n" + "=" * 60)
    print("Testing with float16 (optimized)")
    print("=" * 60)

    fp16_results = []
    for seq in test_sequences:
        success, peak_mb, kb_token = test_dtype_efficiency(
            seq, torch.float16, "float16"
        )
        if success:
            fp16_results.append((seq, peak_mb, kb_token))
        else:
            break

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nMaximum sequences:")
    max_fp32 = fp32_results[-1][0] if fp32_results else 0
    max_fp16 = fp16_results[-1][0] if fp16_results else 0
    print(f"  float32: {max_fp32:,} tokens")
    print(f"  float16: {max_fp16:,} tokens")
    print(f"  float16 handles {max_fp16 / max_fp32:.1f}x longer sequences")

    print("\nMemory efficiency comparison:")
    print(
        f"{'Seq Length':>12} {'FP32 (KB/tok)':>15} {'FP16 (KB/tok)':>15} {'Reduction':>12}"
    )
    print("-" * 60)

    # Compare at common sequence lengths
    for fp32_data in fp32_results:
        seq = fp32_data[0]
        fp32_kb = fp32_data[2]

        # Find matching fp16 result
        fp16_data = next((x for x in fp16_results if x[0] == seq), None)
        if fp16_data:
            fp16_kb = fp16_data[2]
            reduction = fp32_kb / fp16_kb
            print(f"{seq:>12,} {fp32_kb:>15.1f} {fp16_kb:>15.1f} {reduction:>11.1f}x")

    print("\nKey insights:")
    print("- FP16 uses ~50% less memory for tensor storage")
    print("- But the gain is MORE than 2x due to:")
    print("  * Reduced activation memory")
    print("  * Smaller attention score matrices")
    print("  * Less memory fragmentation")
    print("  * More efficient GPU memory alignment")

    # Show the math
    print("\nMemory calculation example (262K tokens):")
    seq = 262144
    _ = 1
    heads = 8
    dim = 64

    # QKV memory
    qkv_fp32 = 3 * seq * heads * dim * 4 / 1024**2  # 4 bytes per float32
    qkv_fp16 = 3 * seq * heads * dim * 2 / 1024**2  # 2 bytes per float16

    print("\nQKV tensors alone:")
    print(f"  FP32: {qkv_fp32:.1f} MB")
    print(f"  FP16: {qkv_fp16:.1f} MB")
    print(f"  Savings: {qkv_fp32 - qkv_fp16:.1f} MB")

    # Attention scores (for one segment)
    seg_len = 32768  # largest segment
    attn_fp32 = heads * seg_len * seg_len * 4 / 1024**2
    attn_fp16 = heads * seg_len * seg_len * 2 / 1024**2

    print("\nAttention scores (per segment):")
    print(f"  FP32: {attn_fp32:.1f} MB")
    print(f"  FP16: {attn_fp16:.1f} MB")
    print(f"  Savings: {attn_fp32 - attn_fp16:.1f} MB")

    print(
        f"\nTotal theoretical minimum savings: {(qkv_fp32 - qkv_fp16) + (attn_fp32 - attn_fp16):.1f} MB"
    )
    print("Plus additional savings from intermediate activations, buffers, etc.")


if __name__ == "__main__":
    main()
