#!/usr/bin/env python3
"""
Simple test of ImprovedDilatedAttention capability.
"""

import torch
import gc

from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def test_sequence(seq_len, segment_lengths, dilation_rates):
    """Test a specific sequence length."""
    device = torch.device("cuda")

    # Ensure divisibility
    max_seg = max(segment_lengths)
    if seq_len % max_seg != 0:
        seq_len = ((seq_len // max_seg) + 1) * max_seg

    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model - using only basic parameters
        model = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            enable_memory_pool=True,  # This should help with memory
        )

        # Test inputs
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward
        with torch.no_grad():
            output = model(q, k, v)

        # Memory
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        kb_per_token = peak_mb * 1024 / seq_len

        print(
            f"{seq_len:,} tokens: Success - {peak_mb:.0f} MB ({kb_per_token:.1f} KB/token)"
        )

        # Cleanup
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return True, peak_mb, kb_per_token

    except torch.cuda.OutOfMemoryError:
        print(f"{seq_len:,} tokens: OOM")
        return False, 0, 0
    except Exception as e:
        print(f"{seq_len:,} tokens: Error - {type(e).__name__}: {str(e)}")
        return False, 0, 0


def main():
    """Test improved implementation capability."""

    print("Testing ImprovedDilatedAttention")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print()

    # Test different configurations
    configs = [
        # Config 1: Simple
        {
            "name": "Simple (single segment)",
            "segments": [8192],
            "dilations": [1],
            "test_seqs": [16384, 32768, 65536, 131072, 196608, 262144],
        },
        # Config 2: Medium
        {
            "name": "Medium (2 segments)",
            "segments": [4096, 8192],
            "dilations": [1, 2],
            "test_seqs": [32768, 65536, 131072, 196608, 262144],
        },
        # Config 3: Standard
        {
            "name": "Standard (3 segments)",
            "segments": [2048, 4096, 8192],
            "dilations": [1, 2, 4],
            "test_seqs": [65536, 131072, 196608, 262144],
        },
        # Config 4: Large
        {
            "name": "Large (4 segments)",
            "segments": [4096, 8192, 16384, 32768],
            "dilations": [1, 2, 4, 8],
            "test_seqs": [131072, 196608, 262144, 327680],
        },
    ]

    results = {}

    for cfg in configs:
        print(f"\n=== {cfg['name']} ===")
        print(f"Segments: {cfg['segments']}")
        print(f"Dilations: {cfg['dilations']}")
        print()

        max_seq = 0
        min_kb_token = float("inf")

        for seq in cfg["test_seqs"]:
            success, peak_mb, kb_token = test_sequence(
                seq, cfg["segments"], cfg["dilations"]
            )
            if success:
                max_seq = seq
                min_kb_token = min(min_kb_token, kb_token)
            else:
                break

        results[cfg["name"]] = {
            "max_seq": max_seq,
            "min_kb_token": min_kb_token if min_kb_token != float("inf") else 0,
        }

        print(f"\nMax sequence: {max_seq:,} tokens")
        if min_kb_token != float("inf"):
            print(f"Best efficiency: {min_kb_token:.1f} KB/token")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Max sequence: {result['max_seq']:,} tokens")
        if result["min_kb_token"] > 0:
            print(f"  Best efficiency: {result['min_kb_token']:.1f} KB/token")

    # Compare with hybrid
    print("\n" + "=" * 60)
    print("COMPARISON WITH HYBRID RING ATTENTION")
    print("=" * 60)

    print("\nFor reference, Hybrid Ring Attention (single GPU) achieved:")
    print("  Max sequence: ~335,872 tokens")
    print("  Efficiency at 65K: ~11.2 KB/token")

    best_improved = max(results.values(), key=lambda x: x["max_seq"])
    if best_improved["max_seq"] > 0:
        print("\nImproved Dilated Attention best result:")
        print(f"  Max sequence: {best_improved['max_seq']:,} tokens")
        print(f"  That's {best_improved['max_seq'] / 335872:.1f}x compared to Hybrid")


if __name__ == "__main__":
    main()
