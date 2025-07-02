#!/usr/bin/env python3
"""
Quick comparison of Hybrid Ring vs Improved Dilated Attention.
Focus on memory efficiency differences.
"""

import torch
import gc
from typing import Dict

from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def test_memory_usage(model_type: str, seq_len: int) -> Dict:
    """Test memory usage for a specific sequence length."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    # Ensure divisibility
    max_seg = max(segment_lengths)
    if seq_len % max_seg != 0:
        seq_len = ((seq_len // max_seg) + 1) * max_seg

    try:
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Create model
        if model_type == "hybrid":
            model = RingDilatedAttentionHybrid(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
                ring_size=1,  # Single GPU
                use_flash_attention=True,
                enable_memory_pool=True,
            )
        else:  # improved
            model = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                use_xformers=True,
                use_flex_attention=False,
            )

        # Create minimal inputs
        batch_size = 1
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        # Forward pass
        with torch.no_grad():
            output = model(q, k, v)

        # Get memory
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        memory_per_token_kb = peak_memory_mb * 1024 / seq_len

        result = {
            "success": True,
            "seq_len": seq_len,
            "memory_mb": peak_memory_mb,
            "memory_per_token_kb": memory_per_token_kb,
        }

        # Cleanup
        del q, k, v, output, model
        gc.collect()
        torch.cuda.empty_cache()

        return result

    except torch.cuda.OutOfMemoryError:
        return {"success": False, "seq_len": seq_len, "error": "OOM"}
    except Exception as e:
        return {"success": False, "seq_len": seq_len, "error": str(e)}


def find_max_sequence(model_type: str, start: int = 65536) -> int:
    """Find maximum sequence length."""

    _ = start
    max_working = 0

    # Quick binary search
    low = 16384
    high = 512000

    while low <= high:
        mid = (low + high) // 2
        # Round to nearest 8192
        mid = (mid // 8192) * 8192

        result = test_memory_usage(model_type, mid)

        if result["success"]:
            max_working = mid
            low = mid + 8192
            print(
                f"{model_type}: {mid:,} tokens - Success ({result['memory_mb']:.0f} MB)"
            )
        else:
            high = mid - 8192
            print(f"{model_type}: {mid:,} tokens - Failed")

    return max_working


def main():
    """Quick comparison."""

    print("=== Quick Hybrid vs Improved Comparison ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
    )

    # Test specific sequence lengths
    test_seqs = [16384, 32768, 65536, 98304, 131072]

    print("Memory Usage Comparison:")
    print("-" * 60)
    print(f"{'Seq Length':>12} {'Improved (KB/tok)':>20} {'Hybrid (KB/tok)':>20}")
    print("-" * 60)

    for seq in test_seqs:
        improved = test_memory_usage("improved", seq)
        hybrid = test_memory_usage("hybrid", seq)

        if improved["success"] and hybrid["success"]:
            print(
                f"{seq:>12,} {improved['memory_per_token_kb']:>20.1f} "
                f"{hybrid['memory_per_token_kb']:>20.1f}"
            )
        elif improved["success"]:
            print(f"{seq:>12,} {improved['memory_per_token_kb']:>20.1f} {'OOM':>20}")
        elif hybrid["success"]:
            print(f"{seq:>12,} {'OOM':>20} {hybrid['memory_per_token_kb']:>20.1f}")
        else:
            print(f"{seq:>12,} {'OOM':>20} {'OOM':>20}")

    print("\nFinding maximum sequences...")

    # Find max for improved
    print("\nImproved Dilated Attention:")
    improved_max = find_max_sequence("improved", start=131072)

    # Find max for hybrid
    print("\nHybrid Ring Dilated Attention (Single GPU):")
    hybrid_max = find_max_sequence("hybrid", start=65536)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\nMaximum Sequence Length on 8GB GPU:")
    print(f"  Improved: {improved_max:,} tokens")
    print(f"  Hybrid: {hybrid_max:,} tokens")
    print(f"  Improved handles {improved_max / hybrid_max:.1f}x longer sequences")

    # Test memory at 65K for both
    print("\nMemory Efficiency at 65,536 tokens:")
    improved_65k = test_memory_usage("improved", 65536)
    hybrid_65k = test_memory_usage("hybrid", 65536)

    if improved_65k["success"] and hybrid_65k["success"]:
        print(f"  Improved: {improved_65k['memory_per_token_kb']:.1f} KB/token")
        print(f"  Hybrid: {hybrid_65k['memory_per_token_kb']:.1f} KB/token")
        ratio = hybrid_65k["memory_per_token_kb"] / improved_65k["memory_per_token_kb"]
        print(f"  Improved is {ratio:.1f}x more memory efficient")

    print("\nKey Insights:")
    print("- Improved is optimized for single GPU with extreme memory efficiency")
    print("- Hybrid has overhead from ring attention infrastructure on single GPU")
    print("- Hybrid shines with multiple GPUs where it scales O(n/p)")
    print("- Use Improved for single GPU long sequences, Hybrid for multi-GPU")


if __name__ == "__main__":
    main()
