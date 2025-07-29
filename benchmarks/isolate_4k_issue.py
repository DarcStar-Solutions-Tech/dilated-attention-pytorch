#!/usr/bin/env python3
"""
Isolate the issue that only appears at 4K sequences.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def test_increasing_sizes():
    """Test with increasing sequence sizes to find where it breaks."""

    print("=== Testing Increasing Sequence Sizes ===")

    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    dilation_rate = 4
    batch_size = 1

    # Test sizes
    sizes = [512, 1024, 2048, 3072, 4096, 5120, 6144, 8192]

    print(
        f"{'Size':<6} | {'Unified Norm':<12} | {'Enhanced Norm':<12} | {'Max Diff':<10} | {'Ratio':<8}"
    )
    print("-" * 65)

    for seq_len in sizes:
        torch.cuda.empty_cache()

        # Create models
        unified = (
            UnifiedHilbertAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        enhanced = (
            UnifiedHilbertAttentionOptimizedEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
                enable_sparse_optimization=True,
            )
            .cuda()
            .eval()
        )

        # Use same weights
        enhanced.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
        enhanced.out_proj.weight.data = unified.out_proj.weight.data.clone()

        # Test input
        torch.manual_seed(42)
        x = torch.randn(
            batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32
        )

        with torch.no_grad():
            out_unified = unified(x)
            out_enhanced = enhanced(x)

        diff = (out_unified - out_enhanced).abs().max().item()
        ratio = out_enhanced.norm().item() / out_unified.norm().item()

        print(
            f"{seq_len:<6} | {out_unified.norm().item():<12.4f} | {out_enhanced.norm().item():<12.4f} | {diff:<10.6f} | {ratio:<8.2f}x"
        )

        # Check which path is used
        if seq_len == 4096:
            print("\nDetailed check for 4K:")
            config = enhanced._get_optimal_config(seq_len)
            print(
                f"  Config: {config['block_m']}x{config['block_n']}, fused_softmax={config.get('use_fused_softmax', True)}"
            )

            M_padded = ((seq_len + segment_size - 1) // segment_size) * segment_size
            use_pytorch = M_padded <= 512 or not enhanced._triton_available
            print(f"  M_padded: {M_padded}")
            print(f"  Use PyTorch: {use_pytorch}")
            print(f"  Use Hilbert: {M_padded > enhanced.hilbert_threshold}")


def check_kernel_config_impact():
    """Check if specific kernel configs cause the issue."""

    print("\n\n=== Testing Kernel Configurations ===")

    seq_len = 4096
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    dilation_rate = 4
    batch_size = 1

    # Test different configurations
    configs_to_test = [
        {
            "block_m": 32,
            "block_n": 32,
            "use_fused_softmax": False,
            "desc": "32x32 no fused",
        },
        {
            "block_m": 64,
            "block_n": 64,
            "use_fused_softmax": False,
            "desc": "64x64 no fused",
        },
        {
            "block_m": 32,
            "block_n": 32,
            "use_fused_softmax": True,
            "desc": "32x32 with fused",
        },
        {
            "block_m": 64,
            "block_n": 64,
            "use_fused_softmax": True,
            "desc": "64x64 with fused",
        },
    ]

    # Reference from Unified
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        out_unified = unified(x)
    unified_norm = out_unified.norm().item()

    print(f"Reference (Unified): {unified_norm:.4f}")
    print()

    for config in configs_to_test:
        # Create custom Enhanced with specific config
        class CustomEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def _get_optimal_config(self, seq_len):
                return {
                    "block_m": config["block_m"],
                    "block_n": config["block_n"],
                    "block_d": min(config["block_m"], self.head_dim),
                    "num_warps": 2 if config["block_m"] == 32 else 4,
                    "use_fused_softmax": config["use_fused_softmax"],
                    "rows_per_block": 1,
                    "fused_block_n": config["block_n"],
                    "enable_prefetch": False,
                }

        model = (
            CustomEnhanced(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                segment_size=segment_size,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        # Copy weights
        model.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
        model.out_proj.weight.data = unified.out_proj.weight.data.clone()

        with torch.no_grad():
            out = model(x)

        norm = out.norm().item()
        ratio = norm / unified_norm

        print(f"{config['desc']:<20}: norm={norm:.4f}, ratio={ratio:.2f}x")


def test_accumulation_issue():
    """Test if the issue is in accumulation across segments."""

    print("\n\n=== Testing Accumulation Across Segments ===")

    seq_len = 4096
    hidden_dim = 512
    num_heads = 8
    segment_size = 128
    dilation_rate = 4
    batch_size = 1

    # Number of segments
    num_segments = seq_len // segment_size
    print(f"Number of segments: {num_segments}")
    print(f"Segment size: {segment_size}")
    print(f"Active positions per segment: {segment_size // dilation_rate}")

    # Create a special input where each segment has different magnitude
    x = torch.zeros(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size
        # Give each segment a different magnitude
        x[:, seg_start:seg_end, :] = torch.randn(
            batch_size, segment_size, hidden_dim, device="cuda"
        ) * (0.1 + seg_idx * 0.01)

    # Test
    unified = (
        UnifiedHilbertAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        .cuda()
        .eval()
    )

    enhanced = (
        UnifiedHilbertAttentionOptimizedEnhanced(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            enable_sparse_optimization=True,
        )
        .cuda()
        .eval()
    )

    with torch.no_grad():
        out_unified = unified(x)
        out_enhanced = enhanced(x)

    # Check per-segment statistics
    print("\nPer-segment output norms:")
    print(f"{'Segment':<8} | {'Unified':<10} | {'Enhanced':<10} | {'Ratio':<8}")
    print("-" * 45)

    for seg_idx in range(min(8, num_segments)):  # First 8 segments
        seg_start = seg_idx * segment_size
        seg_end = seg_start + segment_size

        unified_seg_norm = out_unified[:, seg_start:seg_end, :].norm().item()
        enhanced_seg_norm = out_enhanced[:, seg_start:seg_end, :].norm().item()
        ratio = (
            enhanced_seg_norm / unified_seg_norm
            if unified_seg_norm > 0
            else float("inf")
        )

        print(
            f"{seg_idx:<8} | {unified_seg_norm:<10.4f} | {enhanced_seg_norm:<10.4f} | {ratio:<8.2f}x"
        )


def main():
    print("=== Isolating 4K Sparse Issue ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    test_increasing_sizes()
    check_kernel_config_impact()
    test_accumulation_issue()

    print("\n\n=== Key Finding ===")
    print("The issue appears specifically at larger sequences (3K+)")
    print("This suggests an accumulation or normalization issue in the kernel")
    print("that only manifests with multiple segments.")


if __name__ == "__main__":
    main()
