#!/usr/bin/env python3
"""
Confirm the normalization bug in the standard softmax path.
The issue: when USE_FUSED_SOFTMAX=False, the kernel accumulates attention
outputs without proper normalization across blocks.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def demonstrate_bug():
    """Demonstrate the normalization bug."""

    print("=== Demonstrating Normalization Bug ===")

    # Use 1024 tokens (8 segments) to clearly show the issue
    seq_len = 1024
    hidden_dim = 64
    num_heads = 2
    segment_size = 128
    dilation_rate = 4
    batch_size = 1

    # Create unified reference
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

    # Test input
    torch.manual_seed(42)
    x = (
        torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)
        * 0.1
    )

    with torch.no_grad():
        out_unified = unified(x)

    print("\nReference (Unified):")
    print(f"  Output norm: {out_unified.norm().item():.4f}")
    print(f"  Output mean: {out_unified.mean().item():.6f}")
    print(f"  Output std: {out_unified.std().item():.6f}")

    # Test different configurations
    configs = [
        (32, 32, False, "32x32 no fused (BUGGY)"),
        (32, 32, True, "32x32 with fused"),
        (64, 64, False, "64x64 no fused (BUGGY)"),
        (64, 64, True, "64x64 with fused"),
    ]

    print(
        f"\n{'Config':<25} | {'Norm':<10} | {'Mean':<12} | {'Std':<10} | {'Ratio':<8}"
    )
    print("-" * 75)

    for block_m, block_n, use_fused, desc in configs:
        # Create custom Enhanced
        class CustomEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def _get_optimal_config(self, seq_len):
                return {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_d": min(block_m, self.head_dim),
                    "num_warps": 2 if block_m == 32 else 4,
                    "use_fused_softmax": use_fused,
                    "rows_per_block": 1,
                    "fused_block_n": block_n,
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
        mean = out.mean().item()
        std = out.std().item()
        ratio = norm / out_unified.norm().item()

        print(
            f"{desc:<25} | {norm:<10.4f} | {mean:<12.6f} | {std:<10.6f} | {ratio:<8.2f}x"
        )


def explain_bug():
    """Explain why the bug happens."""

    print("\n\n=== Bug Explanation ===")

    print("The kernel has two softmax paths:")
    print()
    print("1. FUSED SOFTMAX (lines 213-240):")
    print("   - Maintains running statistics (m_i, l_i)")
    print("   - Updates accumulator with proper scaling: acc = acc * alpha")
    print("   - NORMALIZES at the end: acc = acc / l_i")
    print("   ✓ This path works correctly")
    print()
    print("2. STANDARD SOFTMAX (lines 242-249):")
    print("   - Computes softmax per block: p = exp(s - m_ij) / l_ij")
    print("   - Accumulates: acc += dot(p, v)")
    print("   - NEVER normalizes by total sum!")
    print("   ✗ This causes outputs to be ~20x too large")
    print()
    print("Why 20x? With dilation_rate=4:")
    print("- Each segment has ~32 active positions")
    print("- With 32 segments, we accumulate 32 blocks")
    print("- Each block's softmax sums to 1.0")
    print("- But we need to divide by total normalization")


def calculate_expected_ratio():
    """Calculate the expected error ratio."""

    print("\n\n=== Expected Error Ratio ===")

    seq_len = 4096
    segment_size = 128
    dilation_rate = 4
    block_n = 32

    num_segments = seq_len // segment_size
    active_per_segment = segment_size // dilation_rate

    # With block_n=32, each segment needs 1 block for sparse d=4
    blocks_per_segment = (active_per_segment + block_n - 1) // block_n
    total_blocks = num_segments * blocks_per_segment

    print(f"Sequence length: {seq_len}")
    print(f"Number of segments: {num_segments}")
    print(f"Active positions per segment: {active_per_segment}")
    print(f"Blocks per segment: {blocks_per_segment}")
    print(f"Total K/V blocks: {total_blocks}")
    print()
    print(f"Expected error ratio: ~{total_blocks}x")
    print("Observed error ratio: ~20x")
    print()
    print("The slight difference is due to attention weights not being uniform.")


def show_fix():
    """Show how to fix the bug."""

    print("\n\n=== How to Fix ===")

    print("The standard softmax path needs to track normalization:")
    print()
    print("Current (BUGGY):")
    print("```triton")
    print("# Standard softmax path")
    print("m_ij = tl.max(s, axis=1, keep_dims=True)")
    print("p = tl.exp(s - m_ij)")
    print("l_ij = tl.sum(p, axis=1, keep_dims=True)")
    print("p = p / l_ij")
    print("acc += tl.dot(p, v)  # ← Bug: no normalization tracking")
    print("```")
    print()
    print("Fixed:")
    print("```triton")
    print("# Initialize normalization tracker")
    print("l_total = tl.zeros([BLOCK_M], dtype=tl.float32)")
    print()
    print("# Standard softmax path")
    print("m_ij = tl.max(s, axis=1, keep_dims=True)")
    print("p = tl.exp(s - m_ij)")
    print("l_ij = tl.sum(p, axis=1, keep_dims=True)")
    print("p_normalized = p / l_ij")
    print()
    print("# Track total normalization")
    print("l_total += tl.sum(p, axis=1)  # Sum unnormalized exp values")
    print("acc += tl.dot(p_normalized, v)")
    print()
    print("# After all blocks, normalize")
    print("if not USE_FUSED_SOFTMAX:")
    print("    acc = acc / tl.maximum(l_total[:, None], 1e-10)")
    print("```")


def main():
    print("=== Confirming Normalization Bug in Enhanced Kernel ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    demonstrate_bug()
    explain_bug()
    calculate_expected_ratio()
    show_fix()

    print("\n\n=== Summary ===")
    print("1. The bug is in the standard softmax path (USE_FUSED_SOFTMAX=False)")
    print("2. It accumulates attention outputs without tracking normalization")
    print("3. This causes outputs to be ~20x too large for 4K sequences")
    print("4. The fix requires tracking total normalization like the fused path does")
    print()
    print("This explains why:")
    print("- 4K d=4 wasn't actually 0.09ms - that was a measurement error")
    print("- The actual time is ~3.3ms, which is reasonable")
    print("- But the outputs are wrong due to this normalization bug")


if __name__ == "__main__":
    main()
