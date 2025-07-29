#!/usr/bin/env python3
"""
Debug the output mismatch between Unified and Enhanced for sparse patterns.
Enhanced is producing outputs 20-32x larger than expected.
"""

import torch
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def trace_computation():
    """Trace through the computation to find where outputs diverge."""

    # Simple test case
    seq_len = 512  # Small for easier debugging
    dilation_rate = 4
    hidden_dim = 64  # Small for manual verification
    num_heads = 2
    segment_size = 128
    batch_size = 1

    print("=== Tracing Computation ===")
    print(
        f"Config: seq_len={seq_len}, d={dilation_rate}, hidden={hidden_dim}, heads={num_heads}"
    )

    # Create simple input
    torch.manual_seed(42)
    x = (
        torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float32)
        * 0.1
    )

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

    # Make weights identical
    enhanced.qkv_proj.weight.data = unified.qkv_proj.weight.data.clone()
    enhanced.out_proj.weight.data = unified.out_proj.weight.data.clone()

    print("\n1. Input statistics:")
    print(f"   Input norm: {x.norm().item():.4f}")
    print(f"   Input mean: {x.mean().item():.4f}")
    print(f"   Input std: {x.std().item():.4f}")

    # Get intermediate values from unified
    with torch.no_grad():
        # QKV projection
        qkv_unified = unified.qkv_proj(x)
        qkv_unified = qkv_unified.reshape(
            batch_size, seq_len, 3, num_heads, hidden_dim // num_heads
        )
        qkv_unified = qkv_unified.permute(2, 0, 3, 1, 4)
        q_unified, k_unified, v_unified = qkv_unified[0], qkv_unified[1], qkv_unified[2]

        print("\n2. QKV statistics (Unified):")
        print(f"   Q norm: {q_unified.norm().item():.4f}")
        print(f"   K norm: {k_unified.norm().item():.4f}")
        print(f"   V norm: {v_unified.norm().item():.4f}")

        # Final output
        out_unified = unified(x)
        print("\n3. Output (Unified):")
        print(f"   Norm: {out_unified.norm().item():.4f}")
        print(f"   Mean: {out_unified.mean().item():.4f}")
        print(f"   Max: {out_unified.max().item():.4f}")

    # Get intermediate values from enhanced
    with torch.no_grad():
        # Check config
        config = enhanced._get_optimal_config(seq_len)
        print("\n4. Enhanced config:")
        print(f"   Block size: {config['block_m']}x{config['block_n']}")
        print(f"   Fused softmax: {config.get('use_fused_softmax', True)}")

        # QKV projection
        qkv_enhanced = enhanced.qkv_proj(x)
        qkv_enhanced = qkv_enhanced.reshape(
            batch_size, seq_len, 3, num_heads, hidden_dim // num_heads
        )
        qkv_enhanced = qkv_enhanced.permute(2, 0, 3, 1, 4)
        q_enhanced, k_enhanced, v_enhanced = (
            qkv_enhanced[0],
            qkv_enhanced[1],
            qkv_enhanced[2],
        )

        print("\n5. QKV statistics (Enhanced):")
        print(f"   Q norm: {q_enhanced.norm().item():.4f}")
        print(f"   K norm: {k_enhanced.norm().item():.4f}")
        print(f"   V norm: {v_enhanced.norm().item():.4f}")
        print(f"   Q-K-V identical: {torch.allclose(q_unified, q_enhanced)}")

        # Final output
        out_enhanced = enhanced(x)
        print("\n6. Output (Enhanced):")
        print(f"   Norm: {out_enhanced.norm().item():.4f}")
        print(f"   Mean: {out_enhanced.mean().item():.4f}")
        print(f"   Max: {out_enhanced.max().item():.4f}")

    # Compare
    print("\n7. Comparison:")
    diff = (out_unified - out_enhanced).abs()
    print(f"   Max difference: {diff.max().item():.6f}")
    print(f"   Relative error: {(diff.max() / out_unified.abs().max()).item():.2%}")

    # Check scale factor
    print("\n8. Scale analysis:")
    print(f"   Unified scale: {unified.scale:.6f}")
    print(f"   Enhanced scale: {enhanced.scale:.6f}")
    print(f"   Head dim: {unified.head_dim}")

    # Check mask value
    print("\n9. Mask value check:")
    print("   Unified uses: -inf for masking")
    print(f"   Enhanced mask_value: {enhanced.mask_value}")

    # Test with hilbert disabled
    print("\n10. Testing without Hilbert:")
    with torch.no_grad():
        out_unified_no_h = unified(x, use_hilbert=False)
        out_enhanced_no_h = enhanced(x, use_hilbert=False)

    print(f"   Unified norm (no Hilbert): {out_unified_no_h.norm().item():.4f}")
    print(f"   Enhanced norm (no Hilbert): {out_enhanced_no_h.norm().item():.4f}")
    print(
        f"   Difference: {(out_unified_no_h - out_enhanced_no_h).abs().max().item():.6f}"
    )


def test_kernel_directly():
    """Test the Triton kernel directly to isolate the issue."""

    print("\n\n=== Testing Kernel Directly ===")

    # Very simple case
    B, H, M, D = 1, 1, 128, 32  # Single head, single segment
    dilation_rate = 4
    segment_size = 128

    # Create simple inputs
    q = torch.ones(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
    k = torch.ones(B, H, M, D, device="cuda", dtype=torch.float32) * 0.1
    v = (
        torch.eye(M, device="cuda", dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, H, -1, -1)[:, :, :, :D]
    )

    print(f"Testing B={B}, H={H}, M={M}, D={D}, dilation={dilation_rate}")

    # Expected behavior for sparse attention with d=4:
    # Each query position should attend to M/4 = 32 positions
    # With uniform Q and K, attention should be uniform over active positions
    # So each output should be average of 32 positions

    effective_positions = M // dilation_rate
    print(f"Effective positions per query: {effective_positions}")

    # Manual computation
    scale = D**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [1, 1, 128, 128]

    # Apply sparse mask manually
    mask = torch.zeros(M, M, device="cuda", dtype=torch.bool)
    for i in range(M):
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        for j in range(seg_start, min(seg_start + segment_size, M), dilation_rate):
            if j < M:
                mask[i, j] = True

    scores_masked = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores_masked, dim=-1)

    # Check attention pattern
    print("\nAttention pattern (row 0):")
    print(f"Non-zero positions: {torch.nonzero(attn[0, 0, 0] > 0).squeeze().tolist()}")
    print(f"Attention sum: {attn[0, 0, 0].sum().item():.4f}")

    out_manual = torch.matmul(attn, v)
    print(f"\nManual computation output norm: {out_manual.norm().item():.4f}")


def check_mask_value_impact():
    """Check if mask_value is causing the issue."""

    print("\n\n=== Checking Mask Value Impact ===")

    seq_len = 512
    hidden_dim = 64

    # Test different mask values
    mask_values = [-1e9, -1e6, -1e3, -100, -10]

    for mask_val in mask_values:
        # Create modified Enhanced
        class TestEnhanced(UnifiedHilbertAttentionOptimizedEnhanced):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.mask_value = mask_val

        model = (
            TestEnhanced(
                hidden_dim=hidden_dim,
                num_heads=2,
                segment_size=128,
                dilation_rate=4,
            )
            .cuda()
            .eval()
        )

        x = torch.randn(1, seq_len, hidden_dim, device="cuda") * 0.1
        with torch.no_grad():
            out = model(x)

        print(f"Mask value {mask_val}: output norm = {out.norm().item():.4f}")


def main():
    print("=== Debugging Sparse Output Mismatch ===")
    print(f"GPU: {torch.cuda.get_device_name()}")

    trace_computation()
    test_kernel_directly()
    check_mask_value_impact()

    print("\n\n=== Hypothesis ===")
    print("The large output values suggest:")
    print("1. Incorrect softmax normalization")
    print("2. Mask value not being applied correctly")
    print("3. Scale factor being applied incorrectly")
    print("4. Accumulation issue in the kernel")


if __name__ == "__main__":
    main()
