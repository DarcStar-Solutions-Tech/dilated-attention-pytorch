#!/usr/bin/env python3
"""
Debug the mismatch between Triton and PyTorch dilated attention.
"""

import torch
import torch.nn.functional as F


def debug_dilated_attention():
    """Debug step by step to find the mismatch."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    print("=== Debugging Dilated Attention Mismatch ===\n")

    # Simple configuration for debugging
    batch_size = 1
    seq_len = 32
    hidden_dim = 64
    num_heads = 2
    segment_size = 16
    dilation_rate = 2
    head_dim = hidden_dim // num_heads

    # Create module
    module = (
        HilbertAttentionCore(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        )
        .cuda()
        .eval()
    )

    # Create simple input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    # Get QKV
    qkv = module.qkv_proj(x)
    qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
    q, k, v = qkv[0], qkv[1], qkv[2]

    print("Shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")

    # Check what Triton is doing by running forward
    with torch.no_grad():
        triton_output = module(x, use_hilbert=False)

    # Manually compute dilated attention
    print("\nDilated attention configuration:")
    print(f"  Segment size: {segment_size}")
    print(f"  Dilation rate: {dilation_rate}")
    print(f"  Connections per position: {segment_size // dilation_rate}")

    # Build attention mask manually
    mask = torch.zeros(seq_len, seq_len, device="cuda", dtype=torch.bool)

    for i in range(seq_len):
        seg_idx = i // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        print(f"\nQuery {i} (segment {seg_idx}):")
        print(f"  Segment range: [{seg_start}, {seg_end})")
        print("  Attending to: ", end="")

        attending_positions = []
        for j in range(seg_start, seg_end):
            if (j - seg_start) % dilation_rate == 0:
                mask[i, j] = True
                attending_positions.append(j)

        print(attending_positions)

    # Apply attention with our mask
    scores = torch.matmul(q, k.transpose(-2, -1)) * module.scale
    scores_masked = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
    attn_weights = F.softmax(scores_masked, dim=-1)
    manual_output = torch.matmul(attn_weights, v)

    # Reshape and project
    manual_output = manual_output.transpose(1, 2).contiguous()
    manual_output = manual_output.reshape(batch_size, seq_len, hidden_dim)
    manual_output = module.out_proj(manual_output)

    # Compare
    diff = torch.abs(triton_output - manual_output).max().item()
    print(f"\nMax difference: {diff}")

    # Check a specific position
    pos = 8  # Middle of first segment
    print(f"\nDetailed comparison at position {pos}:")
    print(f"  Triton output: {triton_output[0, pos, :5]}")
    print(f"  Manual output: {manual_output[0, pos, :5]}")

    # Check attention weights for debugging
    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"First head, query {pos} attention weights:")
    weights = attn_weights[0, 0, pos]
    nonzero = weights.nonzero().squeeze()
    print(f"  Non-zero positions: {nonzero.tolist()}")
    print(f"  Weights at those positions: {weights[nonzero].tolist()}")

    # Also check what happens in standard (non-Triton) forward
    print("\n=== Checking PyTorch fallback path ===")

    # Force PyTorch path by using small head dimension
    small_module = (
        HilbertAttentionCore(
            hidden_dim=16,  # This gives head_dim=8 < 16 minimum
            num_heads=2,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            use_custom_backward=False,
        )
        .cuda()
        .eval()
    )

    x_small = torch.randn(batch_size, seq_len, 16, device="cuda")
    with torch.no_grad():
        pytorch_path_output = small_module(x_small, use_hilbert=False)

    print(f"PyTorch fallback path output shape: {pytorch_path_output.shape}")


def check_triton_kernel_logic():
    """Check the Triton kernel's dilation logic more carefully."""
    print("\n=== Checking Triton Kernel Logic ===\n")

    # Let's trace through what the kernel should be doing
    seq_len = 32
    segment_size = 16
    dilation_rate = 2

    # Simulate the kernel logic
    for query_idx in [0, 8, 16, 24]:
        seg_idx = query_idx // segment_size
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, seq_len)

        print(f"\nQuery {query_idx}:")
        print(f"  Segment {seg_idx}: [{seg_start}, {seg_end})")

        # What the kernel checks
        for key_idx in range(seq_len):
            in_segment = (key_idx >= seg_start) and (key_idx < seg_end)
            dilation_ok = ((key_idx - seg_start) % dilation_rate) == 0

            if in_segment and dilation_ok:
                print(f"  ✓ Attends to key {key_idx}")


def test_different_configurations():
    """Test various configurations to understand the pattern."""
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    print("\n=== Testing Different Configurations ===\n")

    configs = [
        (32, 32, 1, "Full attention (one segment)"),
        (32, 16, 1, "Two segments, no dilation"),
        (32, 16, 2, "Two segments, dilation 2"),
        (32, 8, 2, "Four segments, dilation 2"),
    ]

    for seq_len, seg_size, dil_rate, desc in configs:
        print(f"\n{desc}:")

        _ = (
            HilbertAttentionCore(
                hidden_dim=64,
                num_heads=2,
                segment_size=seg_size,
                dilation_rate=dil_rate,
            )
            .cuda()
            .eval()
        )

        _ = torch.randn(1, seq_len, 64, device="cuda")

        # Count actual connections
        total_connections = 0
        for i in range(seq_len):
            seg_idx = i // seg_size
            seg_start = seg_idx * seg_size
            seg_end = min(seg_start + seg_size, seq_len)

            connections = 0
            for j in range(seg_start, seg_end):
                if (j - seg_start) % dil_rate == 0:
                    connections += 1

            total_connections += connections

        avg_connections = total_connections / seq_len
        print(f"  Average connections per query: {avg_connections:.1f}")
        print(
            f"  Total FLOPs reduction: {(seq_len * seq_len) / total_connections:.1f}x"
        )


def main():
    """Run debugging."""
    debug_dilated_attention()
    check_triton_kernel_logic()
    test_different_configurations()


if __name__ == "__main__":
    main()
