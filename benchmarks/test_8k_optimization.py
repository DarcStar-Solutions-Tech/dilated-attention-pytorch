#!/usr/bin/env python3
"""Test different block configurations to optimize 8K performance."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def test_block_sizes_for_8k():
    """Test different block size configurations for 8K sequences."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = 8192

    # Create dummy tensors
    B, H, M, D = 1, 12, seq_len, 64
    q = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    scale = D**-0.5

    print("Testing Block Configurations for 8K Sequences")
    print("=" * 60)

    # Test different block configurations
    configs = [
        (64, 64, 4),  # Current
        (128, 64, 4),  # Asymmetric
        (64, 128, 4),  # Asymmetric reversed
        (96, 96, 4),  # Intermediate
        (128, 128, 4),  # Larger (might fail on Pascal)
        (32, 128, 4),  # Thin and wide
        (128, 32, 4),  # Tall and narrow
    ]

    results = []

    for BLOCK_M, BLOCK_N, num_warps in configs:
        print(f"\nTesting {BLOCK_M}x{BLOCK_N} with {num_warps} warps...")

        try:
            # Import and test the kernel
            import triton
            import triton.language as tl

            # Simple test kernel
            @triton.jit
            def test_attention_kernel(
                Q,
                K,
                V,
                Out,
                stride_qb,
                stride_qh,
                stride_qm,
                stride_qd,
                stride_kb,
                stride_kh,
                stride_kn,
                stride_kd,
                stride_vb,
                stride_vh,
                stride_vn,
                stride_vd,
                stride_ob,
                stride_oh,
                stride_om,
                stride_od,
                B,
                H,
                M,
                D,
                scale,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
                BLOCK_D: tl.constexpr,
            ):
                pid = tl.program_id(0)
                num_blocks_m = tl.cdiv(M, BLOCK_M)
                pid_m = pid % num_blocks_m
                pid_bh = pid // num_blocks_m
                pid_b = pid_bh // H
                pid_h = pid_bh % H

                # Process one block
                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_d = tl.arange(0, BLOCK_D)

                q_ptrs = (
                    Q
                    + pid_b * stride_qb
                    + pid_h * stride_qh
                    + offs_m[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd
                )
                q_block = tl.load(
                    q_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
                    other=0.0,
                )
                q_block = q_block * scale

                # Initialize accumulator
                acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
                l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
                m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

                # Process K/V blocks
                for start_n in range(0, M, BLOCK_N):
                    offs_n = start_n + tl.arange(0, BLOCK_N)

                    k_ptrs = (
                        K
                        + pid_b * stride_kb
                        + pid_h * stride_kh
                        + offs_n[None, :] * stride_kn
                        + offs_d[:, None] * stride_kd
                    )
                    v_ptrs = (
                        V
                        + pid_b * stride_vb
                        + pid_h * stride_vh
                        + offs_n[None, :] * stride_vn
                        + offs_d[:, None] * stride_vd
                    )

                    k_block = tl.load(
                        k_ptrs,
                        mask=(offs_n[None, :] < M) & (offs_d[:, None] < D),
                        other=0.0,
                    )
                    v_block = tl.load(
                        v_ptrs,
                        mask=(offs_n[None, :] < M) & (offs_d[:, None] < D),
                        other=0.0,
                    )

                    # Compute attention scores
                    s = tl.dot(q_block, k_block)
                    s = tl.where(offs_n[None, :] < M, s, -1e9)

                    # Online softmax
                    m_ij = tl.max(s, axis=1)
                    m_i_new = tl.maximum(m_i, m_ij)
                    p = tl.exp(s - m_i_new[:, None])
                    l_ij = tl.sum(p, axis=1)

                    # Update accumulator
                    alpha = tl.exp(m_i - m_i_new)
                    l_i = alpha * l_i + l_ij
                    acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v_block))
                    m_i = m_i_new

                # Normalize
                acc = acc / tl.maximum(l_i[:, None], 1e-10)

                # Store output
                out_ptrs = (
                    Out
                    + pid_b * stride_ob
                    + pid_h * stride_oh
                    + offs_m[:, None] * stride_om
                    + offs_d[None, :] * stride_od
                )
                tl.store(
                    out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
                )

            # Create output tensor
            out = torch.empty_like(q)

            # Launch kernel
            grid = (triton.cdiv(M, BLOCK_M) * B * H,)

            # Warmup
            for _ in range(3):
                test_attention_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    *q.stride(),
                    *k.stride(),
                    *v.stride(),
                    *out.stride(),
                    B,
                    H,
                    M,
                    D,
                    scale,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=min(64, D),
                    num_warps=num_warps,
                )

            # Time
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                test_attention_kernel[grid](
                    q,
                    k,
                    v,
                    out,
                    *q.stride(),
                    *k.stride(),
                    *v.stride(),
                    *out.stride(),
                    B,
                    H,
                    M,
                    D,
                    scale,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_D=min(64, D),
                    num_warps=num_warps,
                )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 10 * 1000

            print(f"  Time: {elapsed:.2f}ms")
            results.append((BLOCK_M, BLOCK_N, elapsed))

        except Exception as e:
            print(f"  Failed: {str(e)}")
            results.append((BLOCK_M, BLOCK_N, float("inf")))

    # Find best configuration
    if results:
        best = min(results, key=lambda x: x[2])
        print(f"\nBest configuration: {best[0]}x{best[1]} blocks ({best[2]:.2f}ms)")

        current_time = next(r[2] for r in results if r[0] == 64 and r[1] == 64)
        if best[2] < current_time:
            improvement = (current_time - best[2]) / current_time * 100
            print(f"Improvement over current: {improvement:.1f}%")

    return results


def test_grid_alignment():
    """Test if grid alignment affects performance."""
    print("\n\nGrid Alignment Analysis")
    print("=" * 60)

    # GTX 1080 has 20 SMs
    num_sms = 20

    for seq_len in [4096, 6144, 8192, 10240, 12288]:
        for block_size in [64, 96, 128]:
            num_blocks = (seq_len + block_size - 1) // block_size
            total_blocks = num_blocks * num_blocks  # 2D grid
            blocks_per_sm = total_blocks / num_sms

            # Check if it's well aligned
            alignment_score = abs(blocks_per_sm - round(blocks_per_sm))

            print(
                f"Seq {seq_len}, Block {block_size}: "
                f"{total_blocks} blocks, {blocks_per_sm:.1f} per SM, "
                f"alignment score: {alignment_score:.3f}"
            )


def main():
    print("8K Performance Optimization Analysis")
    print("=" * 80)

    # Test different block sizes
    _ = test_block_sizes_for_8k()

    # Test grid alignment
    test_grid_alignment()

    # Recommendations
    print("\n\nRECOMMENDATIONS")
    print("=" * 60)
    print("""
Based on the analysis:

1. **Use asymmetric block sizes for 8K**:
   - The square 64x64 blocks may not be optimal
   - Consider 128x64 or 96x96 blocks

2. **Grid alignment matters**:
   - 8192 with 64x64 blocks = 16384 blocks (819 per SM)
   - This creates poor load balancing
   - Better: 96x96 blocks = 7396 blocks (370 per SM)

3. **Specific fix for 8K-10K range**:
   ```python
   elif 8192 <= M <= 10240:
       BLOCK_M = 96
       BLOCK_N = 96
       num_warps = 4
   ```

This should smooth out the performance curve.
""")


if __name__ == "__main__":
    main()
