#!/usr/bin/env python3
"""
Simplified fused Hilbert Attention kernel for testing.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_kernel_simple(
    Q,
    K,
    V,
    Out,
    hilbert_map,
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
    """Simplified fused kernel that processes 2x larger blocks."""
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    # Initialize
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Process keys/values - use larger blocks to reduce iterations
    for start_n in range(0, M, BLOCK_N * 2):
        # First block
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < M

        # Get Hilbert indices if provided
        h_idx = (
            tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
            if hilbert_map is not None
            else offs_n
        )

        # Load K and V
        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + h_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + h_idx[None, :] * stride_vn
            + offs_d[:, None] * stride_vd
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        # Attention scores
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e9)

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v))

        l_i = l_i_new
        m_i = m_i_new

        # Second block (if within bounds)
        offs_n2 = start_n + BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n2 = offs_n2 < M

        # Only process if we have valid positions
        has_valid = tl.max(mask_n2.to(tl.int32)) > 0

        if has_valid:
            h_idx2 = (
                tl.load(hilbert_map + offs_n2, mask=mask_n2, other=0)
                if hilbert_map is not None
                else offs_n2
            )

            k_ptrs2 = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + h_idx2[None, :] * stride_kn
                + offs_d[:, None] * stride_kd
            )
            v_ptrs2 = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + h_idx2[None, :] * stride_vn
                + offs_d[:, None] * stride_vd
            )

            k2 = tl.load(k_ptrs2, mask=mask_n2[None, :] & mask_d[:, None], other=0.0)
            v2 = tl.load(v_ptrs2, mask=mask_n2[None, :] & mask_d[:, None], other=0.0)

            s2 = tl.dot(q, k2)
            s2 = tl.where(mask_n2[None, :], s2, -1e9)

            m_ij2 = tl.max(s2, axis=1)
            m_i_new2 = tl.maximum(m_i, m_ij2)
            p2 = tl.exp(s2 - m_i_new2[:, None])
            l_ij2 = tl.sum(p2, axis=1)

            alpha2 = tl.exp(m_i - m_i_new2)
            l_i_new2 = alpha2 * l_i + l_ij2
            acc = acc * alpha2[:, None] + tl.dot(p2, tl.trans(v2))

            l_i = l_i_new2
            m_i = m_i_new2

    # Normalize and store
    acc = acc / tl.maximum(l_i[:, None], 1e-10)
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def launch_fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    hilbert_map: torch.Tensor = None,
) -> torch.Tensor:
    """Launch the fused attention kernel."""
    B, H, M, D = q.shape
    out = torch.empty_like(q)

    # Configuration for 4096 sequence length
    if M <= 2048:
        BLOCK_M = 64
        BLOCK_N = 64
    elif M <= 4096:
        BLOCK_M = 128  # Larger blocks
        BLOCK_N = 128
    else:
        BLOCK_M = 256
        BLOCK_N = 128

    BLOCK_D = min(64, D)

    grid = (triton.cdiv(M, BLOCK_M) * B * H,)

    fused_attention_kernel_simple[grid](
        q,
        k,
        v,
        out,
        hilbert_map,
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
        BLOCK_D=BLOCK_D,
        num_warps=8 if M >= 4096 else 4,
        num_stages=3 if M >= 4096 else 2,
    )

    return out
