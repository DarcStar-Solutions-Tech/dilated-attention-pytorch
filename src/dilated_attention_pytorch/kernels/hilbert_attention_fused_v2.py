#!/usr/bin/env python3
"""
Simple fused kernel for testing at sequence length 4096.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_attention_4096(
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
    """Optimized kernel for 4096 sequence length."""
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query processing
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

    # Process all K/V blocks
    for start_n in range(0, M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < M

        # Load K and V (with optional Hilbert reordering)
        if hilbert_map is not None:
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
        else:
            h_idx = offs_n

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

        # Compute attention
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e9)

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update accumulator
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij
        acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v))

        l_i = l_i_new
        m_i = m_i_new

    # Final normalization
    acc = acc / tl.maximum(l_i[:, None], 1e-10)

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def launch_fused_kernel(q, k, v, scale, hilbert_map=None):
    """Launch the fused kernel."""
    B, H, M, D = q.shape
    out = torch.empty_like(q)

    # Get GPU compute capability for optimal config
    compute_capability = torch.cuda.get_device_capability(q.device)[0]

    # Optimized config based on GPU and sequence length
    if compute_capability < 7:  # Pascal and older (limited shared memory)
        if M == 4096:
            BLOCK_M = 64  # Small blocks for Pascal
            BLOCK_N = 64
            num_warps = 4
        else:
            BLOCK_M = 32
            BLOCK_N = 32
            num_warps = 2
        BLOCK_D = min(32, D)
    else:  # Volta and newer
        if M == 4096:
            BLOCK_M = 128
            BLOCK_N = 128
            num_warps = 4
        elif M <= 2048:
            BLOCK_M = 64
            BLOCK_N = 64
            num_warps = 4
        else:
            BLOCK_M = 128
            BLOCK_N = 64
            num_warps = 4
        BLOCK_D = min(64, D)

    grid = (triton.cdiv(M, BLOCK_M) * B * H,)

    fused_attention_4096[grid](
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
        num_warps=num_warps,
    )

    return out
