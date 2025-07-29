"""
Fixed Hilbert attention kernel that correctly handles sparse patterns.
"""

import triton
import triton.language as tl


@triton.jit
def hilbert_attention_kernel_fixed(
    # Pointers
    Q,
    K,
    V,
    Out,
    hilbert_map,
    # Strides
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
    # Shape
    B,
    H,
    M,
    D,
    # Parameters
    scale,
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fixed Hilbert attention kernel for sparse patterns."""
    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Compute query block boundaries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Masks
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

    # Initialize output and stats
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment boundaries for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # Process keys - iterate through sparse positions directly
    if dilation_rate == 1:
        # Standard dense attention
        for start_n in range(0, M, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Only process keys in the same segment
            in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
            mask_n = (offs_n < M) & in_segment

            # For dense attention, apply Hilbert mapping
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

            # Load keys and values using Hilbert reordering
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

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, v)

            # Update running stats
            l_i = l_i_new
            m_i = m_i_new
    else:
        # Sparse dilated attention
        # Calculate the actual sparse positions we need
        seg_len = seg_end - seg_start
        num_sparse_blocks = tl.cdiv(seg_len, dilation_rate * BLOCK_N)

        for block_idx in range(num_sparse_blocks):
            # Calculate sparse position indices directly
            sparse_start = seg_start + block_idx * dilation_rate * BLOCK_N

            # Create sparse indices
            sparse_block_offsets = tl.arange(0, BLOCK_N) * dilation_rate
            offs_n = sparse_start + sparse_block_offsets

            # Check bounds
            mask_n = (offs_n < seg_end) & (offs_n < M)

            # Apply Hilbert mapping to sparse positions
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

            # Load keys and values using Hilbert reordering
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

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, v)

            # Update running stats
            l_i = l_i_new
            m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])
