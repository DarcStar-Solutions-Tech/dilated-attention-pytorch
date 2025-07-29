"""
Optimized Hilbert Attention kernel that only processes needed positions.
"""

import triton
import triton.language as tl


@triton.jit
def hilbert_attention_kernel_optimized(
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
    """Optimized kernel that only processes positions within segments."""
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

    # Initialize output
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment boundaries for queries
    # Key optimization: All queries in this block are in the same segment
    # So we can compute segment bounds once
    block_start_m = pid_m * BLOCK_M
    seg_idx = block_start_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum((seg_idx + 1) * segment_size, M)

    # For dilated attention, we only need to process every dilation_rate-th position
    # Calculate the actual positions we need to process
    if dilation_rate > 1:
        # Number of positions we actually need in this segment
        num_active = (seg_end - seg_start + dilation_rate - 1) // dilation_rate

        # Process only blocks that contain active positions
        for block_idx in range(0, num_active, BLOCK_N):
            # Calculate actual position indices for this block of active positions
            active_idx = block_idx + tl.arange(0, BLOCK_N)
            actual_n = seg_start + active_idx * dilation_rate

            # Mask for valid positions
            mask_n = (active_idx < num_active) & (actual_n < M)

            # Load Hilbert indices for active positions only
            h_idx = tl.load(hilbert_map + actual_n, mask=mask_n, other=0)

            # Load keys and values
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

            # Compute attention scores
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
            acc = acc * alpha[:, None]
            v_t = tl.trans(v)
            acc += tl.dot(p, v_t)

            # Update for next iteration
            l_i = l_i_new
            m_i = m_i_new
    else:
        # Dense attention - process all positions in segment
        for start_n in range(seg_start, seg_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seg_end

            # Load Hilbert indices
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

            # Load keys and values
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

            # Compute attention scores
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
            acc = acc * alpha[:, None]
            v_t = tl.trans(v)
            acc += tl.dot(p, v_t)

            # Update for next iteration
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
