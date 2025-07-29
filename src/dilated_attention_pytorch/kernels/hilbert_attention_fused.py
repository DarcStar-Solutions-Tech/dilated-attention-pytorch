#!/usr/bin/env python3
"""
Fused Hilbert Attention kernels for improved performance at medium sequence lengths.

This module implements fused kernels that combine multiple operations to reduce
kernel launch overhead, particularly beneficial for sequence lengths around 4096.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_hilbert_attention_kernel(
    # Pointers
    Q,
    K,
    V,
    Out,
    hilbert_map,
    dropout_mask,  # Pre-generated dropout mask if training
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
    stride_db,
    stride_dh,
    stride_dm,
    stride_dn,
    # Shape
    B,
    H,
    M,
    D,
    # Parameters
    scale,
    dropout_p,
    is_causal: tl.constexpr,
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Fused operations flags
    USE_DROPOUT: tl.constexpr,
    USE_HILBERT: tl.constexpr,
    USE_SPARSE: tl.constexpr,
):
    """
    Fused attention kernel that combines:
    1. Hilbert reordering (optional)
    2. Sparse/dilated attention pattern (optional)
    3. Causal masking (optional)
    4. Dropout (optional)
    5. Softmax normalization

    This reduces kernel launches by doing everything in one pass.
    """
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

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment boundaries if using sparse attention
    if USE_SPARSE:
        seg_idx = offs_m // segment_size
        seg_start = seg_idx * segment_size
        seg_end = tl.minimum(seg_start + segment_size, M)

    # Process keys/values in blocks
    # Fused optimization: Process larger blocks to reduce iterations
    FUSED_BLOCK_N = BLOCK_N * 2 if M >= 4096 else BLOCK_N

    for start_n in range(0, M, FUSED_BLOCK_N):
        # Process two blocks at once for better efficiency
        for block_offset in range(0, FUSED_BLOCK_N, BLOCK_N):
            if start_n + block_offset >= M:
                break

            # Key indices
            offs_n = start_n + block_offset + tl.arange(0, BLOCK_N)

            # Unified masking logic
            mask_n = offs_n < M

            # Apply sparse pattern if enabled
            if USE_SPARSE:
                in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
                dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
                mask_n = mask_n & in_segment & dilation_mask

            # Apply causal mask if enabled
            if is_causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                mask_n_2d = mask_n[None, :] & causal_mask
            else:
                mask_n_2d = mask_n[None, :]

            # Skip if no valid positions
            has_valid = tl.max(mask_n.to(tl.int32)) > 0
            if not has_valid:
                continue

            # Get indices (Hilbert or standard)
            if USE_HILBERT:
                h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
            else:
                h_idx = offs_n

            # Fused load of K and V (coalesced memory access)
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

            # Apply masking
            if is_causal:
                s = tl.where(mask_n_2d & mask_m[:, None], s, -1e9)
            else:
                s = tl.where(mask_n[None, :], s, -1e9)

            # Online softmax with stability
            m_ij = tl.max(s, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)

            # Compute exponentials
            p = tl.exp(s - m_i_new[:, None])

            # Apply dropout if enabled (fused into attention)
            if USE_DROPOUT:
                dropout_offs = offs_m[:, None] * M + offs_n[None, :]
                dropout_ptrs = (
                    dropout_mask + pid_b * stride_db + pid_h * stride_dh + dropout_offs
                )
                keep_mask = tl.load(
                    dropout_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=1.0
                )
                p = p * keep_mask / (1.0 - dropout_p)

            l_ij = tl.sum(p, axis=1)

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij

            # Update accumulator (fused with scaling)
            acc = acc * alpha[:, None]
            acc += tl.dot(p, tl.trans(v))

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


@triton.jit
def fused_hilbert_attention_kernel_v2(
    # Pointers for fused QKV tensor
    QKV,  # Shape: [3, B, H, M, D]
    Out,
    hilbert_map,
    # Strides for fused tensor
    stride_qkv_t,
    stride_qkv_b,
    stride_qkv_h,
    stride_qkv_m,
    stride_qkv_d,
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
    # Optimizations
    USE_HILBERT: tl.constexpr,
    PROCESS_MULTIPLE_ROWS: tl.constexpr,
):
    """
    V2: Even more optimized kernel that:
    1. Takes pre-fused QKV tensor to reduce memory loads
    2. Processes multiple query rows per block
    3. Uses larger tile sizes for better efficiency
    """
    # Get program IDs
    pid = tl.program_id(0)

    # Process multiple rows per block
    ROWS_PER_BLOCK = 2 if PROCESS_MULTIPLE_ROWS else 1
    num_blocks_m = tl.cdiv(M, BLOCK_M * ROWS_PER_BLOCK)

    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Process multiple query rows
    for row_idx in range(ROWS_PER_BLOCK):
        row_offset = row_idx * BLOCK_M
        offs_m = pid_m * BLOCK_M * ROWS_PER_BLOCK + row_offset + tl.arange(0, BLOCK_M)

        if tl.min(offs_m) >= M:
            continue

        offs_d = tl.arange(0, BLOCK_D)

        # Masks
        mask_m = offs_m < M
        mask_d = offs_d < D

        # Load Q from fused tensor (dimension 0)
        q_ptrs = (
            QKV
            + 0 * stride_qkv_t
            + pid_b * stride_qkv_b
            + pid_h * stride_qkv_h
            + offs_m[:, None] * stride_qkv_m
            + offs_d[None, :] * stride_qkv_d
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
        q = q * scale

        # Initialize accumulators
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

        # Determine segment for sparse attention
        seg_idx = offs_m // segment_size
        seg_start = seg_idx * segment_size
        seg_end = tl.minimum(seg_start + segment_size, M)

        # Process keys/values with larger tiles
        LARGE_BLOCK_N = BLOCK_N * 4  # Process 4x more at once

        for start_n in range(0, M, LARGE_BLOCK_N):
            # Process sub-blocks within the large block
            for sub_block in range(0, LARGE_BLOCK_N, BLOCK_N):
                if start_n + sub_block >= M:
                    break

                offs_n = start_n + sub_block + tl.arange(0, BLOCK_N)

                # Sparse attention mask
                in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
                dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
                mask_n = (offs_n < M) & in_segment & dilation_mask

                has_valid = tl.max(mask_n.to(tl.int32)) > 0
                if not has_valid:
                    continue

                # Get indices
                if USE_HILBERT:
                    h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
                else:
                    h_idx = offs_n

                # Load K and V from fused tensor (dimensions 1 and 2)
                k_ptrs = (
                    QKV
                    + 1 * stride_qkv_t
                    + pid_b * stride_qkv_b
                    + pid_h * stride_qkv_h
                    + h_idx[None, :] * stride_qkv_m
                    + offs_d[:, None] * stride_qkv_d
                )
                v_ptrs = (
                    QKV
                    + 2 * stride_qkv_t
                    + pid_b * stride_qkv_b
                    + pid_h * stride_qkv_h
                    + h_idx[None, :] * stride_qkv_m
                    + offs_d[:, None] * stride_qkv_d
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


def get_fused_kernel_config(seq_len: int, head_dim: int, device) -> dict:
    """Get optimal configuration for fused kernels based on sequence length."""
    compute_capability = torch.cuda.get_device_capability(device)[0]

    if seq_len <= 2048:
        # Small sequences: standard config
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_D": min(64, head_dim),
            "num_warps": 4,
            "num_stages": 2,
            "use_v2": False,
            "process_multiple_rows": False,
        }
    elif seq_len <= 4096:
        # Medium sequences: optimized for reducing kernel launches
        if compute_capability >= 8:  # Ampere+
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_D": min(64, head_dim),
                "num_warps": 8,
                "num_stages": 3,
                "use_v2": True,  # Use V2 kernel
                "process_multiple_rows": True,
            }
        else:
            return {
                "BLOCK_M": 64,
                "BLOCK_N": 128,
                "BLOCK_D": min(64, head_dim),
                "num_warps": 4,
                "num_stages": 2,
                "use_v2": True,
                "process_multiple_rows": False,
            }
    else:
        # Large sequences: maximize parallelism
        if compute_capability >= 8:
            return {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_D": min(128, head_dim),
                "num_warps": 8,
                "num_stages": 3,
                "use_v2": True,
                "process_multiple_rows": True,
            }
        else:
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_D": min(64, head_dim),
                "num_warps": 4,
                "num_stages": 2,
                "use_v2": False,
                "process_multiple_rows": False,
            }


class FusedHilbertAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,  # Pre-fused [3, B, H, M, D]
        scale: float,
        hilbert_map: torch.Tensor,
        dropout_p: float,
        is_causal: bool,
        segment_size: int,
        dilation_rate: int,
        training: bool,
    ):
        B, H, M, D = qkv.shape[1:]
        out = torch.empty((B, H, M, D), dtype=qkv.dtype, device=qkv.device)

        # Get optimal configuration
        config = get_fused_kernel_config(M, D, qkv.device)

        # Generate dropout mask if training
        dropout_mask = None
        if training and dropout_p > 0:
            dropout_mask = torch.rand((B, H, M, M), device=qkv.device) > dropout_p

        # Pad sequence length if needed
        M_padded = M
        if M % config["BLOCK_M"] != 0:
            M_padded = M + (config["BLOCK_M"] - M % config["BLOCK_M"])
            # Pad tensors
            pad_size = M_padded - M
            qkv = F.pad(qkv, (0, 0, 0, pad_size))
            out = F.pad(out, (0, 0, 0, pad_size))

        # Compute grid
        if config["process_multiple_rows"]:
            grid = (triton.cdiv(M_padded, config["BLOCK_M"] * 2) * B * H,)
        else:
            grid = (triton.cdiv(M_padded, config["BLOCK_M"]) * B * H,)

        # Launch kernel
        if config["use_v2"]:
            fused_hilbert_attention_kernel_v2[grid](
                qkv,
                out,
                hilbert_map,
                *qkv.stride(),
                *out.stride(),
                B,
                H,
                M_padded,
                D,
                scale,
                segment_size,
                dilation_rate,
                BLOCK_M=config["BLOCK_M"],
                BLOCK_N=config["BLOCK_N"],
                BLOCK_D=config["BLOCK_D"],
                USE_HILBERT=hilbert_map is not None,
                PROCESS_MULTIPLE_ROWS=config["process_multiple_rows"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
        else:
            # Use V1 kernel with standard parameters
            grid = (triton.cdiv(M_padded, config["BLOCK_M"]) * B * H,)
            fused_hilbert_attention_kernel[grid](
                qkv[0],
                qkv[1],
                qkv[2],
                out,
                hilbert_map,
                dropout_mask
                if dropout_mask is not None
                else qkv[0],  # Dummy if no dropout
                *qkv[0].stride(),  # Q strides
                *qkv[1].stride(),  # K strides
                *qkv[2].stride(),  # V strides
                *out.stride(),
                *(
                    out.stride() if dropout_mask is not None else (0, 0, 0, 0)
                ),  # Dropout strides
                B,
                H,
                M_padded,
                D,
                scale,
                dropout_p,
                is_causal,
                segment_size,
                dilation_rate,
                BLOCK_M=config["BLOCK_M"],
                BLOCK_N=config["BLOCK_N"],
                BLOCK_D=config["BLOCK_D"],
                USE_DROPOUT=dropout_mask is not None,
                USE_HILBERT=hilbert_map is not None,
                USE_SPARSE=dilation_rate > 1,
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :, :M, :]

        # Save for backward
        ctx.save_for_backward(qkv, out, hilbert_map)
        ctx.scale = scale
        ctx.dropout_p = dropout_p
        ctx.segment_size = segment_size
        ctx.dilation_rate = dilation_rate

        return out

    @staticmethod
    def backward(ctx, dout):
        # Backward pass would be implemented here
        # For now, fall back to PyTorch autograd
        raise NotImplementedError("Fused backward not yet implemented")
