#!/usr/bin/env python3
"""
Optimized Dilated Attention Triton kernel V2.

Key improvements:
1. Computes sparse attention pattern more efficiently
2. Better memory access patterns
3. Optimized for common dilation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def dilated_attention_kernel_v2(
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
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DILATION: tl.constexpr,  # Process multiple dilated positions at once
):
    """Optimized dilated attention kernel."""
    # Program ID
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    mask_m = offs_m < M
    mask_d = offs_d < D
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0) * scale

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)

    # Determine segment for this query block
    # Optimization: Process segment boundaries more efficiently
    query_seg_start = (pid_m * BLOCK_M // segment_size) * segment_size
    _ = query_seg_start + segment_size

    # Key optimization: Process dilated positions in groups
    # Instead of checking every position, jump by dilation_rate
    num_active_per_segment = segment_size // dilation_rate

    # Process keys in dilated pattern
    # Process up to 2 segments (most query blocks span at most 2 segments)
    for seg_idx in range(2):
        seg_start = query_seg_start + seg_idx * segment_size
        seg_mask = seg_start < M

        # Only process if segment is valid
        if tl.where(seg_mask, 1, 0):
            seg_end = tl.minimum(seg_start + segment_size, M)

            # Process dilated positions in groups of BLOCK_DILATION
            for dil_group in range(0, num_active_per_segment, BLOCK_DILATION):
                # Calculate actual key positions for this dilated group
                offs_n = (
                    seg_start
                    + (dil_group + tl.arange(0, BLOCK_DILATION)) * dilation_rate
                )
                mask_n = offs_n < seg_end

                # Load keys and values at dilated positions
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

                k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

                # Compute scores
                scores = tl.dot(q, k)  # [BLOCK_M, BLOCK_DILATION]

                # Apply segment mask
                for m_idx in range(BLOCK_M):
                    query_pos = pid_m * BLOCK_M + m_idx
                    query_seg = query_pos // segment_size
                    key_seg = seg_start // segment_size

                    if query_seg != key_seg:
                        scores[m_idx, :] = -float("inf")

                # Apply valid position mask
                scores = tl.where(mask_n[None, :], scores, -float("inf"))

                # Online softmax
                m_ij = tl.max(scores, axis=1)
                m_i_new = tl.maximum(m_i, m_ij)
                p = tl.exp(scores - m_i_new[:, None])
                l_ij = tl.sum(p, axis=1)

                # Update accumulators
                alpha = tl.exp(m_i - m_i_new)
                acc = acc * alpha[:, None]
                l_i = alpha * l_i + l_ij
                m_i = m_i_new

                # Accumulate weighted values
                acc += tl.dot(p, tl.trans(v))

    # Normalize
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


class DilatedAttentionV2(nn.Module):
    """Optimized dilated attention using improved Triton kernel."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, M, D = x.shape
        H = self.num_heads

        # Pad if needed
        if M % self.segment_size != 0:
            pad_len = self.segment_size - (M % self.segment_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            M_padded = M + pad_len
        else:
            M_padded = M

        # QKV projection
        qkv = self.qkv_proj(x).reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Check if we can use Triton
        if self.head_dim >= 16 and x.device.type == "cuda":
            out = torch.zeros_like(q)

            # Optimized block sizes
            BLOCK_M = 64 if M_padded > 512 else 32
            BLOCK_D = min(64, self.head_dim)
            BLOCK_DILATION = min(16, self.segment_size // self.dilation_rate)

            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            dilated_attention_kernel_v2[grid](
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
                M_padded,
                self.head_dim,
                self.scale,
                self.segment_size,
                self.dilation_rate,
                BLOCK_M,
                BLOCK_D,
                BLOCK_DILATION,
            )
        else:
            # PyTorch fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Create dilated mask
            mask = torch.zeros(M_padded, M_padded, device=x.device, dtype=torch.bool)
            for i in range(M_padded):
                seg_idx = i // self.segment_size
                seg_start = seg_idx * self.segment_size
                seg_end = min(seg_start + self.segment_size, M_padded)

                for j in range(seg_start, seg_end, self.dilation_rate):
                    mask[i, j] = True

            scores.masked_fill_(~mask, -float("inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, M_padded, D)
        if M_padded > M:
            out = out[:, :M]

        out = self.out_proj(out)
        out = self.dropout(out)

        return out
