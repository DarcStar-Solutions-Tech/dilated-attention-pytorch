#!/usr/bin/env python3
"""
Simplified optimized Dilated Attention that's easier for Triton to compile.

Key optimization: Process dilated pattern more efficiently without complex control flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def dilated_attention_kernel_simple(
    # Pointers
    Q,
    K,
    V,
    Out,
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
    """Simplified dilated attention kernel with better Triton compatibility."""
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

    # Load queries for this block
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    # Initialize output accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment boundaries for this query block
    # All queries in a block should be in the same or adjacent segments
    first_query_pos = pid_m * BLOCK_M
    first_seg_idx = first_query_pos // segment_size

    # Process at most 2 segments (current and next)
    for seg_offset in range(2):
        seg_idx = first_seg_idx + seg_offset
        seg_start = seg_idx * segment_size
        seg_end = tl.minimum(seg_start + segment_size, M)

        # Skip if segment is beyond sequence
        seg_valid = seg_start < M

        # Process keys in dilated pattern
        # Key optimization: step by dilation_rate * BLOCK_N for efficiency
        dilated_step = dilation_rate * BLOCK_N

        for start_pos in range(seg_start, seg_end, dilated_step):
            # Load a block of keys with dilation
            offs_n = start_pos + tl.arange(0, BLOCK_N) * dilation_rate

            # Mask for valid positions
            mask_n = (
                (offs_n >= seg_start) & (offs_n < seg_end) & (offs_n < M) & seg_valid
            )

            # Load keys and values
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

            # Compute attention scores
            s = tl.dot(q, k)

            # Apply segment mask - vectorized approach
            # Create segment mask for all queries at once
            query_seg = offs_m // segment_size
            seg_match = query_seg[:, None] == seg_idx

            # Combine masks
            s = tl.where(seg_match & mask_n[None, :], s, -1e9)

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
    acc = acc / (l_i[:, None] + 1e-10)

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


class DilatedAttentionSimpleOpt(nn.Module):
    """
    Simplified optimized Dilated Attention with better Triton compatibility.

    This version focuses on processing efficiency while maintaining
    compatibility with Triton's compilation requirements.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Validate inputs
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with simplified optimized dilated attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        B, M, D = x.shape
        H = self.num_heads

        # Ensure sequence length is compatible with segment size
        if M % self.segment_size != 0:
            pad_len = self.segment_size - (M % self.segment_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            M_padded = M + pad_len
        else:
            M_padded = M

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        # Check if we should use Triton
        use_triton = (
            self.head_dim >= 16
            and M_padded >= 16
            and x.device.type == "cuda"
            and qkv.dtype in [torch.float32, torch.float16]
        )

        if use_triton:
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Convert to float32 if needed
            compute_dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
            if q.dtype == torch.float16:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)

            # Allocate output
            out = torch.zeros_like(q)

            # Configure grid with optimized block sizes
            # For dilated attention, we need to ensure BLOCK_N is at least 16
            active_per_segment = self.segment_size // self.dilation_rate
            BLOCK_M = min(64, M_padded)
            BLOCK_N = (
                max(16, min(32, active_per_segment))
                if self.dilation_rate > 1
                else min(64, M_padded)
            )
            BLOCK_D = min(64, self.head_dim)

            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            # Launch simplified kernel
            dilated_attention_kernel_simple[grid](
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
                BLOCK_N,
                BLOCK_D,
            )

            # Convert back if needed
            if qkv.dtype == torch.float16:
                out = out.to(torch.float16)
        else:
            # Fall back to PyTorch implementation
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply dilated masking
            if self.dilation_rate > 1:
                mask = torch.zeros(
                    M_padded, M_padded, device=x.device, dtype=torch.bool
                )
                for i in range(M_padded):
                    seg_idx = i // self.segment_size
                    seg_start = seg_idx * self.segment_size
                    seg_end = min(seg_start + self.segment_size, M_padded)

                    for j in range(seg_start, seg_end, self.dilation_rate):
                        mask[i, j] = True

                scores = scores.masked_fill(
                    ~mask.unsqueeze(0).unsqueeze(0), -float("inf")
                )

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out
