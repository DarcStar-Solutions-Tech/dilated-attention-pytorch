#!/usr/bin/env python3
"""
Unified Hilbert Attention implementation combining all optimizations.

This module consolidates the best features from multiple Hilbert implementations:
- Core Triton kernels from hilbert_dilated_attention_triton_fixed.py
- Optimized backward pass from hilbert_attention_triton_fixed_optimized.py
- Simplified interface and caching strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def hilbert_attention_kernel(
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
    BLOCK_D: tl.constexpr,
):
    """Unified Hilbert attention forward kernel."""
    # Program ID
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Mask for valid queries
    mask_m = offs_m < M
    mask_d = offs_d < D

    # Get Hilbert positions for queries in this block
    hilbert_pos_q = tl.load(hilbert_map + offs_m, mask=mask_m, other=0)

    # Load queries using Hilbert positions
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos_q[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Scale queries
    q = q * scale

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-10

    # For each query position, compute its segment
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys in the segment with dilation
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset

        # Check if key position is valid
        mask_k = (key_pos < M) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            # Get Hilbert position for this key
            key_hilbert = tl.load(hilbert_map + key_pos, mask=mask_k, other=0)

            # Load key and value using Hilbert position
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + key_hilbert * stride_kn
                + offs_d * stride_kd
            )
            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + key_hilbert * stride_vn
                + offs_d * stride_vd
            )

            k = tl.load(k_ptrs, mask=mask_k & mask_d, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k & mask_d, other=0.0)

            # Compute attention scores for all queries against this key
            scores = tl.sum(q * k[None, :], axis=1)

            # Mask invalid scores
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            # Stable softmax accumulation
            scores_max = tl.max(scores, axis=0)
            scores_exp = tl.exp(scores - scores_max)

            # Update accumulator
            acc += scores_exp[:, None] * v[None, :]
            norm += scores_exp

    # Normalize
    out = acc / norm[:, None]

    # Store output back to original positions
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def standard_attention_kernel(
    # Same signature as hilbert kernel but without hilbert_map
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
    BLOCK_D: tl.constexpr,
):
    """Standard attention kernel without Hilbert reordering."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    # Load queries directly (no Hilbert mapping)
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
    norm = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-10

    # Compute segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys with dilation
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset

        mask_k = (key_pos < M) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + key_pos * stride_kn
                + offs_d * stride_kd
            )
            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + key_pos * stride_vn
                + offs_d * stride_vd
            )

            k = tl.load(k_ptrs, mask=mask_k & mask_d, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k & mask_d, other=0.0)

            scores = tl.sum(q * k[None, :], axis=1)
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            scores_max = tl.max(scores, axis=0)
            scores_exp = tl.exp(scores - scores_max)

            acc += scores_exp[:, None] * v[None, :]
            norm += scores_exp

    out = acc / norm[:, None]

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


class HilbertAttentionFunction(torch.autograd.Function):
    """Custom autograd function with optimized backward pass."""

    @staticmethod
    def forward(
        ctx,
        qkv,
        scale,
        hilbert_map,
        segment_size,
        dilation_rate,
        M_padded,
        M_orig,
        B,
        H,
        D,
    ):
        """Forward pass using Triton kernel."""
        # Split QKV
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Allocate output
        out = torch.zeros_like(q)

        # Configure grid
        BLOCK_M = min(64, M_padded)
        BLOCK_D = min(64, D)
        grid = (triton.cdiv(M_padded, BLOCK_M), B * H)

        # Launch forward kernel
        hilbert_attention_kernel[grid](
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
            M_padded,
            D,
            scale,
            segment_size,
            dilation_rate,
            BLOCK_M,
            BLOCK_D,
        )

        # Save reordered tensors for efficient backward
        # Create inverse mapping
        inverse_map = torch.zeros_like(hilbert_map)
        inverse_map[hilbert_map] = torch.arange(
            len(hilbert_map), device=hilbert_map.device, dtype=hilbert_map.dtype
        )

        # Reorder tensors once for backward pass
        hilbert_map_long = hilbert_map.long()
        q_reordered = q.gather(
            2, hilbert_map_long[None, None, :, None].expand(B, H, M_padded, D)
        )
        k_reordered = k.gather(
            2, hilbert_map_long[None, None, :, None].expand(B, H, M_padded, D)
        )
        v_reordered = v.gather(
            2, hilbert_map_long[None, None, :, None].expand(B, H, M_padded, D)
        )

        ctx.save_for_backward(
            q_reordered, k_reordered, v_reordered, out, hilbert_map, inverse_map
        )
        ctx.scale = scale
        ctx.segment_size = segment_size
        ctx.dilation_rate = dilation_rate
        ctx.M_padded = M_padded
        ctx.M_orig = M_orig

        return out

    @staticmethod
    def backward(ctx, dout):
        """Optimized backward pass using PyTorch operations."""
        q_reordered, k_reordered, v_reordered, out, hilbert_map, inverse_map = (
            ctx.saved_tensors
        )
        B, H, N, D = q_reordered.shape
        scale = ctx.scale
        segment_size = ctx.segment_size
        dilation_rate = ctx.dilation_rate

        # Reshape for efficient computation
        q_r = q_reordered.reshape(B * H, N, D) * scale
        k_r = k_reordered.reshape(B * H, N, D)
        v_r = v_reordered.reshape(B * H, N, D)
        dout_flat = dout.reshape(B * H, N, D)

        # Initialize gradients
        dq_reordered = torch.zeros_like(q_r)
        dk_reordered = torch.zeros_like(k_r)
        dv_reordered = torch.zeros_like(v_r)

        # Process each segment efficiently
        for i in range(0, N, segment_size):
            seg_end = min(i + segment_size, N)
            seg_len = seg_end - i

            # Create dilation mask if needed
            if dilation_rate > 1:
                active_positions = torch.arange(
                    0, seg_len, dilation_rate, device=q_r.device
                )
                mask = torch.zeros(
                    seg_len, seg_len, dtype=torch.bool, device=q_r.device
                )
                mask[:, active_positions] = True
                mask[active_positions, :] = True
                attn_mask = mask.unsqueeze(0)
            else:
                attn_mask = None

            # Extract segment
            q_seg = q_r[:, i:seg_end]
            k_seg = k_r[:, i:seg_end]
            v_seg = v_r[:, i:seg_end]
            dout_seg = dout_flat[:, i:seg_end]

            # Recompute attention weights
            scores = torch.bmm(q_seg, k_seg.transpose(-2, -1))

            if attn_mask is not None:
                scores.masked_fill_(~attn_mask, float("-inf"))

            # Stable softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Gradient computations
            dv_reordered[:, i:seg_end] += torch.bmm(
                attn_weights.transpose(-2, -1), dout_seg
            )

            dattn = torch.bmm(dout_seg, v_seg.transpose(-2, -1))
            dattn_weights = attn_weights * (
                dattn - (dattn * attn_weights).sum(dim=-1, keepdim=True)
            )

            dq_reordered[:, i:seg_end] += torch.bmm(dattn_weights, k_seg) * scale
            dk_reordered[:, i:seg_end] += torch.bmm(
                dattn_weights.transpose(-2, -1), q_seg
            )

        # Reshape back
        dq_reordered = dq_reordered.reshape(B, H, N, D)
        dk_reordered = dk_reordered.reshape(B, H, N, D)
        dv_reordered = dv_reordered.reshape(B, H, N, D)

        # Reverse Hilbert reordering
        inverse_map_long = inverse_map.long()
        dq = dq_reordered.gather(
            2, inverse_map_long[None, None, :, None].expand(B, H, N, D)
        )
        dk = dk_reordered.gather(
            2, inverse_map_long[None, None, :, None].expand(B, H, N, D)
        )
        dv = dv_reordered.gather(
            2, inverse_map_long[None, None, :, None].expand(B, H, N, D)
        )

        # Combine gradients for QKV
        dqkv = torch.stack([dq, dk, dv], dim=0)

        return dqkv, None, None, None, None, None, None, None, None, None


def create_hilbert_mapping(seq_len: int) -> torch.Tensor:
    """Create Hilbert curve mapping for sequences."""
    # For simplicity, using snake pattern (similar to Hilbert curve properties)
    # Can be replaced with true Hilbert curve if needed

    if seq_len <= 64:
        return torch.arange(seq_len, dtype=torch.int32)

    grid_size = int(math.ceil(math.sqrt(seq_len)))
    mapping = torch.zeros(seq_len, dtype=torch.long)
    idx = 0

    for row in range(grid_size):
        if row % 2 == 0:
            # Left to right
            for col in range(grid_size):
                if idx < seq_len:
                    linear_pos = row * grid_size + col
                    if linear_pos < seq_len:
                        mapping[linear_pos] = idx
                        idx += 1
        else:
            # Right to left (snake pattern)
            for col in range(grid_size - 1, -1, -1):
                if idx < seq_len:
                    linear_pos = row * grid_size + col
                    if linear_pos < seq_len:
                        mapping[linear_pos] = idx
                        idx += 1

    # Fill any remaining positions
    for i in range(seq_len):
        if i >= idx:
            mapping[i] = i

    return mapping.int()


class HilbertAttentionCore(nn.Module):
    """
    Unified Hilbert Attention implementation with all optimizations.

    This consolidates the best features from all Hilbert implementations:
    - Efficient Triton kernels for forward pass
    - Optimized PyTorch backward pass
    - Configurable custom backward
    - Hilbert mapping caching
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        use_custom_backward: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_custom_backward = use_custom_backward

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        if seq_len not in self._hilbert_cache:
            mapping = create_hilbert_mapping(seq_len)
            self._hilbert_cache[seq_len] = mapping.to(device)
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Forward pass with optional Hilbert ordering.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            use_hilbert: Whether to use Hilbert curve reordering

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

        if use_hilbert and self.use_custom_backward and self.training:
            # Use custom backward for training
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
            out = HilbertAttentionFunction.apply(
                qkv,
                self.scale,
                hilbert_map,
                self.segment_size,
                self.dilation_rate,
                M_padded,
                M,
                B,
                H,
                self.head_dim,
            )
        else:
            # Use standard forward (for inference or when custom backward disabled)
            q, k, v = qkv[0], qkv[1], qkv[2]
            out = torch.zeros_like(q)

            # Configure grid
            BLOCK_M = min(64, M_padded)
            BLOCK_D = min(64, self.head_dim)
            grid = (triton.cdiv(M_padded, BLOCK_M), B * H)

            if use_hilbert:
                hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
                hilbert_attention_kernel[grid](
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
                    M_padded,
                    self.head_dim,
                    self.scale,
                    self.segment_size,
                    self.dilation_rate,
                    BLOCK_M,
                    BLOCK_D,
                )
            else:
                standard_attention_kernel[grid](
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
                )

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


# Backward compatibility aliases
HilbertAttentionTritonFixed = HilbertAttentionCore
HilbertAttentionTritonOptimized = HilbertAttentionCore
