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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Simplified Hilbert attention kernel - processes full sequence with Hilbert reordering."""
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
    m_i = (
        tl.zeros([BLOCK_M], dtype=tl.float32) - 1e6
    )  # Use large negative instead of -inf

    # Determine segment boundaries for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # Process all keys in the segment (simplified approach)
    for start_n in range(0, M, BLOCK_N):
        # Key indices
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Check if keys are in the same segment as queries and apply dilation
        in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
        dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
        mask_n = (offs_n < M) & in_segment & dilation_mask

        # Load Hilbert indices
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

        # Compute attention scores
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e6)

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


@triton.jit
def hilbert_attention_bwd_kernel(
    # Gradients
    dOut,
    dQ,
    dK,
    dV,
    # Forward tensors
    Q,
    K,
    V,
    Out,
    hilbert_map,
    # Strides
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dod,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dvd,
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
    """Backward pass kernel for Hilbert attention."""
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

    # Load queries and output gradients
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    dout_ptrs = (
        dOut
        + pid_b * stride_dob
        + pid_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_dod
    )
    dout = tl.load(dout_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize gradient accumulators
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Determine segment boundaries for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # First pass: recompute attention and accumulate dV
    for start_n in range(0, M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Check if keys are in the same segment as queries and apply dilation
        in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
        dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
        mask_n = (offs_n < M) & in_segment & dilation_mask

        # Load Hilbert indices
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

        # Recompute attention scores
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e6)

        # Compute softmax
        s_max = tl.max(s, axis=1)
        p = tl.exp(s - s_max[:, None])
        p_sum = tl.sum(p, axis=1)
        p = p / (p_sum[:, None] + 1e-10)

        # Compute gradients
        # dV += p^T @ dout
        _ = tl.dot(tl.trans(p), dout)

        # For dQ and dK, we need: dp = dout @ v^T
        v_t = tl.trans(v)
        dp = tl.dot(dout, v_t)

        # Softmax backward: ds = p * (dp - sum(p * dp))
        dp_sum = tl.sum(p * dp, axis=1)
        ds = p * (dp - dp_sum[:, None])

        # dQ += ds @ k^T * scale
        dq_acc += tl.dot(ds, tl.trans(k)) * scale

        # Accumulate dV gradient (atomic add needed in real implementation)
        # For now, we'll just compute it locally
        _ = (
            dV
            + pid_b * stride_dvb
            + pid_h * stride_dvh
            + h_idx[None, :] * stride_dvn
            + offs_d[:, None] * stride_dvd
        )
        # Note: This needs atomic operations for correctness across blocks
        # tl.atomic_add(dv_ptrs, dv_contrib, mask=mask_n[None, :] & mask_d[:, None])

    # Store dQ gradients
    dq_ptrs = (
        dQ
        + pid_b * stride_dqb
        + pid_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :] * stride_dqd
    )
    tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_d[None, :])


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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Standard attention kernel without Hilbert reordering."""
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
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e6

    # Determine segment boundaries for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # Process all keys in the segment
    for start_n in range(0, M, BLOCK_N):
        # Key indices
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Check if keys are in the same segment as queries and apply dilation
        in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
        dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
        mask_n = (offs_n < M) & in_segment & dilation_mask

        # Load keys and values directly (no Hilbert reordering)
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
        s = tl.where(mask_n[None, :], s, -1e6)

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
        BLOCK_N = min(64, M_padded)
        BLOCK_D = min(64, D)
        grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

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
            BLOCK_N,
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

        # For float16, we need to ensure computations are done in float32
        compute_dtype = torch.float32 if x.dtype == torch.float16 else x.dtype
        if x.dtype == torch.float16:
            qkv = qkv.to(compute_dtype)

        # Check if dimensions meet Triton requirements
        use_triton = self.head_dim >= 16 and M_padded >= 16 and x.device.type == "cuda"

        if use_hilbert and self.use_custom_backward and self.training and use_triton:
            # Use custom backward for training with Triton
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
        elif not use_triton and use_hilbert:
            # Fall back to PyTorch implementation for small dimensions
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Get Hilbert mapping
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)

            # Reorder K and V using Hilbert mapping
            hilbert_indices = hilbert_map.long()
            k_reordered = k.gather(
                2,
                hilbert_indices[None, None, :, None].expand(
                    B, H, M_padded, self.head_dim
                ),
            )
            v_reordered = v.gather(
                2,
                hilbert_indices[None, None, :, None].expand(
                    B, H, M_padded, self.head_dim
                ),
            )

            # Standard attention computation
            scores = torch.matmul(q, k_reordered.transpose(-2, -1)) * self.scale

            # Apply segment masking
            for i in range(0, M_padded, self.segment_size):
                segment_end = min(i + self.segment_size, M_padded)
                # Create mask for dilation
                if self.dilation_rate > 1:
                    for j in range(i, segment_end):
                        for k_idx in range(i, segment_end):
                            if (k_idx - i) % self.dilation_rate != 0:
                                scores[:, :, j, k_idx] = -1e9

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v_reordered)
        else:
            # Use standard forward (for inference or when custom backward disabled)
            q, k, v = qkv[0], qkv[1], qkv[2]
            out = torch.zeros_like(q)

            # Check if we can use Triton for standard attention
            if use_triton:
                # Configure grid for Triton
                BLOCK_M = min(64, M_padded)
                BLOCK_N = min(64, M_padded)
                BLOCK_D = min(64, self.head_dim)
                grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

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
                        BLOCK_N,
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
                        BLOCK_N,
                        BLOCK_D,
                    )
            else:
                # Fall back to PyTorch implementation for small dimensions
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

                # Apply segment masking
                for i in range(0, M_padded, self.segment_size):
                    segment_end = min(i + self.segment_size, M_padded)
                    # Apply dilation masking
                    if self.dilation_rate > 1:
                        for j in range(i, segment_end):
                            for k_idx in range(i, segment_end):
                                if (k_idx - i) % self.dilation_rate != 0:
                                    scores[:, :, j, k_idx] = -1e9

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                out = torch.matmul(attn_weights, v)

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Convert back to original dtype if needed
        if x.dtype == torch.float16:
            out = out.to(x.dtype)

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out
