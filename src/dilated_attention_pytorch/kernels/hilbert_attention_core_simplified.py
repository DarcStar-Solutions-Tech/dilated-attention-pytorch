#!/usr/bin/env python3
"""
Simplified Hilbert Attention implementation that Triton can compile.

This version removes complex control flow to ensure Triton compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def hilbert_attention_kernel_simple(
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
    """Simplified Hilbert attention kernel - full attention within each block."""
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

    # Process all keys in blocks
    for start_n in range(0, M, BLOCK_N):
        # Key indices
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < M

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


def create_hilbert_mapping(seq_len: int) -> torch.Tensor:
    """Create a simple Hilbert-like mapping."""
    if seq_len <= 64:
        return torch.arange(seq_len, dtype=torch.int32)

    # Simple snake pattern
    grid_size = int(math.ceil(math.sqrt(seq_len)))
    mapping = torch.arange(seq_len, dtype=torch.int32)

    idx = 0
    for row in range(grid_size):
        row_start = row * grid_size
        row_end = min(row_start + grid_size, seq_len)
        row_len = row_end - row_start

        if row % 2 == 0:
            # Left to right
            for i in range(row_len):
                if idx < seq_len:
                    mapping[idx] = row_start + i
                    idx += 1
        else:
            # Right to left (reverse)
            for i in range(row_len - 1, -1, -1):
                if idx < seq_len:
                    mapping[idx] = row_start + i
                    idx += 1

    return mapping


class HilbertAttentionCoreSimplified(nn.Module):
    """
    Simplified Hilbert Attention that Triton can compile.

    This version uses simpler kernels without complex control flow.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        use_custom_backward: bool = False,  # Ignored for now
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"
            )

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping."""
        if seq_len not in self._hilbert_cache:
            mapping = create_hilbert_mapping(seq_len)
            self._hilbert_cache[seq_len] = mapping.to(device)
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            use_hilbert: Whether to use Hilbert reordering

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        B, M, D = x.shape
        H = self.num_heads

        # Pad to multiple of block size
        BLOCK_SIZE = 64
        if M % BLOCK_SIZE != 0:
            pad_len = BLOCK_SIZE - (M % BLOCK_SIZE)
            x = F.pad(x, (0, 0, 0, pad_len))
            M_padded = M + pad_len
        else:
            M_padded = M

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get Hilbert mapping
        if use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
        else:
            hilbert_map = torch.arange(M_padded, device=x.device, dtype=torch.int32)

        # Allocate output
        out = torch.zeros_like(q)

        # Configure kernel
        BLOCK_M = min(64, M_padded)
        BLOCK_N = min(64, M_padded)
        BLOCK_D = min(64, self.head_dim)

        num_blocks_m = triton.cdiv(M_padded, BLOCK_M)
        grid = (num_blocks_m * B * H,)

        # Launch kernel
        hilbert_attention_kernel_simple[grid](
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
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
        )

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


# For backward compatibility
HilbertAttentionCore = HilbertAttentionCoreSimplified
HilbertAttentionFunction = None  # Not implemented in simplified version
