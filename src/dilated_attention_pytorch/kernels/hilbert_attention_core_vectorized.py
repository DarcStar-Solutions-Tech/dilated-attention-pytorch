#!/usr/bin/env python3
"""
Vectorized Hilbert Attention implementation using Triton-compatible patterns.

This version uses proper vectorized operations that Triton can compile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def hilbert_attention_fwd_kernel(
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
    """Simplified attention kernel using block-wise computation."""
    # Get program IDs
    pid = tl.program_id(0)
    pid_m = pid // (B * H)
    pid_bh = pid % (B * H)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    # Create masks
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

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over blocks of keys/values
    for start_n in range(0, M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Key/value indices
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < M

        # For Hilbert ordering, map the indices
        # Note: For simplicity, we'll apply Hilbert mapping to keys/values
        # In practice, you might want to reorder Q,K,V before calling the kernel
        hilbert_idx = tl.load(hilbert_map + offs_n_curr, mask=mask_n, other=0)

        # Load keys and values using Hilbert indices
        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + hilbert_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + hilbert_idx[None, :] * stride_vn
            + offs_d[:, None] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        # Compute attention scores: S = Q @ K^T
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, k)

        # Apply mask for invalid positions
        s = tl.where(mask_n[None, :], s, float("-inf"))

        # Compute numerically stable softmax
        m_ij = tl.max(s, axis=1)
        s = s - m_ij[:, None]
        p = tl.exp(s)
        l_ij = tl.sum(p, axis=1)

        # Update running statistics
        m_i = tl.maximum(l_i, m_ij)
        l_i = tl.exp(l_i - m_i) * l_i + tl.exp(m_ij - m_i) * l_ij

        # Update accumulator
        p_scale = tl.exp(m_ij - m_i)
        acc = acc * p_scale[:, None]

        # Transpose v for matmul: [BLOCK_D, BLOCK_N]
        v_t = tl.trans(v)
        acc += tl.dot(p, v_t)

    # Final normalization
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


class HilbertAttentionVectorized(nn.Module):
    """
    Vectorized Hilbert Attention using proper Triton patterns.

    This implementation uses block-wise matrix multiplications that Triton
    can efficiently compile and execute.
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

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def create_hilbert_mapping(self, seq_len: int) -> torch.Tensor:
        """Create a simple Hilbert-like mapping."""
        if seq_len <= 64:
            return torch.arange(seq_len, dtype=torch.int32)

        # Simple snake pattern as Hilbert approximation
        grid_size = int(math.ceil(math.sqrt(seq_len)))
        mapping = torch.zeros(seq_len, dtype=torch.int32)
        idx = 0

        for row in range(grid_size):
            if row % 2 == 0:
                for col in range(grid_size):
                    if idx < seq_len:
                        mapping[idx] = row * grid_size + col
                        idx += 1
            else:
                for col in range(grid_size - 1, -1, -1):
                    if idx < seq_len:
                        mapping[idx] = row * grid_size + col
                        idx += 1

        return mapping

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping."""
        if seq_len not in self._hilbert_cache:
            mapping = self.create_hilbert_mapping(seq_len)
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

        # Pad sequence length to multiple of block size
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
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(64, self.head_dim)

        grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

        # Launch kernel
        hilbert_attention_fwd_kernel[grid](
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
def create_hilbert_mapping(seq_len: int) -> torch.Tensor:
    """Create Hilbert curve mapping for sequences."""
    if seq_len <= 64:
        return torch.arange(seq_len, dtype=torch.int32)

    grid_size = int(math.ceil(math.sqrt(seq_len)))
    mapping = torch.zeros(seq_len, dtype=torch.int32)
    idx = 0

    for row in range(grid_size):
        if row % 2 == 0:
            for col in range(grid_size):
                if idx < seq_len:
                    mapping[idx] = row * grid_size + col
                    idx += 1
        else:
            for col in range(grid_size - 1, -1, -1):
                if idx < seq_len:
                    mapping[idx] = row * grid_size + col
                    idx += 1

    return mapping
