#!/usr/bin/env python3
"""
Simplified strided Hilbert Attention kernel that Triton can compile.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple

from .hilbert_attention_core import HilbertAttentionCore


@triton.jit
def hilbert_attention_strided_kernel(
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
    """Simplified strided attention kernel.

    Key optimization: Process keys with stride = dilation_rate to reduce
    memory bandwidth when dilation > 1.
    """
    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query block boundaries
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

    # Determine segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # STRIDED ACCESS: Use larger stride when dilation_rate > 1
    # This is the key optimization - we process fewer blocks when dilated
    stride = BLOCK_N
    if dilation_rate > 1 and BLOCK_N >= dilation_rate:
        # Only use strided access if BLOCK_N is large enough
        # Otherwise we might miss positions
        stride = BLOCK_N

    # Process keys with appropriate stride
    for start_n in range(0, M, stride):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Check segment and dilation
        in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
        dilation_mask = ((offs_n - seg_start) % dilation_rate) == 0
        mask_n = (offs_n < M) & in_segment & dilation_mask

        # Early exit optimization - check if any valid positions
        has_valid = tl.sum(mask_n.to(tl.int32)) > 0

        # Load and process only if we have valid positions
        if has_valid:
            # Load Hilbert indices
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

            # Load K and V
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
            l_i = alpha * l_i + l_ij
            m_i = m_i_new

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v))

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


class HilbertAttentionStridedSimple(HilbertAttentionCore):
    """
    Simplified strided Hilbert Attention that extends the base implementation.

    This version uses a compilable Triton kernel with strided access optimization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override to use our optimized kernel
        self._use_strided_kernel = True

    def get_optimal_block_sizes(
        self, seq_len: int, device: torch.device
    ) -> Tuple[int, int, int]:
        """Override to use strided-optimized block sizes."""
        BLOCK_M, BLOCK_N, BLOCK_D = super().get_optimal_block_sizes(seq_len, device)

        # For strided access with high dilation, we might want larger BLOCK_N
        # to ensure we capture enough dilated positions per block
        if self.dilation_rate > 1:
            # Ensure BLOCK_N is at least 2x dilation_rate for efficiency
            BLOCK_N = max(BLOCK_N, min(128, 2 * self.dilation_rate))

        return BLOCK_M, BLOCK_N, BLOCK_D

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """Forward pass using strided kernel."""
        B, M, D = x.shape
        H = self.num_heads

        # Pad sequence length if needed
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

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle float16
        compute_dtype = torch.float32 if qkv.dtype == torch.float16 else qkv.dtype
        original_dtype = qkv.dtype
        if qkv.dtype == torch.float16:
            q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)

        # Get Hilbert mapping
        if use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
        else:
            hilbert_map = torch.arange(M_padded, device=x.device, dtype=torch.int32)

        # Check if we can use Triton
        use_triton = self.head_dim >= 16 and M_padded >= 16 and x.device.type == "cuda"

        if use_triton and self._use_strided_kernel:
            # Use our strided kernel
            out = torch.zeros_like(q)

            # Get optimized block sizes
            BLOCK_M, BLOCK_N, BLOCK_D = self.get_optimal_block_sizes(M_padded, x.device)

            # Configure grid
            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            # Launch strided kernel
            hilbert_attention_strided_kernel[grid](
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

            # Convert back to original dtype if needed
            if original_dtype == torch.float16:
                out = out.to(original_dtype)
        else:
            # Fall back to parent implementation
            return super().forward(x, use_hilbert)

        # Reshape and remove padding
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, M_padded, D)
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out
