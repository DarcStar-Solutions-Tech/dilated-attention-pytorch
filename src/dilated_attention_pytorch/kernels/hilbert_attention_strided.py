#!/usr/bin/env python3
"""
Hilbert Attention with optimized strided access for dilated attention.

This implementation significantly reduces memory bandwidth usage by only
processing the positions that are actually used in dilated attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple

from .cache_manager import BoundedCache


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
    """Strided Hilbert attention kernel optimized for dilated patterns.

    Key optimization: Only process positions that match the dilation pattern,
    significantly reducing memory bandwidth usage.
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

    # Determine segment boundaries for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # STRIDED ACCESS OPTIMIZATION:
    # Instead of processing all positions and masking, we directly compute
    # only the positions that match our dilation pattern

    if dilation_rate == 1:
        # No dilation - standard processing
        for start_n in range(0, M, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)

            # Only process keys in the same segment
            in_segment = (offs_n >= seg_start) & (offs_n < seg_end)
            mask_n = (offs_n < M) & in_segment

            if not tl.sum(mask_n):
                continue

            # Load and process
            h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

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
            l_i = alpha * l_i + l_ij
            m_i = m_i_new

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, tl.trans(v))
    else:
        # DILATED ATTENTION - STRIDED ACCESS
        # Process each dilation offset separately to maximize memory efficiency

        # Calculate effective positions per block considering dilation
        positions_per_block = BLOCK_N // dilation_rate
        if positions_per_block < 1:
            positions_per_block = 1

        # Process each offset within the dilation pattern
        for offset in range(dilation_rate):
            # Start from the first valid position for this offset
            start_pos = seg_start + offset
            if start_pos >= seg_end:
                continue

            # Process positions with stride = dilation_rate
            for pos in range(start_pos, seg_end, dilation_rate * positions_per_block):
                # Create indices for this strided block
                indices = tl.arange(0, positions_per_block)
                actual_positions = pos + indices * dilation_rate

                # Mask for valid positions
                valid_mask = actual_positions < seg_end

                if not tl.sum(valid_mask):
                    continue

                # Load Hilbert indices for these specific positions
                h_idx = tl.load(
                    hilbert_map + actual_positions, mask=valid_mask, other=0
                )

                # Compute pointers with proper broadcasting
                # We need to handle the fact that we have fewer positions than BLOCK_N
                k_base = K + pid_b * stride_kb + pid_h * stride_kh
                v_base = V + pid_b * stride_vb + pid_h * stride_vh

                # Create expanded masks for the attention computation
                # Since we have positions_per_block positions, we need to pad
                if positions_per_block < BLOCK_N:
                    # Pad the indices and masks
                    padded_h_idx = tl.zeros([BLOCK_N], dtype=h_idx.dtype)
                    padded_mask = tl.zeros([BLOCK_N], dtype=tl.int1)

                    # Copy valid values
                    padded_h_idx = tl.where(
                        tl.arange(0, BLOCK_N) < positions_per_block,
                        h_idx[tl.arange(0, BLOCK_N) % positions_per_block],
                        0,
                    )
                    padded_mask = tl.where(
                        tl.arange(0, BLOCK_N) < positions_per_block,
                        valid_mask[tl.arange(0, BLOCK_N) % positions_per_block],
                        False,
                    )

                    h_idx_use = padded_h_idx
                    mask_use = padded_mask
                else:
                    h_idx_use = h_idx
                    mask_use = valid_mask

                # Load K and V for these specific positions
                k_ptrs = (
                    k_base
                    + h_idx_use[None, :] * stride_kn
                    + offs_d[:, None] * stride_kd
                )
                v_ptrs = (
                    v_base
                    + h_idx_use[None, :] * stride_vn
                    + offs_d[:, None] * stride_vd
                )

                k = tl.load(k_ptrs, mask=mask_use[None, :] & mask_d[:, None], other=0.0)
                v = tl.load(v_ptrs, mask=mask_use[None, :] & mask_d[:, None], other=0.0)

                # Compute attention scores
                s = tl.dot(q, k)
                s = tl.where(mask_use[None, :], s, -1e9)

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


class HilbertAttentionStrided(nn.Module):
    """
    Hilbert Attention with optimized strided access for dilated patterns.

    This implementation reduces memory bandwidth by only processing positions
    that are actually used in the dilated attention pattern.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        use_custom_backward: bool = False,
    ):
        super().__init__()

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
        # Initialize bounded cache
        self._hilbert_cache = BoundedCache(
            max_size=32, max_memory_mb=100.0, name=f"{self.__class__.__name__}_hilbert"
        )
        self._device_capability = None

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        # Try to get from cache
        mapping = self._hilbert_cache.get(seq_len)

        if mapping is None or mapping.device != device:
            from .hilbert_attention_core import create_hilbert_mapping

            # Create new mapping
            mapping = create_hilbert_mapping(seq_len).to(device)
            # Store in cache (will handle LRU eviction if needed)
            self._hilbert_cache.put(seq_len, mapping)

        return mapping

    def get_optimal_block_sizes(
        self, seq_len: int, device: torch.device
    ) -> Tuple[int, int, int]:
        """Get optimal block sizes for strided access."""
        if self._device_capability is None and device.type == "cuda":
            self._device_capability = torch.cuda.get_device_capability(device)

        compute_capability = (
            self._device_capability[0] if self._device_capability else 6
        )

        # For strided access, we may want different block sizes
        # especially for BLOCK_N when dilation_rate > 1
        if compute_capability < 7:  # Pascal
            BLOCK_M = min(32, seq_len)
            # Adjust BLOCK_N based on dilation rate for better efficiency
            BLOCK_N = min(32, max(16, 32 // self.dilation_rate) * self.dilation_rate)
            BLOCK_D = min(32, self.head_dim)
        elif compute_capability < 8:  # Volta/Turing
            BLOCK_M = min(64, seq_len)
            BLOCK_N = min(64, max(32, 64 // self.dilation_rate) * self.dilation_rate)
            BLOCK_D = min(64, self.head_dim)
        else:  # Ampere+
            BLOCK_M = min(128, seq_len)
            BLOCK_N = min(128, max(64, 128 // self.dilation_rate) * self.dilation_rate)
            BLOCK_D = min(128, self.head_dim)

        # Ensure minimum sizes
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_D = max(16, BLOCK_D)

        return BLOCK_M, BLOCK_N, BLOCK_D

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """Forward pass with strided access optimization."""
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

        # Get Hilbert mapping
        if use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
        else:
            hilbert_map = torch.arange(M_padded, device=x.device, dtype=torch.int32)

        # Check if we can use Triton
        if self.head_dim >= 16 and M_padded >= 16 and x.device.type == "cuda":
            # Use strided kernel
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
        else:
            # Fallback to PyTorch implementation
            out = self._pytorch_forward(q, k, v, hilbert_map)

        # Reshape and remove padding
        out = out.transpose(1, 2).contiguous()
        out = out.reshape(B, M_padded, D)
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

    def _pytorch_forward(self, q, k, v, hilbert_map):
        """PyTorch fallback implementation."""
        B, H, M, D = q.shape

        # Reorder k and v using Hilbert mapping
        k = k.gather(2, hilbert_map[None, None, :, None].expand(B, H, M, D))
        v = v.gather(2, hilbert_map[None, None, :, None].expand(B, H, M, D))

        # Compute attention with dilation
        out = torch.zeros_like(q)

        for seg_start in range(0, M, self.segment_size):
            seg_end = min(seg_start + self.segment_size, M)

            # Get segment queries
            q_seg = q[:, :, seg_start:seg_end, :]

            if self.dilation_rate == 1:
                # No dilation
                k_seg = k[:, :, seg_start:seg_end, :]
                v_seg = v[:, :, seg_start:seg_end, :]

                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale
                attn = F.softmax(scores, dim=-1)
                out[:, :, seg_start:seg_end, :] = torch.matmul(attn, v_seg)
            else:
                # Dilated attention - only attend to dilated positions
                _ = seg_end - seg_start
                out_seg = torch.zeros_like(q_seg)

                # Process each dilation offset
                for offset in range(self.dilation_rate):
                    positions = torch.arange(
                        seg_start + offset, seg_end, self.dilation_rate, device=q.device
                    )
                    if len(positions) == 0:
                        continue

                    # Get dilated k, v
                    k_dilated = k[:, :, positions, :]
                    v_dilated = v[:, :, positions, :]

                    # All queries attend to these dilated positions
                    scores = (
                        torch.matmul(q_seg, k_dilated.transpose(-2, -1)) * self.scale
                    )
                    attn = F.softmax(scores, dim=-1)
                    out_seg += torch.matmul(attn, v_dilated)

                out[:, :, seg_start:seg_end, :] = out_seg / self.dilation_rate

        return out
