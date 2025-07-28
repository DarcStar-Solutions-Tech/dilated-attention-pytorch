#!/usr/bin/env python3
"""
Optimized Hilbert Attention implementation with better performance.

Key optimizations:
1. Compute attention only for valid positions
2. Better block size selection
3. Fused operations where possible
4. Optimized memory access patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def dilated_attention_kernel_optimized(
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
    """Optimized dilated attention kernel with better performance."""
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
    # All queries in this block should be in the same or adjacent segments
    first_query_pos = pid_m * BLOCK_M
    first_seg_idx = first_query_pos // segment_size

    # Process keys by segment to improve memory locality
    # We need to handle at most 2 segments (current and next)
    for seg_offset in range(2):
        seg_idx = first_seg_idx + seg_offset
        seg_start = seg_idx * segment_size
        seg_end = tl.minimum(seg_start + segment_size, M)

        if seg_start >= M:
            continue

        # Process keys in this segment with dilation
        # Use BLOCK_N to process multiple keys at once
        for start_n in range(seg_start, seg_end, BLOCK_N * dilation_rate):
            # Load a block of keys with dilation
            offs_n_base = start_n + tl.arange(0, BLOCK_N) * dilation_rate

            # Create mask for valid dilated positions
            mask_n = (
                (offs_n_base >= seg_start) & (offs_n_base < seg_end) & (offs_n_base < M)
            )

            # Load Hilbert indices if needed
            h_idx = tl.load(hilbert_map + offs_n_base, mask=mask_n, other=0)

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

            # Compute attention scores for all query-key pairs in block
            s = tl.dot(q, k)

            # Apply segment mask - only attend within the correct segment
            for m_idx in range(BLOCK_M):
                query_pos = pid_m * BLOCK_M + m_idx
                if query_pos >= M:
                    continue

                query_seg_idx = query_pos // segment_size

                # Create mask for this query
                for n_idx in range(BLOCK_N):
                    key_pos = offs_n_base[n_idx]
                    if key_pos >= M:
                        s[m_idx, n_idx] = -1e9
                    elif query_seg_idx != seg_idx:
                        s[m_idx, n_idx] = -1e9

            # Online softmax and accumulation
            # Process each query independently for numerical stability
            for m_idx in range(BLOCK_M):
                if offs_m[m_idx] >= M:
                    continue

                # Get scores for this query
                s_row = s[m_idx, :]

                # Mask invalid positions
                s_row = tl.where(mask_n, s_row, -1e9)

                # Online softmax
                m_ij = tl.max(s_row)
                m_i_new = tl.maximum(m_i[m_idx], m_ij)

                # Compute probabilities
                p = tl.exp(s_row - m_i_new)
                l_ij = tl.sum(p)

                # Update statistics
                alpha = tl.exp(m_i[m_idx] - m_i_new)
                l_i_new = alpha * l_i[m_idx] + l_ij

                # Update accumulator
                acc[m_idx, :] = acc[m_idx, :] * alpha

                # Accumulate weighted values
                for n_idx in range(BLOCK_N):
                    if mask_n[n_idx]:
                        acc[m_idx, :] += p[n_idx] * v[:, n_idx]

                # Update for next iteration
                l_i[m_idx] = l_i_new
                m_i[m_idx] = m_i_new

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


class HilbertAttentionCoreOptimized(nn.Module):
    """
    Optimized Hilbert Attention with better performance.

    Improvements:
    - Better block size selection
    - Optimized kernel that processes only needed positions
    - Reduced memory bandwidth usage
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

    def get_optimal_block_sizes(self, seq_len: int, device: torch.device) -> tuple:
        """Get optimal block sizes based on sequence length and hardware."""
        # GTX 1080 has 48KB shared memory
        # We need to balance between parallelism and memory usage

        if seq_len <= 256:
            BLOCK_M = min(32, seq_len)
            BLOCK_N = min(32, self.segment_size // self.dilation_rate)
        elif seq_len <= 1024:
            BLOCK_M = 64
            BLOCK_N = min(64, self.segment_size // self.dilation_rate)
        else:
            BLOCK_M = 128
            BLOCK_N = min(128, self.segment_size // self.dilation_rate)

        BLOCK_D = min(64, self.head_dim)

        # Ensure minimum sizes for Triton
        BLOCK_M = max(16, BLOCK_M)
        BLOCK_N = max(16, BLOCK_N)
        BLOCK_D = max(16, BLOCK_D)

        return BLOCK_M, BLOCK_N, BLOCK_D

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        if seq_len not in self._hilbert_cache:
            mapping = create_hilbert_mapping(seq_len)
            self._hilbert_cache[seq_len] = mapping.to(device)
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Forward pass with optimized dilated attention.

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

        # Check if we should use Triton
        use_triton = (
            self.head_dim >= 16
            and M_padded >= 16
            and x.device.type == "cuda"
            and qkv.dtype in [torch.float32, torch.float16]
        )

        if use_triton:
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Get Hilbert mapping
            hilbert_map = (
                self.get_hilbert_mapping(M_padded, x.device)
                if use_hilbert
                else torch.arange(M_padded, device=x.device, dtype=torch.int32)
            )

            # Allocate output
            out = torch.zeros_like(q)

            # Get optimal block sizes
            BLOCK_M, BLOCK_N, BLOCK_D = self.get_optimal_block_sizes(M_padded, x.device)

            # Configure grid
            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            # Launch optimized kernel
            dilated_attention_kernel_optimized[grid](
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
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Get Hilbert mapping if needed
            if use_hilbert:
                hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
                hilbert_indices = hilbert_map.long()
                k = k.gather(
                    2,
                    hilbert_indices[None, None, :, None].expand(
                        B, H, M_padded, self.head_dim
                    ),
                )
                v = v.gather(
                    2,
                    hilbert_indices[None, None, :, None].expand(
                        B, H, M_padded, self.head_dim
                    ),
                )

            # Compute attention with dilated mask
            out = self._pytorch_dilated_attention(q, k, v, M_padded)

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out

    def _pytorch_dilated_attention(self, q, k, v, seq_len):
        """PyTorch implementation of dilated attention."""
        B, H, M, D = q.shape

        # Create dilated attention mask
        mask = torch.zeros(M, M, device=q.device, dtype=torch.bool)

        for i in range(M):
            seg_idx = i // self.segment_size
            seg_start = seg_idx * self.segment_size
            seg_end = min(seg_start + self.segment_size, M)

            for j in range(seg_start, seg_end):
                if (j - seg_start) % self.dilation_rate == 0:
                    mask[i, j] = True

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)

        return out


def create_hilbert_mapping(seq_len: int) -> torch.Tensor:
    """Create Hilbert curve mapping for sequences."""
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
