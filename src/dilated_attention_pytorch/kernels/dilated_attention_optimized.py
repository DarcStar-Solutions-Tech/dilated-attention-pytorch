#!/usr/bin/env python3
"""
Optimized Dilated Attention implementation that only processes dilated positions.

Key optimizations:
1. Pre-compute valid position indices
2. Process only dilated positions (not all then mask)
3. Better memory access patterns
4. Reduced computational complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def dilated_attention_kernel_optimized(
    # Pointers
    Q,
    K,
    V,
    Out,
    # Position indices for dilated pattern
    valid_positions,  # Pre-computed valid key positions for each segment
    num_valid_per_seg,  # Number of valid positions per segment
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
    max_valid_per_seg: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VALID: tl.constexpr,  # Process multiple valid positions at once
):
    """Optimized kernel that only processes dilated positions."""
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

    # Process each query position
    for m_idx in range(BLOCK_M):
        query_pos = pid_m * BLOCK_M + m_idx
        valid_query = query_pos < M

        # Determine which segment this query belongs to
        seg_idx = tl.where(valid_query, query_pos // segment_size, 0)

        # Get number of valid positions for this segment
        num_valid = tl.where(valid_query, tl.load(num_valid_per_seg + seg_idx), 0)

        # Process valid positions in blocks
        for valid_start in range(0, max_valid_per_seg, BLOCK_VALID):
            # Check if we should process this block
            process_block = valid_query & (valid_start < num_valid)

            # Load valid position indices
            _ = tl.minimum(valid_start + BLOCK_VALID, num_valid)
            valid_range = tl.arange(0, BLOCK_VALID)
            valid_mask = process_block & ((valid_start + valid_range) < num_valid)

            # Get actual key positions
            pos_offset = seg_idx * max_valid_per_seg + valid_start
            key_positions = tl.load(
                valid_positions + pos_offset + valid_range, mask=valid_mask, other=0
            )

            # Load keys and values at valid positions only
            k_block = tl.zeros([BLOCK_D, BLOCK_VALID], dtype=tl.float32)
            v_block = tl.zeros([BLOCK_D, BLOCK_VALID], dtype=tl.float32)

            for v_idx in range(BLOCK_VALID):
                v_mask = valid_mask[v_idx]
                key_pos = tl.where(v_mask, key_positions[v_idx], 0)

                k_ptr = (
                    K
                    + pid_b * stride_kb
                    + pid_h * stride_kh
                    + key_pos * stride_kn
                    + offs_d * stride_kd
                )
                v_ptr = (
                    V
                    + pid_b * stride_vb
                    + pid_h * stride_vh
                    + key_pos * stride_vn
                    + offs_d * stride_vd
                )

                k_vec = tl.load(k_ptr, mask=v_mask & mask_d, other=0.0)
                v_vec = tl.load(v_ptr, mask=v_mask & mask_d, other=0.0)

                k_block[:, v_idx] = tl.where(v_mask, k_vec, 0.0)
                v_block[:, v_idx] = tl.where(v_mask, v_vec, 0.0)

            # Compute attention scores for this query with valid keys
            q_vec = q[m_idx, :]
            scores = tl.sum(q_vec[:, None] * k_block, axis=0)
            scores = tl.where(valid_mask, scores, -1e9)

            # Online softmax
            m_ij = tl.max(scores)
            m_i_new = tl.where(process_block, tl.maximum(m_i[m_idx], m_ij), m_i[m_idx])
            p = tl.exp(scores - m_i_new)
            l_ij = tl.sum(p)

            # Update statistics
            alpha = tl.exp(m_i[m_idx] - m_i_new)
            l_i[m_idx] = tl.where(process_block, alpha * l_i[m_idx] + l_ij, l_i[m_idx])
            m_i[m_idx] = m_i_new

            # Update accumulator for this query
            acc[m_idx, :] = tl.where(
                process_block, acc[m_idx, :] * alpha, acc[m_idx, :]
            )

            # Accumulate weighted values
            for v_idx in range(BLOCK_VALID):
                v_mask = valid_mask[v_idx] & process_block
                acc[m_idx, :] = tl.where(
                    v_mask, acc[m_idx, :] + p[v_idx] * v_block[:, v_idx], acc[m_idx, :]
                )

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


class DilatedAttentionOptimized(nn.Module):
    """
    Optimized Dilated Attention that only processes dilated positions.

    This implementation pre-computes which positions to attend to and only
    processes those positions, resulting in significant speedup for high
    dilation rates.
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

        # Cache for position indices
        self._position_cache = {}

    def get_valid_positions(self, seq_len: int, device: torch.device):
        """Pre-compute valid positions for dilated attention pattern."""
        cache_key = (seq_len, self.segment_size, self.dilation_rate)

        if cache_key not in self._position_cache:
            num_segments = (seq_len + self.segment_size - 1) // self.segment_size
            max_valid_per_seg = self.segment_size // self.dilation_rate

            # Pre-allocate arrays
            valid_positions = torch.zeros(
                num_segments * max_valid_per_seg, dtype=torch.int32, device=device
            )
            num_valid_per_seg = torch.zeros(
                num_segments, dtype=torch.int32, device=device
            )

            # Fill valid positions for each segment
            for seg_idx in range(num_segments):
                seg_start = seg_idx * self.segment_size
                seg_end = min(seg_start + self.segment_size, seq_len)

                valid_count = 0
                for pos in range(seg_start, seg_end, self.dilation_rate):
                    valid_positions[seg_idx * max_valid_per_seg + valid_count] = pos
                    valid_count += 1

                num_valid_per_seg[seg_idx] = valid_count

            self._position_cache[cache_key] = (
                valid_positions,
                num_valid_per_seg,
                max_valid_per_seg,
            )

        return self._position_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optimized dilated attention.

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

        if use_triton and self.dilation_rate > 1:
            # Use optimized kernel for dilated attention
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Get pre-computed valid positions
            valid_positions, num_valid_per_seg, max_valid_per_seg = (
                self.get_valid_positions(M_padded, x.device)
            )

            # Convert to float32 if needed
            compute_dtype = torch.float32 if q.dtype == torch.float16 else q.dtype
            if q.dtype == torch.float16:
                q = q.to(compute_dtype)
                k = k.to(compute_dtype)
                v = v.to(compute_dtype)

            # Allocate output
            out = torch.zeros_like(q)

            # Configure grid
            BLOCK_M = min(
                32, M_padded
            )  # Smaller blocks for better per-query processing
            BLOCK_D = min(64, self.head_dim)
            BLOCK_VALID = min(16, max_valid_per_seg)

            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            # Launch optimized kernel
            dilated_attention_kernel_optimized[grid](
                q,
                k,
                v,
                out,
                valid_positions,
                num_valid_per_seg,
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
                max_valid_per_seg,
                BLOCK_M,
                BLOCK_D,
                BLOCK_VALID,
            )

            # Convert back if needed
            if qkv.dtype == torch.float16:
                out = out.to(torch.float16)
        else:
            # Fall back to standard implementation for non-dilated or small sequences
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply dilated masking if needed
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


class DilatedAttentionOptimizedV2(nn.Module):
    """
    Alternative optimized implementation using sparse tensor operations.

    This version uses PyTorch's sparse tensor operations for even better
    performance with very high dilation rates.
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
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for sparse masks
        self._sparse_mask_cache = {}

    def get_sparse_mask(self, seq_len: int, device: torch.device):
        """Create sparse attention mask for dilated pattern."""
        cache_key = (seq_len, self.segment_size, self.dilation_rate)

        if cache_key not in self._sparse_mask_cache:
            # Create indices for sparse mask
            indices = []

            for i in range(seq_len):
                seg_idx = i // self.segment_size
                seg_start = seg_idx * self.segment_size
                seg_end = min(seg_start + self.segment_size, seq_len)

                for j in range(seg_start, seg_end, self.dilation_rate):
                    indices.append([i, j])

            if indices:
                indices = torch.tensor(indices, device=device).t()
                values = torch.ones(indices.shape[1], device=device)
                sparse_mask = torch.sparse_coo_tensor(
                    indices, values, (seq_len, seq_len), device=device
                )
            else:
                # Empty mask
                sparse_mask = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), device=device, dtype=torch.long),
                    torch.zeros(0, device=device),
                    (seq_len, seq_len),
                    device=device,
                )

            self._sparse_mask_cache[cache_key] = sparse_mask

        return self._sparse_mask_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using sparse operations."""
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
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.dilation_rate > 1 and M_padded > 1000:  # Use sparse for large sequences
            # Get sparse mask
            sparse_mask = self.get_sparse_mask(M_padded, x.device)

            # Reshape for batch processing
            q_flat = q.reshape(B * H, M_padded, self.head_dim)
            k_flat = k.reshape(B * H, M_padded, self.head_dim)
            v_flat = v.reshape(B * H, M_padded, self.head_dim)

            # Process each batch/head
            out_list = []
            for b in range(B * H):
                # Compute scores using sparse operations
                scores = torch.sparse.mm(
                    sparse_mask, torch.mm(q_flat[b], k_flat[b].t()) * self.scale
                )

                # Apply softmax row-wise
                scores_dense = scores.to_dense()
                scores_dense[scores_dense == 0] = -float("inf")
                attn_weights = F.softmax(scores_dense, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Apply attention
                out_b = torch.mm(attn_weights, v_flat[b])
                out_list.append(out_b)

            out = torch.stack(out_list).reshape(B, H, M_padded, self.head_dim)
        else:
            # Standard implementation for small sequences or no dilation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

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

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        if M_padded > M:
            out = out[:, :M, :]

        out = self.out_proj(out)
        out = self.dropout(out)

        return out
