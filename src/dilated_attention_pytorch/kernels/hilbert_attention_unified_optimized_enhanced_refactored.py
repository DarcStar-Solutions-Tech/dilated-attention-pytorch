#!/usr/bin/env python3
"""
Refactored Enhanced Optimized Unified Hilbert Attention.

This version addresses the complexity issues identified in the refactoring analysis:
- Extracted configuration strategies
- Removed dead code
- Simplified softmax paths
- Cleaner parameter handling
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .cache_manager import BoundedCache
from .config_strategies import (
    AttentionConfig,
    AttentionConstants,
    ConfigStrategyFactory,
    OptimizationLevel,
)


@triton.jit
def unified_hilbert_attention_kernel_enhanced_v2(
    # Pointers
    Q,
    K,
    V,
    Out,
    hilbert_map,
    # Strides - Q
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    # Strides - K
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    # Strides - V
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    # Strides - Out
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
    ROWS_PER_BLOCK: tl.constexpr = 1,
):
    """
    Enhanced Hilbert attention kernel - refactored version.

    Key changes:
    - Removed USE_FUSED_SOFTMAX (always use online softmax after fix)
    - Removed ENABLE_PREFETCH (commented code removed)
    - Simplified parameter list
    - Cleaner implementation
    """
    # Constants
    MASK_VALUE: tl.constexpr = -1e9

    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)

    # Multi-row processing support
    if ROWS_PER_BLOCK > 1:
        blocks_per_batch_head = tl.cdiv(num_blocks_m, ROWS_PER_BLOCK)
        pid_m = (pid % blocks_per_batch_head) * ROWS_PER_BLOCK
        pid_bh = pid // blocks_per_batch_head
    else:
        pid_m = pid % num_blocks_m
        pid_bh = pid // num_blocks_m

    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Pre-compute segment boundaries
    block_start_m = pid_m * BLOCK_M
    seg_idx = block_start_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum((seg_idx + 1) * segment_size, M)

    # Query block setup
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
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
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(
        tl.float32
    )
    q = q * scale

    # Initialize accumulator and softmax state
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + MASK_VALUE

    # Process key-value blocks
    if dilation_rate > 1:
        # Sparse attention path
        num_active = (seg_end - seg_start + dilation_rate - 1) // dilation_rate

        for block_idx in range(0, num_active, BLOCK_N):
            # Calculate sparse positions
            active_idx = block_idx + tl.arange(0, BLOCK_N)
            actual_n = seg_start + active_idx * dilation_rate
            mask_n = (active_idx < num_active) & (actual_n < M)

            # Load Hilbert indices
            if hilbert_map is not None:
                h_idx = tl.load(hilbert_map + actual_n, mask=mask_n, other=0)
            else:
                h_idx = actual_n

            # Load K and V
            k_base = K + pid_b * stride_kb + pid_h * stride_kh
            v_base = V + pid_b * stride_vb + pid_h * stride_vh

            k_ptrs = k_base + h_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + h_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )

            # Compute attention scores
            s = tl.dot(q, tl.trans(k))
            s = tl.where(mask_n[None, :], s, MASK_VALUE)

            # Online softmax (always used after normalization fix)
            m_ij = tl.max(s, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.exp(s - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i = alpha * l_i + l_ij

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, v)

            # Update for next iteration
            m_i = m_i_new
    else:
        # Dense attention path
        for start_n in range(seg_start, seg_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seg_end

            # Load Hilbert indices
            if hilbert_map is not None:
                h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
            else:
                h_idx = offs_n

            # Load K and V
            k_base = K + pid_b * stride_kb + pid_h * stride_kh
            v_base = V + pid_b * stride_vb + pid_h * stride_vh

            k_ptrs = k_base + h_idx[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = v_base + h_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd

            k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(
                tl.float32
            )

            # Compute attention scores
            s = tl.dot(q, tl.trans(k))
            s = tl.where(mask_n[None, :], s, MASK_VALUE)

            # Online softmax (unified implementation)
            m_ij = tl.max(s, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.exp(s - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i = alpha * l_i + l_ij

            # Update accumulator
            acc = acc * alpha[:, None] + tl.dot(p, v)

            # Update for next iteration
            m_i = m_i_new

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


class UnifiedHilbertAttentionOptimizedEnhancedRefactored(nn.Module):
    """
    Refactored Enhanced Hilbert Attention.

    Key improvements:
    - Cleaner configuration through strategies
    - Removed dead code
    - Simplified parameters
    - Better separation of concerns
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        cache_size: int = 32,
        cache_memory_mb: float = 100.0,
        hilbert_threshold: int = 1024,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.scale = self.head_dim**-0.5
        self.hilbert_threshold = hilbert_threshold
        self.optimization_level = optimization_level

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

        # Hilbert mapping cache
        self._hilbert_cache = BoundedCache(
            max_size=cache_size,
            max_memory_mb=cache_memory_mb,
            name="UnifiedHilbertEnhancedRefactored_cache",
        )

        # Detect compute capability and create config strategies
        if torch.cuda.is_available():
            self.compute_capability = torch.cuda.get_device_capability()[0]
        else:
            self.compute_capability = 0

        self._triton_available = hasattr(triton, "jit") and torch.cuda.is_available()

        # Create configuration strategies
        self.dense_strategy = ConfigStrategyFactory.create_strategy(
            is_sparse=False,
            compute_capability=self.compute_capability,
            optimization_level=optimization_level,
        )
        self.sparse_strategy = ConfigStrategyFactory.create_strategy(
            is_sparse=True,
            compute_capability=self.compute_capability,
            optimization_level=optimization_level,
        )

    def _get_attention_config(self, seq_len: int) -> AttentionConfig:
        """Get optimal configuration for sequence length."""

        if self.dilation_rate > 1:
            # Sparse configuration
            effective_len = seq_len // self.dilation_rate
            sparsity = 1.0 - (1.0 / self.dilation_rate)
            return self.sparse_strategy.get_config(
                seq_len=seq_len,
                head_dim=self.head_dim,
                effective_len=effective_len,
                sparsity=sparsity,
            )
        else:
            # Dense configuration
            return self.dense_strategy.get_config(
                seq_len=seq_len, head_dim=self.head_dim
            )

    def forward(
        self,
        x: torch.Tensor,
        use_hilbert: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass with refactored implementation."""

        B, M, D = x.shape
        _ = x.device

        # Pad sequence to multiple of segment_size
        M_padded = (
            (M + self.segment_size - 1) // self.segment_size
        ) * self.segment_size
        if M != M_padded:
            x = F.pad(x, (0, 0, 0, M_padded - M))

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Only use Hilbert if sequence length exceeds threshold
        use_hilbert = use_hilbert and M_padded > self.hilbert_threshold

        # Get optimal configuration
        config = self._get_attention_config(M_padded)

        # For very short sequences or causal masking, use PyTorch
        if (
            M_padded <= AttentionConstants.SEQ_PYTORCH_THRESHOLD
            or is_causal
            or not self._triton_available
        ):
            out = self._pytorch_forward(q, k, v, use_hilbert, is_causal)
        else:
            out = self._triton_forward(q, k, v, M_padded, use_hilbert, config)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M_padded, D)

        # Remove padding
        if M != M_padded:
            out = out[:, :M, :]

        out = self.out_proj(out)

        if self.dropout_layer is not None:
            out = self.dropout_layer(out)

        return out

    def _pytorch_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        use_hilbert: bool,
        is_causal: bool,
    ) -> torch.Tensor:
        """PyTorch implementation for short sequences."""

        if use_hilbert:
            _, _, M, _ = q.shape
            hilbert_map = self._get_hilbert_mapping(M, q.device)
            k = k[:, :, hilbert_map]
            v = v[:, :, hilbert_map]

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )

    def _triton_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        M_padded: int,
        use_hilbert: bool,
        config: AttentionConfig,
    ) -> torch.Tensor:
        """Forward pass using Triton kernel."""

        B, H, M, D = q.shape

        # Get Hilbert mapping if needed
        if use_hilbert:
            hilbert_map = self._get_hilbert_mapping(M_padded, q.device)
        else:
            hilbert_map = torch.arange(M_padded, device=q.device, dtype=torch.int32)

        # Allocate output
        out = torch.empty_like(q)

        # Grid configuration
        block_config = config.block_config
        rows_per_block = config.rows_per_block
        num_blocks_m = triton.cdiv(M, block_config.block_m)

        if rows_per_block > 1:
            grid = (triton.cdiv(num_blocks_m, rows_per_block) * B * H,)
        else:
            grid = (num_blocks_m * B * H,)

        # Launch kernel
        unified_hilbert_attention_kernel_enhanced_v2[grid](
            # Pointers
            q,
            k,
            v,
            out,
            hilbert_map,
            # Strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            # Shape
            B,
            H,
            M,
            D,
            # Parameters
            self.scale,
            self.segment_size,
            self.dilation_rate,
            # Meta-parameters
            block_config.block_m,
            block_config.block_n,
            block_config.block_d,
            rows_per_block,
            num_warps=block_config.num_warps,
        )

        return out

    def _get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""

        cache_key = (seq_len, self.segment_size, self.dilation_rate, str(device))
        mapping = self._hilbert_cache.get(cache_key)

        if mapping is None:
            if self.dilation_rate > 1:
                # Segment-local mapping for sparse patterns
                mapping = self._create_segment_local_hilbert_mapping(
                    seq_len, self.segment_size, self.dilation_rate
                ).to(device)
            else:
                # Global mapping for dense patterns
                mapping = self._create_hilbert_mapping(seq_len).to(device)

            self._hilbert_cache.put(cache_key, mapping)

        return mapping

    def _create_segment_local_hilbert_mapping(
        self, seq_len: int, segment_size: int, dilation_rate: int
    ) -> torch.Tensor:
        """Create Hilbert mapping for sparse patterns."""

        mapping = torch.arange(seq_len, dtype=torch.int32)
        num_segments = (seq_len + segment_size - 1) // segment_size

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_end = min(seg_start + segment_size, seq_len)

            # Get sparse positions
            sparse_positions = []
            for i in range(0, seg_end - seg_start, dilation_rate):
                pos = seg_start + i
                if pos < seg_end:
                    sparse_positions.append(pos)

            # Apply Hilbert ordering to sparse positions
            if len(sparse_positions) > 4:
                n_sparse = len(sparse_positions)
                hilbert_perm = self._create_hilbert_mapping(n_sparse)
                hilbert_indices = [hilbert_perm[i].item() for i in range(n_sparse)]

                reordered_sparse = [sparse_positions[hidx] for hidx in hilbert_indices]

                for new_idx, old_pos in enumerate(sparse_positions):
                    mapping[old_pos] = reordered_sparse[new_idx]

        return mapping

    @staticmethod
    def _create_hilbert_mapping(seq_len: int) -> torch.Tensor:
        """Create Hilbert curve mapping."""

        grid_size = 1 << math.ceil(math.log2(math.sqrt(seq_len)) + 0.5)

        def hilbert_index(x: int, y: int, size: int) -> int:
            index = 0
            s = size // 2
            while s > 0:
                rx = (x & s) > 0
                ry = (y & s) > 0
                index += s * s * ((3 * rx) ^ ry)
                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x
                s //= 2
            return index

        positions = []
        for i in range(seq_len):
            x = i % grid_size
            y = i // grid_size
            if y < grid_size:
                h_idx = hilbert_index(x, y, grid_size)
                positions.append((h_idx, i))

        positions.sort(key=lambda p: p[0])

        inverse_mapping = torch.zeros(seq_len, dtype=torch.int32)
        for hilbert_pos, (_, orig_pos) in enumerate(positions):
            inverse_mapping[orig_pos] = hilbert_pos

        return inverse_mapping

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return self._hilbert_cache.get_stats()

    def clear_cache(self):
        """Clear the Hilbert mapping cache."""
        self._hilbert_cache.clear()
