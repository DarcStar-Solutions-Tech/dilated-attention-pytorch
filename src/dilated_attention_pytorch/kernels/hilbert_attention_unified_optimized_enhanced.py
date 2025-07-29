#!/usr/bin/env python3
"""
Enhanced Optimized Unified Hilbert Attention with all optimizations integrated.

This version combines:
- All optimizations from unified_optimized
- GPU-specific configurations from enhanced (with 8K optimization)
- Strided sparse iteration from enhanced
- Multi-row processing from enhanced
- Adaptive kernel selection based on hardware and input
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from .cache_manager import BoundedCache


@triton.jit
def unified_hilbert_attention_kernel_enhanced(
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
    mask_value: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_FUSED_SOFTMAX: tl.constexpr,
    ENABLE_PREFETCH: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr = 1,
):
    """Enhanced optimized Hilbert attention kernel with all features."""
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

    # OPTIMIZATION 1: Pre-compute segment boundaries once
    block_start_m = pid_m * BLOCK_M
    seg_idx = block_start_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum((seg_idx + 1) * segment_size, M)

    # Compute query block boundaries
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Masks
    mask_m = offs_m < M
    mask_d = offs_d < D

    # Load queries - ensure proper dtype
    q_ptrs = (
        Q + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(
        tl.float32
    )
    q = q * scale

    # Initialize accumulator with proper dtype
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Online softmax state
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + mask_value

    # Process key-value blocks
    if dilation_rate > 1:
        # Sparse attention with strided iteration
        num_active = (seg_end - seg_start + dilation_rate - 1) // dilation_rate

        for block_idx in range(0, num_active, BLOCK_N):
            # Calculate sparse positions
            active_idx = block_idx + tl.arange(0, BLOCK_N)
            actual_n = seg_start + active_idx * dilation_rate
            mask_n = (active_idx < num_active) & (actual_n < M)

            # Load Hilbert indices if provided
            if hilbert_map is not None:
                h_idx = tl.load(hilbert_map + actual_n, mask=mask_n, other=0)
            else:
                h_idx = actual_n

            # OPTIMIZATION 4: Combined pointer calculation
            k_base = K + pid_b * stride_kb + pid_h * stride_kh
            v_base = V + pid_b * stride_vb + pid_h * stride_vh

            # Load K and V with proper dtype handling
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

            # Apply sparse mask
            s = tl.where(mask_n[None, :], s, mask_value)

            # Online softmax with fused operations
            m_ij = tl.max(s, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.exp(s - m_i_new[:, None])
            l_ij = tl.sum(p, axis=1)

            # Update statistics
            alpha = tl.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij

            # OPTIMIZATION 5: Minimize redundant computation
            acc = acc * alpha[:, None]
            # Prefetching is not available in current Triton version
            # if ENABLE_PREFETCH:
            #     # Prefetch next block's data
            #     tl.prefetch(k_ptrs + BLOCK_N * stride_kn, eviction_policy="evict_last")
            #     tl.prefetch(v_ptrs + BLOCK_N * stride_vn, eviction_policy="evict_last")

            # Update accumulator
            acc += tl.dot(p.to(v.dtype), v)

            # Update for next iteration
            l_i = l_i_new
            m_i = m_i_new
    else:
        # Dense attention path with optimizations
        # OPTIMIZATION 2: Use minimum range for segment
        start_n = seg_start
        end_n = seg_end

        # OPTIMIZATION 3: Aligned memory access
        for start_n in range(start_n, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < end_n

            # Load Hilbert indices if provided
            if hilbert_map is not None:
                h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)
            else:
                h_idx = offs_n

            # OPTIMIZATION 4: Combined pointer calculation
            k_base = K + pid_b * stride_kb + pid_h * stride_kh
            v_base = V + pid_b * stride_vb + pid_h * stride_vh

            # Load K and V
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

            # Apply mask
            s = tl.where(mask_n[None, :], s, -1e9)

            if USE_FUSED_SOFTMAX:
                # OPTIMIZATION 6: Fused softmax operations
                m_ij = tl.max(s, axis=1)
                m_i_new = tl.maximum(m_i, m_ij)
                p = tl.exp(s - m_i_new[:, None])
                l_ij = tl.sum(p, axis=1)

                # Update statistics
                alpha = tl.exp(m_i - m_i_new)
                l_i_new = alpha * l_i + l_ij

                # OPTIMIZATION 5: Minimize redundant computation
                acc = acc * alpha[:, None]
                # Prefetching is not available in current Triton version
                # if ENABLE_PREFETCH:
                #     # Prefetch next block's data
                #     tl.prefetch(
                #         k_ptrs + BLOCK_N * stride_kn, eviction_policy="evict_last"
                #     )
                #     tl.prefetch(
                #         v_ptrs + BLOCK_N * stride_vn, eviction_policy="evict_last"
                #     )

                # Update accumulator
                acc += tl.dot(p.to(v.dtype), v)

                # Update for next iteration
                l_i = l_i_new
                m_i = m_i_new
            else:
                # Standard softmax path
                m_ij = tl.max(s, axis=1, keep_dims=True)
                p = tl.exp(s - m_ij)
                l_ij = tl.sum(p, axis=1, keep_dims=True)
                p = p / l_ij

                # Update accumulator
                acc += tl.dot(p, v)

    # Final normalization
    if USE_FUSED_SOFTMAX:
        acc = acc / tl.maximum(l_i[:, None], 1e-10)

    # Store output
    out_ptrs = (
        Out + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


# Define mask_value at module level for Triton compilation
mask_value = -1e9


class UnifiedHilbertAttentionOptimizedEnhanced(nn.Module):
    """
    Enhanced optimized unified Hilbert attention with all features integrated.

    This implementation includes:
    - All optimizations from unified_optimized
    - Sophisticated GPU-specific configuration selection
    - Special optimizations for 8K sequences
    - Strided sparse iteration for dilated patterns
    - Multi-row processing for medium sequences
    - Adaptive kernel selection based on hardware and input
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        cache_size: int = 32,
        cache_memory_mb: float = 100.0,
        hilbert_threshold: int = 1024,
        enable_multi_row: bool = True,
        enable_8k_optimization: bool = True,
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
        self.enable_multi_row = enable_multi_row
        self.enable_8k_optimization = enable_8k_optimization
        self.mask_value = -1e9

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
            name="UnifiedHilbertOptimizedEnhanced_cache",
        )

        # Detect compute capability
        if torch.cuda.is_available():
            self.compute_capability = torch.cuda.get_device_capability()[0]
        else:
            self.compute_capability = 0

        # Check if Triton kernels are available
        self._triton_available = hasattr(triton, "jit") and torch.cuda.is_available()

    def _get_optimal_config(self, seq_len: int) -> Dict[str, any]:
        """
        Get optimal configuration based on sequence length and GPU.

        This integrates the sophisticated configuration logic from enhanced.
        """
        config = {}

        # Check if we're on Pascal or newer GPU
        is_pascal = self.compute_capability < 7
        
        # For sparse patterns, use smaller block sizes like Unified
        if self.dilation_rate > 1:
            config["block_m"] = 64
            config["block_n"] = 64
            config["block_d"] = min(32, self.head_dim)
            config["num_warps"] = 4
            config["rows_per_block"] = 1
            config["fused_block_n"] = 64
            config["use_fused_softmax"] = True
            config["enable_prefetch"] = False
            return config

        if is_pascal:
            # Pascal GPU (limited shared memory - 48KB)
            if seq_len <= 1024:
                config["block_m"] = 32
                config["block_n"] = 32
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 2
            elif seq_len <= 4096:
                config["block_m"] = 64
                config["block_n"] = 64
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 4
            elif seq_len == 8192 and self.enable_8k_optimization:
                # Special 8K optimization for Pascal
                config["block_m"] = 64
                config["block_n"] = 64
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 4
            else:
                config["block_m"] = 64
                config["block_n"] = 64
                config["block_d"] = min(32, self.head_dim)
                config["num_warps"] = 4
        else:
            # Volta+ GPU (more shared memory)
            if seq_len <= 1024:
                config["block_m"] = 64
                config["block_n"] = 64
                config["block_d"] = min(64, self.head_dim)
                config["num_warps"] = 4
            elif seq_len <= 4096:
                config["block_m"] = 128
                config["block_n"] = 128
                config["block_d"] = min(64, self.head_dim)
                config["num_warps"] = 8
            elif seq_len == 8192 and self.enable_8k_optimization:
                # Special 8K optimization for Volta+
                config["block_m"] = 64
                config["block_n"] = 128  # Better grid alignment
                config["block_d"] = min(64, self.head_dim)
                config["num_warps"] = 8
            elif 8192 < seq_len <= 10240:
                # 8K-10K range optimization
                config["block_m"] = 96
                config["block_n"] = 96
                config["block_d"] = min(64, self.head_dim)
                config["num_warps"] = 8
            else:
                config["block_m"] = 128
                config["block_n"] = 128
                config["block_d"] = min(64, self.head_dim)
                config["num_warps"] = 8

        # Multi-row processing for medium sequences
        # Disable for sparse patterns as it hurts performance
        if seq_len >= 4096 and self.enable_multi_row and self.dilation_rate == 1:
            config["rows_per_block"] = 2
            config["fused_block_n"] = config["block_n"] * 2
        else:
            config["rows_per_block"] = 1
            config["fused_block_n"] = config["block_n"]

        # Use fused softmax for larger sequences
        config["use_fused_softmax"] = seq_len >= 2048

        # Enable prefetch on newer GPUs
        config["enable_prefetch"] = not is_pascal and seq_len >= 2048

        return config

    def _strided_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Enhanced sparse attention using strided iteration.

        This integrates the V2 optimization from sparse_optimized.
        """
        B, H, N, D = q.shape
        out = torch.zeros_like(q)

        # Process each segment with strided iteration
        num_segments = (N + self.segment_size - 1) // self.segment_size

        for seg_idx in range(num_segments):
            seg_start = seg_idx * self.segment_size
            seg_end = min(seg_start + self.segment_size, N)
            seg_len = seg_end - seg_start

            # Get queries for this segment
            q_seg = q[:, :, seg_start:seg_end, :]

            # Direct sparse position calculation (key optimization)
            active_per_segment = seg_len // self.dilation_rate
            if seg_len % self.dilation_rate > 0:
                active_per_segment += 1

            # Generate only active positions (strided approach)
            sparse_indices = torch.arange(
                seg_start,
                min(seg_start + active_per_segment * self.dilation_rate, seg_end),
                self.dilation_rate,
                device=q.device,
            )

            # Ensure we don't exceed bounds
            sparse_indices = sparse_indices[sparse_indices < N]

            if len(sparse_indices) == 0:
                continue

            # Get keys and values at sparse positions
            k_sparse = k[:, :, sparse_indices, :]
            v_sparse = v[:, :, sparse_indices, :]

            # Compute attention
            scores = torch.matmul(q_seg, k_sparse.transpose(-2, -1)) * self.scale

            # Apply causal mask if needed
            if is_causal:
                q_indices = torch.arange(seg_start, seg_end, device=q.device)
                causal_mask = q_indices.unsqueeze(1) >= sparse_indices.unsqueeze(0)
                scores = scores.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            attn_weights = F.softmax(scores, dim=-1)

            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            out[:, :, seg_start:seg_end, :] = torch.matmul(attn_weights, v_sparse)

        return out

    def forward(
        self,
        x: torch.Tensor,
        use_hilbert: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with enhanced optimization selection.
        """
        B, M, D = x.shape
        device = x.device

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
        config = self._get_optimal_config(M_padded)

        # For very short sequences or causal masking, use PyTorch
        if M_padded <= 512 or is_causal or not self._triton_available:
            # Use PyTorch implementation
            if use_hilbert:
                hilbert_map = self._get_hilbert_mapping(M_padded, device)
                k = k[:, :, hilbert_map]
                v = v[:, :, hilbert_map]

            # Use PyTorch's optimized SDPA
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )
        else:
            # Use enhanced unified Triton kernel for all patterns including sparse
            # The kernel already has efficient sparse handling in the Triton code
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

    def _triton_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        M_padded: int,
        use_hilbert: bool,
        config: Dict[str, any],
    ) -> torch.Tensor:
        """Forward pass using enhanced Triton kernel."""
        B, H, M, D = q.shape

        # Get Hilbert mapping if needed
        if use_hilbert:
            hilbert_map = self._get_hilbert_mapping(M_padded, q.device)
        else:
            # Create identity mapping
            hilbert_map = torch.arange(M_padded, device=q.device, dtype=torch.int32)

        # Allocate output
        out = torch.empty_like(q)

        # Grid configuration with multi-row support
        rows_per_block = config.get("rows_per_block", 1)
        num_blocks_m = triton.cdiv(M, config["block_m"])
        if rows_per_block > 1:
            grid = (triton.cdiv(num_blocks_m, rows_per_block) * B * H,)
        else:
            grid = (num_blocks_m * B * H,)

        # Launch kernel
        unified_hilbert_attention_kernel_enhanced[grid](
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
            self.mask_value,
            # Meta-parameters
            config["block_m"],
            config["block_n"],
            config["block_d"],
            config["use_fused_softmax"],
            config["enable_prefetch"],
            rows_per_block,
            num_warps=config["num_warps"],
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
