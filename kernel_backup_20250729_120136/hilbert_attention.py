"""
Simplified Hilbert Attention implementation.

This module provides a clean, efficient implementation of Hilbert-ordered attention
with support for dilated/sparse patterns. All optimizations are applied automatically
based on the input parameters and available hardware.
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_manager import BoundedCache


class HilbertAttention(nn.Module):
    """
    Hilbert-ordered attention with automatic optimization.

    This implementation automatically selects the best computation strategy
    based on sequence length, dilation rate, and available hardware.

    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        segment_size: Size of each attention segment (default: 128)
        dilation_rate: Dilation rate for sparse attention (default: 1)
        dropout: Dropout probability (default: 0.0)
        cache_size: Maximum cached Hilbert mappings (default: 32)
        cache_memory_mb: Maximum cache memory in MB (default: 100.0)

    Example:
        # Standard attention
        attn = HilbertAttention(hidden_dim=768, num_heads=12)

        # Dilated attention
        attn = HilbertAttention(
            hidden_dim=768,
            num_heads=12,
            dilation_rate=4
        )
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
            name="HilbertAttention_cache",
        )

        # Try to import Triton kernels
        self._triton_available = False
        try:
            from .hilbert_attention_core import HilbertAttentionFunction

            self._triton_forward = HilbertAttentionFunction.apply
            self._triton_available = True
        except (ImportError, RuntimeError):
            pass

        # Try to import fused kernels
        self._fused_kernels_available = False
        self._fused_forward_fn = None
        try:
            from .hilbert_attention_fused_v2 import launch_fused_kernel

            self._fused_kernels_available = True
            self._fused_forward_fn = launch_fused_kernel
        except (ImportError, RuntimeError):
            pass

    def forward(
        self,
        x: torch.Tensor,
        use_hilbert: bool = True,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with automatic optimization selection.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            use_hilbert: Whether to use Hilbert curve reordering
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
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

        # Check if we should use fused kernels for medium sequences
        # Based on benchmarks, fused kernels are optimal for 2K-16K sequences
        # But avoid dimension mismatch issues on Pascal GPUs
        compute_capability = (
            torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
        )
        use_fused_kernel = (
            self._triton_available
            and device.type == "cuda"
            and 2048 <= M_padded <= 16384  # Extended range for better performance
            and hasattr(self, "_fused_kernels_available")
            and self._fused_kernels_available
            # Avoid fused kernels on Pascal when BLOCK_D < head_dim
            and not (compute_capability < 7 and self.head_dim > 32)
        )

        # Select computation method
        if use_fused_kernel:
            # Use fused kernel for better performance at medium sequence lengths
            out = self._fused_forward(qkv, M_padded, M, B, use_hilbert, is_causal)
        elif (
            self._triton_available
            and device.type == "cuda"
            and use_hilbert
            and not is_causal  # Triton kernel doesn't support causal masking yet
        ):
            # Triton kernel handles Hilbert reordering internally
            out = self._triton_forward_wrapper(q, k, v, M_padded, M, B, use_hilbert)
        else:
            # PyTorch implementation
            if use_hilbert:
                # Apply Hilbert reordering to k and v for all attention types
                hilbert_map = self._get_hilbert_mapping(M_padded, device)
                k = k[:, :, hilbert_map]
                v = v[:, :, hilbert_map]

            if self.dilation_rate > 1:
                # Sparse attention for dilated patterns
                out = self._sparse_attention(q, k, v, is_causal)
            else:
                # Standard attention
                out = self._standard_attention(q, k, v, is_causal)

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

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        # Use PyTorch's optimized implementation when available
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=self.scale,
            )

        # Fallback implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            mask = torch.tril(torch.ones(scores.shape[-2:], device=scores.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        return torch.matmul(attn_weights, v)

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Sparse attention for dilated patterns."""
        B, H, N, D = q.shape
        out = torch.zeros_like(q)

        # Process each segment
        num_segments = (N + self.segment_size - 1) // self.segment_size

        for seg_idx in range(num_segments):
            seg_start = seg_idx * self.segment_size
            seg_end = min(seg_start + self.segment_size, N)
            seg_len = seg_end - seg_start

            # Get queries for this segment
            q_seg = q[:, :, seg_start:seg_end, :]

            # Calculate sparse positions
            num_sparse = (seg_len + self.dilation_rate - 1) // self.dilation_rate
            sparse_indices = torch.arange(
                seg_start,
                min(seg_start + num_sparse * self.dilation_rate, N),
                self.dilation_rate,
                device=q.device,
            )

            # Get keys and values at sparse positions
            # Note: k and v are already Hilbert-reordered if use_hilbert was True
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

    def _fused_forward(
        self,
        qkv: torch.Tensor,
        M_padded: int,
        M_orig: int,
        B: int,
        use_hilbert: bool,
        is_causal: bool,
    ) -> torch.Tensor:
        """Forward pass using fused kernels for better performance."""
        # Extract Q, K, V from stacked tensor
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get Hilbert mapping if needed
        if use_hilbert:
            hilbert_map = self._get_hilbert_mapping(M_padded, qkv.device)
        else:
            hilbert_map = None

        # For now, fused kernel doesn't support sparse attention
        if self.dilation_rate > 1:
            # Fall back to regular sparse attention
            if use_hilbert and hilbert_map is not None:
                k = k[:, :, hilbert_map]
                v = v[:, :, hilbert_map]
            return self._sparse_attention(q, k, v, is_causal)

        # Call fused kernel
        return self._fused_forward_fn(
            q,
            k,
            v,
            self.scale,
            hilbert_map,
        )

    def _triton_forward_wrapper(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        M_padded: int,
        M_orig: int,
        B: int,
        use_hilbert: bool,
    ) -> torch.Tensor:
        """Wrapper for Triton kernel forward pass."""
        # Get Hilbert mapping if needed
        if use_hilbert:
            hilbert_map = self._get_hilbert_mapping(M_padded, q.device)
        else:
            # Identity mapping for standard attention
            hilbert_map = torch.arange(M_padded, device=q.device, dtype=torch.int32)

        # Stack QKV for Triton kernel
        qkv = torch.stack([q, k, v], dim=0)

        # Call Triton kernel
        return self._triton_forward(
            qkv,
            self.scale,
            hilbert_map,
            self.segment_size,
            self.dilation_rate,
            M_padded,
            M_orig,
            B,
            self.num_heads,
            self.head_dim,
        )

    def _get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        # Create cache key that includes dilation rate for sparse patterns
        cache_key = (seq_len, self.segment_size, self.dilation_rate)
        mapping = self._hilbert_cache.get(cache_key)

        if mapping is None or mapping.device != device:
            if seq_len <= self.hilbert_threshold:
                # Use identity mapping for sequences below threshold
                mapping = torch.arange(seq_len, dtype=torch.int32).to(device)
            elif self.dilation_rate > 1:
                # For sparse patterns, create segment-local Hilbert mapping
                mapping = self._create_segment_local_hilbert_mapping(
                    seq_len, self.segment_size, self.dilation_rate
                ).to(device)
            else:
                # For dense patterns, use global Hilbert mapping
                mapping = self._create_hilbert_mapping(seq_len).to(device)
            self._hilbert_cache.put(cache_key, mapping)

        return mapping

    def _create_segment_local_hilbert_mapping(
        self, seq_len: int, segment_size: int, dilation_rate: int
    ) -> torch.Tensor:
        """
        Create Hilbert mapping optimized for sparse patterns.

        For sparse patterns, we apply Hilbert ordering within each segment's
        sparse positions to maintain locality while preserving the sparse access pattern.
        """
        # Start with identity mapping
        mapping = torch.arange(seq_len, dtype=torch.int32)

        # Process each segment
        num_segments = (seq_len + segment_size - 1) // segment_size

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_size
            seg_end = min(seg_start + segment_size, seq_len)
            seg_len = seg_end - seg_start

            # Get sparse positions in this segment
            sparse_positions = []
            for i in range(0, seg_len, dilation_rate):
                pos = seg_start + i
                if pos < seg_end:
                    sparse_positions.append(pos)

            # Apply Hilbert ordering to sparse positions within segment
            if (
                len(sparse_positions) > 4
            ):  # Only apply Hilbert if we have enough positions
                # Create Hilbert curve for the sparse positions
                n_sparse = len(sparse_positions)

                # For small numbers of positions, use simple optimized patterns
                if n_sparse <= 16:
                    # Simple 2D snake pattern for small sets
                    grid_size = int(math.ceil(math.sqrt(n_sparse)))
                    hilbert_indices = []

                    for row in range(grid_size):
                        row_indices = []
                        for col in range(grid_size):
                            idx = row * grid_size + col
                            if idx < n_sparse:
                                row_indices.append(idx)

                        # Reverse even rows for snake pattern
                        if row % 2 == 1:
                            row_indices.reverse()
                        hilbert_indices.extend(row_indices)
                else:
                    # Use actual Hilbert curve for larger sets
                    hilbert_perm = self._create_hilbert_mapping(n_sparse)
                    hilbert_indices = [hilbert_perm[i].item() for i in range(n_sparse)]

                # Apply the reordering to maintain sparse structure
                # Map sparse position index to actual position
                reordered_sparse = [
                    sparse_positions[hidx] for hidx in hilbert_indices[:n_sparse]
                ]

                # Update the mapping to preserve sparse pattern
                for new_idx, old_pos in enumerate(sparse_positions):
                    mapping[old_pos] = reordered_sparse[new_idx]

        return mapping

    @staticmethod
    def _create_hilbert_mapping(seq_len: int) -> torch.Tensor:
        """Create Hilbert curve mapping for sequence."""
        # Note: threshold check is done in _get_hilbert_mapping
        # This method assumes we want actual Hilbert mapping

        # Find grid size
        grid_size = 1 << math.ceil(math.log2(math.sqrt(seq_len)) + 0.5)

        # Generate 2D Hilbert curve
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

        # Create mapping
        positions = []
        for i in range(seq_len):
            x = i % grid_size
            y = i // grid_size
            if y < grid_size:
                h_idx = hilbert_index(x, y, grid_size)
                positions.append((h_idx, i))

        # Sort by Hilbert index
        positions.sort(key=lambda p: p[0])

        # Create inverse mapping: original_pos -> hilbert_pos
        inverse_mapping = torch.zeros(seq_len, dtype=torch.int32)
        for hilbert_pos, (_, orig_pos) in enumerate(positions):
            inverse_mapping[orig_pos] = hilbert_pos

        return inverse_mapping

    def clear_cache(self) -> None:
        """Clear the Hilbert mapping cache."""
        self._hilbert_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self._hilbert_cache.get_stats()
