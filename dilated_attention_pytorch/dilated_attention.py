"""
Dilated Attention implementation using the refactored core architecture.

This module provides the standard dilated attention mechanism from the LongNet paper.
"""

from collections.abc import Sequence
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    xops = None

from .core import (
    BaseDilatedAttention,
    DilatedAttentionConfig,
    optimize_attention_computation,
)

# Try to import enhanced memory pool
try:
    from .core.enhanced_memory_pool import get_enhanced_memory_pool

    HAS_ENHANCED_MEMORY_POOL = True
except ImportError:
    HAS_ENHANCED_MEMORY_POOL = False
    import warnings

    warnings.warn("Enhanced memory pool not available")


class DilatedAttention(BaseDilatedAttention):
    """
    Implement dilated, scaled dot product attention with softmax.

    This implementation follows the LongNet paper, supporting variable segment
    lengths and dilation rates for efficient long-sequence attention.

    Args:
        segment_lengths: List of segment lengths for each attention group
        dilation_rates: List of dilation rates corresponding to each segment
        softmax_scale: Temperature for softmax (default: 1/sqrt(d))
        attention_dropout: Dropout rate for attention (default: 0.0)
        op: Optional xFormers attention operation
        enable_memory_pool: Enable enhanced memory pool for tensor allocation (default: True)
        enable_profiling: Enable memory profiling when using memory pool (default: False)
        lightweight_pool: Use lightweight pool config for better performance (default: True)

    Example:
        >>> attention = DilatedAttention(
        ...     segment_lengths=[2048, 4096, 8192],
        ...     dilation_rates=[1, 2, 4],
        ...     attention_dropout=0.1,
        ...     enable_memory_pool=True,
        ...     lightweight_pool=True
        ... )
    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        softmax_scale: float | None = None,
        attention_dropout: float = 0.0,
        op: Any | None = None,  # xops.AttentionOp when available
        enable_memory_pool: bool = True,
        enable_profiling: bool = False,
        lightweight_pool: bool = True,
        **kwargs,
    ):
        # Create configuration
        config = DilatedAttentionConfig(
            segment_lengths=list(segment_lengths),
            dilation_rates=list(dilation_rates),
            dropout=attention_dropout,
            **kwargs,
        )

        # Initialize base class
        super().__init__(config)

        # Store additional parameters
        self.softmax_scale = softmax_scale
        self.op = op

        # Enhanced memory pool integration
        self.enable_memory_pool = enable_memory_pool and HAS_ENHANCED_MEMORY_POOL
        self.lightweight_pool = lightweight_pool
        self._memory_pool = None
        if self.enable_memory_pool:
            if lightweight_pool:
                # Use lightweight pool for better performance based on lessons learned
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=False,  # Disable for speed
                    enable_bucketed=True,  # Keep for common sizes
                    enable_numa=False,  # Disable for speed
                    enable_profiling=enable_profiling,
                )
            else:
                # Full memory pool with all features
                self._memory_pool = get_enhanced_memory_pool(
                    enable_fragment_aware=True,
                    enable_bucketed=True,
                    enable_numa=True,
                    enable_profiling=enable_profiling,
                )

    def _allocate_tensor(self, shape, dtype, device, strategy="auto", zero_init=True):
        """
        Allocate tensor using enhanced memory pool if enabled.

        Args:
            shape: Tensor shape tuple
            dtype: Tensor data type
            device: Target device
            strategy: Allocation strategy for memory pool
            zero_init: Whether to zero-initialize the tensor

        Returns:
            Allocated tensor (optionally zero-initialized)
        """
        if self._memory_pool is not None:
            # Use enhanced memory pool with strategy selection
            tensor = self._memory_pool.allocate(shape, dtype, device, strategy)
            # Zero-initialize only if requested (avoid cost for temp tensors)
            if zero_init:
                tensor.zero_()
            return tensor
        else:
            # Fallback to direct allocation
            if zero_init:
                return torch.zeros(shape, dtype=dtype, device=device)
            else:
                return torch.empty(shape, dtype=dtype, device=device)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for dilated attention.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            attention_mask: Optional attention mask (not supported with xFormers)

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]

        Note:
            Input shape convention: (batch, seq_len, num_heads, head_dim)
            This matches the LongNet paper's notation.
        """
        # Validate inputs using base class method
        self._validate_forward_inputs(query, key, value, attention_mask)

        # Extract dimensions
        b, n, h, d = query.shape

        # Get head groups from base class cache
        group_sizes, head_ranges = self._get_head_groups(h)

        # Initialize output tensor using memory pool
        # Use enhanced memory pool for main output tensor
        device, dtype = query.device, query.dtype
        out = self._allocate_tensor(query.shape, dtype, device, strategy="auto")

        # Process each attention group
        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths, strict=False)
        ):
            if g == 0:  # Skip empty groups
                continue

            # Split sequences into segments
            q_seg = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k_seg = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v_seg = rearrange(value, "b (n s) h d -> b n s h d", s=s)

            # Apply dilation with offset
            offset = i % r
            hmin, hmax = head_ranges[i]

            q_dil = q_seg[:, :, offset::r, hmin:hmax, :]
            k_dil = k_seg[:, :, offset::r, hmin:hmax, :]
            v_dil = v_seg[:, :, offset::r, hmin:hmax, :]

            # Fold segments into batch dimension
            q_batch = rearrange(q_dil, "b n s h d -> (b n) s h d")
            k_batch = rearrange(k_dil, "b n s h d -> (b n) s h d")
            v_batch = rearrange(v_dil, "b n s h d -> (b n) s h d")

            # Apply attention
            if HAS_XFORMERS and self.op is not None:
                # Use xFormers memory efficient attention
                attn_bias = xops.LowerTriangularMask() if is_causal else None
                x = xops.memory_efficient_attention(
                    query=q_batch,
                    key=k_batch,
                    value=v_batch,
                    op=self.op,
                    attn_bias=attn_bias,
                    p=self.dropout if self.training else 0.0,
                    scale=self.softmax_scale,
                )
            else:
                # Use optimized attention from core utilities
                # optimize_attention_computation expects [..., seq_len, num_heads, head_dim]
                # We have [(b n), s, h, d] which matches the expected format
                x = optimize_attention_computation(
                    q_batch,
                    k_batch,
                    v_batch,
                    is_causal=is_causal,
                    attention_mask=None,  # Mask not supported in segments
                    dropout_p=self.dropout if self.training else 0.0,
                )

            # Unfold segments from batch dimension
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)

            # Gather outputs with proper indexing
            # Thread-safe accumulation: avoid in-place operations on shared tensors
            with self._cache_lock:
                out_seg = rearrange(out, "b (n s) h d -> b n s h d", s=s)
                out_seg[:, :, offset::r, hmin:hmax, :] = (
                    out_seg[:, :, offset::r, hmin:hmax, :] + x
                )
                out = rearrange(out_seg, "b n s h d -> b (n s) h d", s=s)

        # Normalize by number of groups (Eq. 10 from paper)
        # NOTE: Normalization must happen before dropout for mathematical correctness
        out = out / self.num_groups

        # Apply dropout if configured (after normalization)
        out = self._apply_dropout(out)

        return out

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        repr_str = super().extra_repr()
        if self.softmax_scale is not None:
            repr_str += f", softmax_scale={self.softmax_scale}"
        if self.op is not None:
            repr_str += f", op={self.op.__class__.__name__}"
        return repr_str


# Backward compatibility function
def create_dilated_attention(
    segment_lengths: Sequence[int], dilation_rates: Sequence[int], **kwargs
) -> DilatedAttention:
    """
    Create a dilated attention module (backward compatibility).

    Args:
        segment_lengths: List of segment lengths
        dilation_rates: List of dilation rates
        **kwargs: Additional arguments passed to DilatedAttention

    Returns:
        DilatedAttention module
    """
    return DilatedAttention(segment_lengths, dilation_rates, **kwargs)
