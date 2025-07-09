"""
Dynamic Dilated Attention with automatic segment size selection.

This module provides dilated attention implementations that automatically
select optimal segment sizes based on runtime conditions.
"""

import warnings
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn

from .base.dilated_attention import DilatedAttention
from .base.improved_dilated_attention import ImprovedDilatedAttention
from .utils.dynamic_segment_selector import (
    DynamicSegmentSelector,
    SegmentSelectionConfig,
)


class DynamicDilatedAttention(nn.Module):
    """
    Dilated Attention with dynamic segment size selection.

    This implementation automatically selects optimal segment lengths based on:
    - Available GPU memory
    - Sequence length
    - Batch size
    - Hardware capabilities

    Args:
        min_segment_size: Minimum segment size (default: 512)
        max_segment_size: Maximum segment size (default: 65536)
        selector_config: Configuration for segment selection
        fallback_segments: Fallback segment lengths if dynamic selection fails
        improved: Whether to use ImprovedDilatedAttention (default: True)
        **kwargs: Additional arguments passed to the base attention class

    Example:
        >>> attention = DynamicDilatedAttention()
        >>> # Segments are selected automatically based on input
        >>> output = attention(query, key, value)
    """

    def __init__(
        self,
        min_segment_size: int = 512,
        max_segment_size: int = 65536,
        selector_config: Optional[SegmentSelectionConfig] = None,
        fallback_segments: Optional[List[int]] = None,
        improved: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Initialize segment selector
        if selector_config is None:
            selector_config = SegmentSelectionConfig(
                min_segment_size=min_segment_size, max_segment_size=max_segment_size
            )
        self.selector = DynamicSegmentSelector(selector_config)

        # Fallback configuration
        self.fallback_segments = fallback_segments or [2048, 4096, 8192]
        self.fallback_dilation_rates = [1, 2, 4]

        # Choose base attention class
        self.attention_class = (
            ImprovedDilatedAttention if improved else DilatedAttention
        )

        # Store additional kwargs for attention initialization
        self.attention_kwargs = kwargs

        # Current attention module (initialized on first forward)
        self._attention: Optional[Union[DilatedAttention, ImprovedDilatedAttention]] = (
            None
        )
        self._current_segments: Optional[List[int]] = None
        self._current_dilation_rates: Optional[List[int]] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        need_weights: bool = False,
        average_attn_weights: bool = True,
        force_segment_update: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with dynamic segment selection.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads
            force_segment_update: Force recomputation of segment sizes

        Returns:
            Output tensor or tuple of (output, attention_weights)
        """
        # Extract dimensions
        batch_size, seq_len, num_heads, head_dim = query.shape

        # Check if we need to update segments
        should_update = (
            self._attention is None
            or force_segment_update
            or self._needs_segment_update(seq_len)
        )

        if should_update:
            try:
                # Select optimal segments
                segment_lengths, dilation_rates = self.selector.select_segments(
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=query.dtype,
                    force_refresh=force_segment_update,
                )

                # Validate segments
                if not self._validate_segments(segment_lengths, seq_len):
                    raise ValueError("Invalid segment configuration")

                # Create new attention module
                self._create_attention_module(segment_lengths, dilation_rates)

            except Exception as e:
                warnings.warn(
                    f"Dynamic segment selection failed: {e}. "
                    f"Using fallback configuration."
                )
                self._create_attention_module(
                    self.fallback_segments, self.fallback_dilation_rates
                )

        # Forward through attention
        # Note: DilatedAttention doesn't support need_weights/average_attn_weights
        output = self._attention(query=query, key=key, value=value, is_causal=is_causal)

        if need_weights:
            # Return dummy weights for compatibility
            batch_size, seq_len, num_heads, _ = query.shape
            dummy_weights = (
                torch.ones(
                    batch_size,
                    num_heads,
                    seq_len,
                    seq_len,
                    device=query.device,
                    dtype=query.dtype,
                )
                / seq_len
            )
            return output, dummy_weights

        return output

    def _needs_segment_update(self, seq_len: int) -> bool:
        """Check if segment configuration needs updating."""
        if self._current_segments is None:
            return True

        # Check if sequence length is compatible with current segments
        max_segment = max(self._current_segments)
        return seq_len % max_segment != 0

    def _validate_segments(self, segment_lengths: List[int], seq_len: int) -> bool:
        """Validate segment configuration."""
        if not segment_lengths:
            return False

        # Check divisibility
        max_segment = max(segment_lengths)
        if seq_len % max_segment != 0:
            # Try to adjust the largest segment
            for divisor in range(max_segment, 0, -1):
                if seq_len % divisor == 0:
                    segment_lengths[-1] = divisor
                    return True
            return False

        return True

    def _create_attention_module(
        self, segment_lengths: List[int], dilation_rates: List[int]
    ):
        """Create or update the attention module."""
        self._current_segments = segment_lengths
        self._current_dilation_rates = dilation_rates

        # Create new attention module
        self._attention = self.attention_class(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            **self.attention_kwargs,
        )

        # Transfer to same device if needed
        if hasattr(self, "_device_tracker"):
            self._attention = self._attention.to(self._device_tracker)

    def get_current_configuration(self) -> Tuple[List[int], List[int]]:
        """Get the current segment configuration."""
        return (self._current_segments or [], self._current_dilation_rates or [])

    def clear_cache(self):
        """Clear segment selection cache."""
        self.selector.clear_cache()

    def _apply(self, fn):
        """Override _apply to track device changes."""
        super()._apply(fn)

        # Track device for lazy initialization
        if hasattr(fn, "__self__") and hasattr(fn.__self__, "device"):
            self._device_tracker = fn.__self__.device

        return self


class DynamicMultiheadDilatedAttention(nn.Module):
    """
    Multihead Dilated Attention with dynamic segment selection.

    This provides a drop-in replacement for nn.MultiheadAttention with
    automatic segment size optimization.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
        selector_config: Configuration for segment selection
        **kwargs: Additional arguments

    Example:
        >>> attention = DynamicMultiheadDilatedAttention(embed_dim=768, num_heads=12)
        >>> # Use like standard nn.MultiheadAttention
        >>> output, _ = attention(query, key, value)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        selector_config: Optional[SegmentSelectionConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Dynamic attention
        self.attention = DynamicDilatedAttention(
            selector_config=selector_config, dropout=dropout, **kwargs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with nn.MultiheadAttention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim] or [seq_len, batch, embed_dim]
            key: Key tensor
            value: Value tensor
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average weights across heads
            is_causal: Whether to apply causal masking

        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle batch_first dimension
        if query.dim() == 3 and query.shape[0] != key.shape[0]:
            # Assume seq_len first, convert to batch first
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            batch_first = False
        else:
            batch_first = True

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply dynamic dilated attention
        if need_weights:
            attn_output, attn_weights = self.attention(
                q,
                k,
                v,
                is_causal=is_causal,
                need_weights=True,
                average_attn_weights=average_attn_weights,
            )
        else:
            attn_output = self.attention(
                q, k, v, is_causal=is_causal, need_weights=False
            )
            attn_weights = None

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        # Convert back if needed
        if not batch_first:
            output = output.transpose(0, 1)

        return output, attn_weights

    def get_current_configuration(self) -> Tuple[List[int], List[int]]:
        """Get the current segment configuration."""
        return self.attention.get_current_configuration()


def create_dynamic_dilated_attention(
    attention_type: str = "improved",
    selector_config: Optional[SegmentSelectionConfig] = None,
    **kwargs,
) -> DynamicDilatedAttention:
    """
    Factory function for creating dynamic dilated attention.

    Args:
        attention_type: Type of attention ("basic" or "improved")
        selector_config: Configuration for segment selection
        **kwargs: Additional arguments

    Returns:
        DynamicDilatedAttention instance
    """
    improved = attention_type.lower() == "improved"

    return DynamicDilatedAttention(
        selector_config=selector_config, improved=improved, **kwargs
    )
