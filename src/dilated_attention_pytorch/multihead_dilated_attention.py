"""
Multihead Dilated Attention implementation using the refactored core architecture.

This module provides a drop-in replacement for nn.MultiheadAttention with
dilated attention and MAGNETO-style layer normalization.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .core import (
    BaseMultiheadDilatedAttention,
    DilatedAttentionConfig,
    MultiheadConfig,
    merge_attention_heads,
    split_attention_heads,
)
from .dilated_attention import DilatedAttention

# Handle xformers availability
try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

    # Create dummy xops for type hints
    class xops:
        class AttentionOp:
            pass


class MultiheadDilatedAttention(BaseMultiheadDilatedAttention):
    """
    Multi-head dilated attention with MAGNETO-style initialization.

    This module provides a drop-in replacement for torch.nn.MultiheadAttention
    with dilated attention patterns and optional layer normalization following
    the MAGNETO architecture.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dilation_rates: List of dilation rates for each attention group
        segment_lengths: List of segment lengths for each attention group
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        layer_norm: Whether to apply layer norm before output (default: True)
        layer_norm_eps: Epsilon for layer normalization (default: 1e-5)
        gamma_init: MAGNETO initialization scale (default: 1.0)
        device: Device to place parameters on
        dtype: Data type for parameters
        op: Optional xFormers attention operation

    Example:
        >>> attention = MultiheadDilatedAttention(
        ...     embed_dim=768,
        ...     num_heads=12,
        ...     dilation_rates=[1, 2, 4],
        ...     segment_lengths=[2048, 4096, 8192]
        ... )
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        op: xops.AttentionOp | None = None,
    ):
        # Create configurations
        multihead_config = MultiheadConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )

        attention_config = DilatedAttentionConfig(
            segment_lengths=list(segment_lengths),
            dilation_rates=list(dilation_rates),
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        # Initialize base class
        super().__init__(multihead_config, attention_config)

        # Store xFormers operation
        self.op = op

        # Create attention module with xFormers support
        self.attention = self._create_attention_module()

    def _create_attention_module(self) -> DilatedAttention:
        """Create the underlying dilated attention module."""
        return DilatedAttention(
            segment_lengths=self.attention_config.segment_lengths,
            dilation_rates=self.attention_config.dilation_rates,
            attention_dropout=self.attention_config.dropout,
            op=None,  # xformers op, optional
            device=self.device,
            dtype=self.dtype,
        )

    def _init_qkv_projections(self, factory_kwargs):
        """Initialize QKV projections."""
        # Create separate projections (not fused)
        self.q_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=self.bias, **factory_kwargs
        )
        self.k_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=self.bias, **factory_kwargs
        )
        self.v_proj = nn.Linear(
            self.embed_dim, self.embed_dim, bias=self.bias, **factory_kwargs
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """
        Forward pass for multihead dilated attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (uses query if None)
            value: Value tensor (uses query if None)
            key_padding_mask: Mask for padded positions [batch, seq_len]
            need_weights: Whether to return attention weights (not supported)
            attn_mask: Attention mask [seq_len, seq_len] or [batch, seq_len, seq_len]
            is_causal: Whether to apply causal masking
            average_attn_weights: Whether to average attention weights (unused)

        Returns:
            If need_weights is False:
                Attention output [batch, seq_len, embed_dim]
            If need_weights is True:
                Tuple of (output, None) - weights not supported

        Note:
            This module follows the MAGNETO architecture with layer normalization
            applied before the output projection when layer_norm=True.
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query

        # Validate inputs
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors (batch, seq_len, embed_dim), got shapes: "
                f"query={query.shape}, key={key.shape}, value={value.shape}"
            )

        b, n, d = query.shape
        if key.shape[:2] != (b, n) or value.shape[:2] != (b, n):
            raise ValueError(
                f"Batch size and sequence length must match for query, key, value. "
                f"Got query={query.shape}, key={key.shape}, value={value.shape}"
            )

        if d != self.embed_dim:
            raise ValueError(
                f"Input embedding dimension ({d}) doesn't match expected ({self.embed_dim})"
            )

        # Project to QKV
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Apply layer normalization to queries and keys if enabled
        q, k = self._apply_layer_norm(q, k)

        # Split into attention heads
        q = split_attention_heads(q, self.num_heads)  # [b, n, h, d]
        k = split_attention_heads(k, self.num_heads)
        v = split_attention_heads(v, self.num_heads)

        # Combine attention mask and key padding mask if provided
        combined_mask = self._combine_masks(
            attn_mask, key_padding_mask, batch_size=b, seq_len=n
        )

        # Apply dilated attention
        attn_output = self.attention(
            q, k, v, is_causal=is_causal, attention_mask=combined_mask
        )

        # Merge attention heads
        attn_output = merge_attention_heads(attn_output, self.num_heads, self.head_dim)

        # Apply post-attention layer norm if enabled (MAGNETO style)
        if self.multihead_config.layer_norm:
            attn_output = self.q_ln(attn_output)  # Reuse q_ln as post-attn norm

        # Output projection
        output = self.out_proj(attn_output)

        # For consistency, always return a tuple when requested
        # This matches the behavior of nn.MultiheadAttention
        if need_weights:
            # Attention weights not supported with dilated attention
            return output, None
        else:
            # Check if we should always return tuple (for compatibility)
            if getattr(self, "_always_return_tuple", False):
                return output, None
            return output

    def _combine_masks(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> Tensor | None:
        """Combine attention mask and key padding mask."""
        if attn_mask is None and key_padding_mask is None:
            return None

        combined_mask = None

        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                combined_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
                combined_mask = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.dim()}D")

        # Handle key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.dim() != 2:
                raise ValueError(
                    f"key_padding_mask must be 2D [batch, seq_len], got {key_padding_mask.dim()}D"
                )
            # Convert key_padding_mask to attention mask format
            # key_padding_mask: True = padded (ignore), False = valid
            # attention_mask: 0 = attend, -inf = ignore
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            # Expand to [batch, 1, seq_len, seq_len]
            padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
            # Convert to float mask with -inf for padded positions
            padding_mask = padding_mask.float().masked_fill(padding_mask, float("-inf"))

            if combined_mask is None:
                combined_mask = padding_mask
            else:
                # Combine masks by adding (both use -inf for ignored positions)
                combined_mask = combined_mask + padding_mask

        return combined_mask


# Backward compatibility
def create_multihead_dilated_attention(
    embed_dim: int,
    num_heads: int,
    dilation_rates: Sequence[int],
    segment_lengths: Sequence[int],
    **kwargs,
) -> MultiheadDilatedAttention:
    """
    Create a multihead dilated attention module (backward compatibility).

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dilation_rates: List of dilation rates
        segment_lengths: List of segment lengths
        **kwargs: Additional arguments

    Returns:
        MultiheadDilatedAttention module
    """
    return MultiheadDilatedAttention(
        embed_dim, num_heads, dilation_rates, segment_lengths, **kwargs
    )
