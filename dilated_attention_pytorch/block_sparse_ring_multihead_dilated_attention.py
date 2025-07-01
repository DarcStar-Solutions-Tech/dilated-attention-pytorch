"""
Block-Sparse Ring Multihead Dilated Attention - Memory Efficient Implementation

Multihead wrapper for the memory-efficient block-sparse implementation.
"""

from typing import Any

from torch import Tensor, nn

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


class BlockSparseRingMultiheadDilatedAttention(nn.Module):
    """
    Memory-efficient multihead block-sparse ring dilated attention.

    Drop-in replacement for nn.MultiheadAttention with block-sparse optimization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bias: bool = True,
        sparse_config: SparsePatternConfig | None = None,
        batch_first: bool = True,
        **kwargs,
    ):
        """
        Initialize multihead block-sparse attention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            segment_lengths: List of segment lengths for dilated attention
            dilation_rates: List of dilation rates
            dropout: Dropout probability
            bias: Whether to use bias in projections
            sparse_config: Sparse pattern configuration
            batch_first: Whether batch dimension is first
            **kwargs: Additional arguments for BlockSparseRingDilatedAttention
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.dropout = dropout

        # Extract device and dtype from kwargs
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        # Factory kwargs for device and dtype
        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Sparse attention
        self.sparse_attention = BlockSparseRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            **kwargs,
        )

        # For compatibility with some implementations
        self._always_return_tuple = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> tuple[Tensor, dict[str, Any] | None]:
        """
        Forward pass.

        Args:
            query: Query tensor [batch, seq_len, embed_dim] if batch_first
            key: Key tensor [batch, seq_len, embed_dim] if batch_first
            value: Value tensor [batch, seq_len, embed_dim] if batch_first
            key_padding_mask: Mask for padded keys [batch, seq_len]
            need_weights: Whether to return attention weights
            attn_mask: Attention mask [seq_len, seq_len] or [batch, seq_len, seq_len]
            is_causal: Whether to apply causal masking
            average_attn_weights: Ignored (for compatibility)

        Returns:
            output: Attention output [batch, seq_len, embed_dim]
            attn_weights: Sparse attention weights dict if need_weights, else None
        """
        _ = average_attn_weights  # Ignored for compatibility
        # Handle batch_first
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape

        # Projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for attention: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] where True means padded
            # Expand to [batch, seq_len, num_heads, 1]
            mask = key_padding_mask.unsqueeze(2).unsqueeze(3)
            k = k.masked_fill(mask, 0.0)
            v = v.masked_fill(mask, 0.0)

        # Apply attention mask if provided
        if attn_mask is not None:
            # This is more complex for sparse attention
            # For now, we'll log a warning
            import warnings

            warnings.warn(
                "attn_mask is not fully supported in BlockSparseRingMultiheadDilatedAttention. "
                "Use is_causal=True for causal masking instead.",
                stacklevel=2,
            )

        # Compute sparse attention
        if need_weights:
            attn_output, attn_weights = self.sparse_attention(
                q, k, v, is_causal=is_causal, return_attention_weights=True
            )
        else:
            attn_output = self.sparse_attention(
                q, k, v, is_causal=is_causal, return_attention_weights=False
            )
            attn_weights = None

        # Reshape output: [batch, seq_len, embed_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout_layer(output)

        # Handle batch_first for output
        if not self.batch_first:
            output = output.transpose(0, 1)

        # Return format
        if need_weights or self._always_return_tuple:
            return output, attn_weights
        return output
