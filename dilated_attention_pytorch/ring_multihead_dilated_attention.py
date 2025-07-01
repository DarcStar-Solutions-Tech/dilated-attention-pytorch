"""
Ring Multihead Dilated Attention - Multihead wrapper for Ring Attention.

This module provides a drop-in replacement for nn.MultiheadAttention using
the Ring Dilated Attention mechanism for O(n) memory scaling.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .ring_dilated_attention_v2_flash import RingDilatedAttentionV2Flash


class RingMultiheadDilatedAttention(nn.Module):
    """
    Multihead wrapper for Ring Dilated Attention with Flash Attention.

    This provides a drop-in replacement for nn.MultiheadAttention with:
    - O(n/ring_size) memory scaling through Ring Attention
    - Flash Attention optimization for improved performance
    - Support for dilated attention patterns
    - MAGNETO-style LayerNorm for improved training stability
    - Compatible with all Ring Attention optimizations

    Args:
        embed_dim: Total embedding dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        segment_lengths: List of segment lengths for dilated attention
        dilation_rates: List of dilation rates corresponding to segments
        dropout: Dropout probability
        bias: Whether to include bias in projections
        layer_norm: Whether to apply layer normalization (MAGNETO)
        layer_norm_eps: Epsilon for layer normalization
        gamma_init: Initial value for MAGNETO gamma parameters
        ring_size: Size of the ring for distributed operation
        device: Device to place parameters on
        dtype: Data type for parameters
        batch_first: Whether input is batch-first (default: True)
        enable_memory_pool: Whether to use memory pooling
        enable_pattern_cache: Whether to cache attention patterns
        use_flash_attention: Whether to use Flash Attention optimization
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = False,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        batch_first: bool = True,
        # Memory optimization options
        enable_memory_pool: bool = True,
        enable_pattern_cache: bool = True,
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # Validate inputs
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        assert len(segment_lengths) == len(dilation_rates), (
            "segment_lengths and dilation_rates must have same length"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.bias = bias
        self.layer_norm = layer_norm
        self.layer_norm_eps = layer_norm_eps
        self.gamma_init = gamma_init
        self.batch_first = batch_first

        # Device and dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or torch.float32

        # Factory kwargs for linear layers
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # Input projections - fused QKV for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Optional layer normalization (MAGNETO)
        if layer_norm:
            self.q_norm = nn.LayerNorm(
                self.head_dim, eps=layer_norm_eps, **factory_kwargs
            )
            self.k_norm = nn.LayerNorm(
                self.head_dim, eps=layer_norm_eps, **factory_kwargs
            )

            # MAGNETO gamma parameters
            self.gamma_q = nn.Parameter(
                torch.full((self.head_dim,), gamma_init, **factory_kwargs)
            )
            self.gamma_k = nn.Parameter(
                torch.full((self.head_dim,), gamma_init, **factory_kwargs)
            )
        else:
            self.q_norm = None
            self.k_norm = None
            self.gamma_q = None
            self.gamma_k = None

        # Ring Attention module - using Flash variant for best performance
        self.ring_attention = RingDilatedAttentionV2Flash(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=enable_memory_pool,
            use_pattern_cache=enable_pattern_cache,
            use_flash_attention=use_flash_attention,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        # QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        # Output projection
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through Ring Multihead Dilated Attention.

        Args:
            query: Query tensor of shape (batch, seq_len, embed_dim)
            key: Key tensor (if None, uses query)
            value: Value tensor (if None, uses key)
            need_weights: Whether to return attention weights (not supported)
            attn_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Tuple of (output, None) where output has shape (batch, seq_len, embed_dim)
            The second element is None as Ring Attention doesn't return weights
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = key

        # For cross-attention, we need separate projections
        if query is not key:
            raise NotImplementedError(
                "Cross-attention not yet implemented for Ring Attention. "
                "Currently only self-attention is supported."
            )

        if need_weights:
            raise ValueError(
                "Ring Attention does not support returning attention weights "
                "due to distributed computation."
            )

        # Get shapes
        if self.batch_first:
            batch_size, seq_len, _ = query.shape
        else:
            seq_len, batch_size, _ = query.shape
            # Convert to batch first
            query = query.transpose(0, 1)

        # Fused QKV projection for self-attention
        qkv = self.qkv_proj(query)  # (batch, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, batch, seq_len, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply layer normalization if enabled (MAGNETO)
        if self.layer_norm:
            q = self.q_norm(q) * self.gamma_q
            k = self.k_norm(k) * self.gamma_k

        # Apply Ring Dilated Attention
        # Ring attention expects (batch, seq_len, num_heads, head_dim)
        # Note: RingDilatedAttentionV2Collective doesn't support attention_mask yet
        if attn_mask is not None:
            raise NotImplementedError(
                "Ring Attention does not yet support custom attention masks. "
                "Only is_causal masking is supported."
            )

        attn_output = self.ring_attention(q, k, v, is_causal=is_causal)

        # Reshape and apply output projection
        # attn_output: (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        # Convert back to seq_first if needed
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        s = (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"dropout={self.dropout}"
        )
        if self.layer_norm:
            s += f", layer_norm=True, gamma_init={self.gamma_init}"
        return s
