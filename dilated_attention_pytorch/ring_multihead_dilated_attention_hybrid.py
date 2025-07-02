"""
Ring Multihead Dilated Attention Hybrid - Drop-in replacement for nn.MultiheadAttention.

Wraps the hybrid ring attention with projection layers for a complete multihead implementation.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ring_dilated_attention_hybrid import RingDilatedAttentionHybrid


class RingMultiheadDilatedAttentionHybrid(nn.Module):
    """
    Multihead wrapper for Hybrid Ring Dilated Attention.

    Drop-in replacement for nn.MultiheadAttention with:
    - True ring attention (O(n/p) memory scaling)
    - Full dilation support in multi-GPU mode
    - All V2 optimizations
    - V3's numerical stability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        # Ring configuration
        ring_size: Optional[int] = None,
        # V2 optimization features
        enable_memory_pool: bool = True,
        use_pattern_cache: bool = True,
        use_flash_attention: bool = True,
        # MAGNETO features (from V2)
        layer_norm: bool = False,
        gamma_init: float = 1.0,
        gate_fn: Optional[str] = None,
        gate_bias: float = 0.0,
        gate_temperature: float = 1.0,
    ):
        """Initialize multihead hybrid ring attention."""
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        # Key and value dimensions
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        # Projection layers
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            self.kdim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            self.vdim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.out_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        # Optional bias for key/value
        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.zeros(1, 1, embed_dim, device=device, dtype=dtype)
            )
            self.bias_v = nn.Parameter(
                torch.zeros(1, 1, embed_dim, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias_k", None)
            self.register_parameter("bias_v", None)

        # MAGNETO features (from V2)
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if layer_norm:
            self.q_ln = nn.LayerNorm(embed_dim, device=device, dtype=dtype)
            self.k_ln = nn.LayerNorm(embed_dim, device=device, dtype=dtype)

            # Initialize gamma parameters for MAGNETO
            if gamma_init != 1.0:
                with torch.no_grad():
                    self.q_ln.weight.fill_(gamma_init)
                    self.k_ln.weight.fill_(gamma_init)

        # Gating mechanism (from V2)
        self.gate_fn = gate_fn
        self.gate_bias = gate_bias
        self.gate_temperature = gate_temperature

        if gate_fn is not None:
            self.gate_proj = nn.Linear(
                embed_dim, embed_dim, bias=True, device=device, dtype=dtype
            )
            if gate_bias != 0.0:
                nn.init.constant_(self.gate_proj.bias, gate_bias)

        # Core attention module - the hybrid implementation
        self.attention = RingDilatedAttentionHybrid(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=enable_memory_pool,
            use_pattern_cache=use_pattern_cache,
            use_flash_attention=use_flash_attention,
        )

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform (matching nn.MultiheadAttention)."""
        for param in [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight]:
            nn.init.xavier_uniform_(param)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

        # Special initialization for output projection (from V2)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass compatible with nn.MultiheadAttention.

        Args:
            query, key, value: Input tensors
            key_padding_mask: Mask for padded positions
            need_weights: Whether to return attention weights (not supported)
            attn_mask: Additional attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to apply causal masking

        Returns:
            (output, None) - attention weights not supported in ring attention
        """
        if not self.batch_first:
            # Convert to batch_first
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Apply layer norm if enabled (MAGNETO)
        if self.layer_norm:
            q = self.q_ln(q)
            k = self.k_ln(k)

        # Add bias if enabled
        if self.add_bias_kv:
            k = torch.cat([k, self.bias_k.expand(batch_size, -1, -1)], dim=1)
            v = torch.cat([v, self.bias_v.expand(batch_size, -1, -1)], dim=1)

        # Add zero attention if enabled
        if self.add_zero_attn:
            zero_attn_shape = (batch_size, 1, self.embed_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
            )
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
            )

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        # Apply attention
        attn_output = self.attention(q, k, v, is_causal=is_causal)

        # Reshape output
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.embed_dim)

        # Apply gating if enabled (from V2)
        if self.gate_fn is not None:
            gate = self.gate_proj(query)

            if self.gate_fn == "sigmoid":
                gate = torch.sigmoid(gate / self.gate_temperature)
            elif self.gate_fn == "tanh":
                gate = torch.tanh(gate / self.gate_temperature)
            elif self.gate_fn == "relu":
                gate = F.relu(gate)
            elif self.gate_fn == "gelu":
                gate = F.gelu(gate)
            else:
                raise ValueError(f"Unknown gate function: {self.gate_fn}")

            attn_output = gate * attn_output

        # Output projection
        output = self.out_proj(attn_output)

        if not self.batch_first:
            output = output.transpose(0, 1)

        # Ring attention doesn't support returning attention weights
        if need_weights:
            warnings.warn(
                "Ring attention does not support returning attention weights. "
                "Returning None for attention weights.",
                UserWarning,
            )

        return output, None


# Create convenience functions for easy instantiation
def create_ring_multihead_attention_hybrid(
    embed_dim: int,
    num_heads: int,
    segment_lengths: list[int],
    dilation_rates: list[int],
    **kwargs,
) -> RingMultiheadDilatedAttentionHybrid:
    """
    Create a hybrid ring multihead attention module.

    This provides the best of both V2 and V3:
    - True ring attention with O(n/p) memory scaling
    - Full dilation support
    - All optimizations from V2
    - Numerical stability from V3
    """
    return RingMultiheadDilatedAttentionHybrid(
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        **kwargs,
    )
