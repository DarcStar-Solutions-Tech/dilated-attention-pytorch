"""
Fixed RingDilatedAttentionProduction with standardized API.

This is a wrapper that provides the standardized API while maintaining
compatibility with the original implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

from .ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
)
from .core.standardized_api import (
    StandardizedRingConfig,
    StandardizedRingAttentionMixin,
)


class RingDilatedAttentionProductionFixed(nn.Module, StandardizedRingAttentionMixin):
    """
    Fixed version of RingDilatedAttentionProduction with standardized API.

    This wrapper provides backward compatibility with the expected API
    while using the original implementation internally.
    """

    def __init__(
        self,
        # Accept both StandardizedRingConfig and individual parameters
        config: Optional[StandardizedRingConfig] = None,
        # Individual parameters for backward compatibility
        dim: Optional[int] = None,
        heads: Optional[int] = None,
        segment_lengths: Optional[List[int]] = None,
        dilation_rates: Optional[List[int]] = None,
        ring_size: Optional[int] = None,
        dropout: float = 0.0,
        # RingAttentionConfig can also be passed directly
        ring_attention_config: Optional[RingAttentionConfig] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle different initialization patterns
        if ring_attention_config is not None:
            # Direct RingAttentionConfig provided
            self.ring_config = ring_attention_config
            self.dim = dim or kwargs.get("head_dim", 64)
            self.heads = heads or kwargs.get("num_heads", 8)
        elif config is not None:
            # StandardizedRingConfig provided
            self.dim = config.dim
            self.heads = config.heads
            self.ring_config = config.to_ring_attention_config()
        else:
            # Individual parameters provided
            if segment_lengths is None or dilation_rates is None:
                raise ValueError("segment_lengths and dilation_rates are required")

            self.dim = dim or kwargs.get("head_dim", 64)
            self.heads = heads or kwargs.get("num_heads", 8)

            # Create StandardizedRingConfig from parameters
            std_config = StandardizedRingConfig(
                dim=self.dim,
                heads=self.heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                ring_size=ring_size,
                dropout=dropout,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if hasattr(StandardizedRingConfig, k)
                },
            )
            self.ring_config = std_config.to_ring_attention_config()

        # Initialize the actual implementation
        self.attention = RingDilatedAttentionProduction(self.ring_config)

        # Store parameters for property access
        self.segment_lengths = self.ring_config.segment_lengths
        self.dilation_rates = self.ring_config.dilation_rates
        self.dropout_p = self.ring_config.dropout
        self.ring_size = self.ring_config.ring_size

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass with standardized input/output format.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to use causal masking
            return_attention_weights: Whether to return attention weights

        Returns:
            output: Attention output [batch, seq_len, num_heads, head_dim]
            attention_weights: Optional attention weights if requested
        """
        # The original implementation expects the same format, so just pass through
        output = self.attention(q, k, v, is_causal=is_causal)

        if return_attention_weights:
            # Original doesn't return weights, so return None
            return output, None
        else:
            return output

    @classmethod
    def from_config(cls, config: StandardizedRingConfig, **kwargs):
        """Create instance from standardized config."""
        return cls(config=config, **kwargs)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"dim={self.dim}, heads={self.heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"dropout={self.dropout_p}"
        )
