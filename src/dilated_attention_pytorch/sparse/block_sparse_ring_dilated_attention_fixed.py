"""
Fixed BlockSparseRingDilatedAttention with standardized API.

This wrapper ensures the Block-Sparse implementation uses the standardized API
consistently.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Tuple

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from ..core.standardized_api import (
    StandardizedRingConfig,
    StandardizedRingAttentionMixin,
)


class BlockSparseRingDilatedAttentionFixed(nn.Module, StandardizedRingAttentionMixin):
    """
    Fixed version of BlockSparseRingDilatedAttention with standardized API.

    This wrapper ensures consistent API while maintaining the efficient
    block-sparse implementation.
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
        # Sparse-specific parameters
        sparse_pattern_config: Optional[
            Union[Dict[str, Any], SparsePatternConfig]
        ] = None,
        sparsity_ratio: Optional[float] = None,
        block_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        # Handle different initialization patterns
        if config is not None:
            # StandardizedRingConfig provided
            self.dim = config.dim
            self.heads = config.heads
            segment_lengths = config.segment_lengths
            dilation_rates = config.dilation_rates
            ring_size = config.ring_size or 1
            dropout = config.dropout

            # Get sparse config from StandardizedRingConfig
            if config.sparse_pattern_config:
                sparse_pattern_config = config.to_sparse_pattern_config()
            elif config.sparsity_ratio is not None or config.block_size is not None:
                sparse_pattern_config = config.to_sparse_pattern_config()
            else:
                sparse_pattern_config = sparse_pattern_config
        else:
            # Individual parameters provided
            if segment_lengths is None or dilation_rates is None:
                raise ValueError("segment_lengths and dilation_rates are required")

            self.dim = dim or kwargs.get("head_dim", 64)
            self.heads = heads or kwargs.get("num_heads", 8)
            ring_size = ring_size or 1

        # Prepare kwargs for the original implementation
        init_kwargs = {
            "dropout": dropout,
            "ring_size": ring_size,
            "sparse_pattern_config": sparse_pattern_config,
        }

        # Add any additional kwargs that the original implementation accepts
        for key in [
            "use_gradient_checkpointing",
            "enable_memory_pool",
            "mixed_precision",
        ]:
            if key in kwargs:
                init_kwargs[key] = kwargs[key]

        # Handle separate sparsity parameters
        if sparsity_ratio is not None:
            init_kwargs["sparsity_ratio"] = sparsity_ratio
        if block_size is not None:
            init_kwargs["block_size"] = block_size

        # Initialize the actual implementation
        try:
            self.attention = BlockSparseRingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                **init_kwargs,
            )
        except Exception:
            # If the original fails, try with sparse_config instead of sparse_pattern_config
            if "sparse_pattern_config" in init_kwargs:
                init_kwargs["sparse_config"] = init_kwargs.pop("sparse_pattern_config")
            self.attention = BlockSparseRingDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                **init_kwargs,
            )

        # Store parameters for property access
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout_p = dropout
        self.ring_size = ring_size

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
            attention_weights: Optional attention weights (sparse format) if requested
        """
        return self.attention(
            q,
            k,
            v,
            is_causal=is_causal,
            return_attention_weights=return_attention_weights,
        )

    @classmethod
    def from_config(cls, config: StandardizedRingConfig, **kwargs):
        """Create instance from standardized config."""
        return cls(config=config, **kwargs)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        sparse_config = getattr(self.attention, "sparse_config", None)
        sparse_info = ""
        if sparse_config:
            sparse_info = (
                f", sparsity_ratio={getattr(sparse_config, 'sparsity_ratio', 'N/A')}"
            )
            sparse_info += f", block_size={getattr(sparse_config, 'block_size', 'N/A')}"

        return (
            f"dim={self.dim}, heads={self.heads}, "
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"dropout={self.dropout_p}{sparse_info}"
        )
