"""
Standardized API for Ring and Hilbert Attention implementations.

This module provides a consistent API across all Ring-based attention implementations
to fix the inconsistent initialization patterns.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch.nn as nn


@dataclass
class StandardizedRingConfig:
    """Standardized configuration for all Ring-based attention implementations."""

    # Core attention parameters
    dim: Optional[int] = None  # Head dimension (per-head)
    heads: Optional[int] = None  # Number of attention heads
    embed_dim: Optional[int] = None  # Total embedding dimension (heads * dim)

    # Dilated attention parameters
    segment_lengths: List[int] = None
    dilation_rates: List[int] = None

    # Ring attention parameters
    ring_size: Optional[int] = None

    # Common parameters
    dropout: float = 0.0
    bias: bool = False

    # Memory and optimization
    use_memory_pool: bool = True
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    memory_pool_size: int = 10

    # Sparse pattern config (for block-sparse variants)
    sparse_pattern_config: Optional[Dict[str, Any]] = None
    sparsity_ratio: Optional[float] = None
    block_size: Optional[int] = None

    # Hilbert-specific parameters
    use_hilbert: bool = False
    hilbert_chunk_size: Optional[int] = None

    # Flash attention parameters
    use_flash: bool = False

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Handle embed_dim vs (dim, heads)
        if self.embed_dim is not None and self.dim is None and self.heads is not None:
            self.dim = self.embed_dim // self.heads
        elif self.dim is not None and self.heads is not None and self.embed_dim is None:
            self.embed_dim = self.dim * self.heads

        # Validate segment lengths and dilation rates
        if self.segment_lengths and self.dilation_rates:
            assert len(self.segment_lengths) == len(self.dilation_rates), (
                "segment_lengths and dilation_rates must have same length"
            )

    def to_ring_attention_config(self):
        """Convert to RingAttentionConfig for compatibility."""
        from ..ring_dilated_attention_production import RingAttentionConfig

        return RingAttentionConfig(
            segment_lengths=self.segment_lengths,
            dilation_rates=self.dilation_rates,
            dropout=self.dropout,
            ring_size=self.ring_size,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_memory_pool=self.use_memory_pool,
            memory_pool_size=self.memory_pool_size,
            mixed_precision=self.mixed_precision,
        )

    def to_sparse_pattern_config(self):
        """Convert to SparsePatternConfig for compatibility."""
        from ..block_sparse_ring_dilated_attention import SparsePatternConfig

        if self.sparse_pattern_config:
            return SparsePatternConfig(**self.sparse_pattern_config)
        elif self.sparsity_ratio is not None or self.block_size is not None:
            kwargs = {}
            if self.sparsity_ratio is not None:
                kwargs["sparsity_ratio"] = self.sparsity_ratio
            if self.block_size is not None:
                kwargs["block_size"] = self.block_size
            return SparsePatternConfig(**kwargs)
        else:
            return SparsePatternConfig()


def create_standardized_ring_attention(
    attention_type: str,
    # Core parameters
    dim: Optional[int] = None,
    heads: Optional[int] = None,
    embed_dim: Optional[int] = None,
    # Dilated attention parameters
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    # Ring parameters
    ring_size: Optional[int] = None,
    # Common parameters
    dropout: float = 0.0,
    # Sparse parameters
    sparse_pattern_config: Optional[Dict[str, Any]] = None,
    sparsity_ratio: Optional[float] = None,
    block_size: Optional[int] = None,
    # Other parameters
    **kwargs,
) -> nn.Module:
    """
    Factory function to create Ring attention implementations with standardized API.

    Args:
        attention_type: Type of attention to create:
            - 'production': RingDilatedAttentionProduction
            - 'hybrid': RingDilatedAttentionHybrid
            - 'hilbert': RingDilatedAttentionHilbertOptimized
            - 'block_sparse': BlockSparseRingDilatedAttention
            - 'multihead_hybrid': RingMultiheadDilatedAttentionHybrid
            - 'block_sparse_multihead': BlockSparseRingMultiheadDilatedAttention
        dim: Head dimension (required for most types)
        heads: Number of attention heads (required for most types)
        embed_dim: Total embedding dimension (alternative to dim * heads)
        segment_lengths: List of segment lengths for dilated attention
        dilation_rates: List of dilation rates for dilated attention
        ring_size: Ring size for distributed attention
        dropout: Dropout rate
        sparse_pattern_config: Configuration for sparse patterns
        sparsity_ratio: Sparsity ratio for block-sparse variants
        block_size: Block size for block-sparse variants
        **kwargs: Additional parameters specific to each implementation

    Returns:
        Initialized attention module with standardized API
    """
    # Create standardized config
    config = StandardizedRingConfig(
        dim=dim,
        heads=heads,
        embed_dim=embed_dim,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        ring_size=ring_size,
        dropout=dropout,
        sparse_pattern_config=sparse_pattern_config,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
        **{k: v for k, v in kwargs.items() if hasattr(StandardizedRingConfig, k)},
    )

    # Route to appropriate implementation
    if attention_type == "production":
        from ..ring_dilated_attention_production_fixed import (
            RingDilatedAttentionProductionFixed,
        )

        return RingDilatedAttentionProductionFixed(config)

    elif attention_type == "hybrid":
        from ..ring_dilated_attention_hybrid_fixed import (
            RingDilatedAttentionHybridFixed,
        )

        return RingDilatedAttentionHybridFixed(config)

    elif attention_type == "hilbert":
        from ..ring_dilated_attention_hilbert_optimized_fixed import (
            RingDilatedAttentionHilbertOptimizedFixed,
        )

        return RingDilatedAttentionHilbertOptimizedFixed(config)

    elif attention_type == "block_sparse":
        from ..block_sparse_ring_dilated_attention_fixed import (
            BlockSparseRingDilatedAttentionFixed,
        )

        return BlockSparseRingDilatedAttentionFixed(config)

    elif attention_type == "multihead_hybrid":
        from ..ring_multihead_dilated_attention_hybrid_fixed import (
            RingMultiheadDilatedAttentionHybridFixed,
        )

        return RingMultiheadDilatedAttentionHybridFixed(config)

    elif attention_type == "block_sparse_multihead":
        from ..block_sparse_ring_multihead_dilated_attention_fixed import (
            BlockSparseRingMultiheadDilatedAttentionFixed,
        )

        return BlockSparseRingMultiheadDilatedAttentionFixed(config)

    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


class StandardizedRingAttentionMixin:
    """
    Mixin class to add standardized API support to existing implementations.

    This allows gradual migration of existing classes to the standardized API.
    """

    @classmethod
    def from_config(cls, config: StandardizedRingConfig, **kwargs):
        """Create instance from standardized config."""
        raise NotImplementedError("Subclasses must implement from_config")

    @classmethod
    def from_params(
        cls,
        dim: Optional[int] = None,
        heads: Optional[int] = None,
        embed_dim: Optional[int] = None,
        segment_lengths: List[int] = None,
        dilation_rates: List[int] = None,
        ring_size: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Create instance from individual parameters."""
        config = StandardizedRingConfig(
            dim=dim,
            heads=heads,
            embed_dim=embed_dim,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            ring_size=ring_size,
            dropout=dropout,
            **kwargs,
        )
        return cls.from_config(config, **kwargs)
