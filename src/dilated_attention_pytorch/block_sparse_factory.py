"""
Unified factory for creating block-sparse attention variants.

This module provides a consistent interface for creating different
block-sparse attention implementations based on use case.
"""

from typing import List, Optional, Union, Literal

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from .block_sparse_hierarchical import (
    BlockSparseHierarchical,
    HierarchicalConfig,
    get_hierarchical_presets,
)
from .block_sparse_adaptive_fixed import (
    BlockSparseAdaptive,
    AdaptiveConfig,
)
from .block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
)
from .block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
)
from .distributed_sparse_config import DistributedSparseConfig


BlockSparseVariant = Literal[
    "auto", "base", "hierarchical", "adaptive", "multihead", "distributed"
]


def create_block_sparse_attention(
    variant: BlockSparseVariant = "auto",
    segment_lengths: Optional[List[int]] = None,
    dilation_rates: Optional[List[int]] = None,
    # Common parameters
    sparse_config: Optional[SparsePatternConfig] = None,
    sparsity_ratio: Optional[float] = None,
    block_size: Optional[int] = None,
    # Variant-specific configs
    hierarchical_config: Optional[HierarchicalConfig] = None,
    adaptive_config: Optional[AdaptiveConfig] = None,
    distributed_config: Optional[DistributedSparseConfig] = None,
    # Multihead parameters
    embed_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    # Other parameters
    **kwargs,
) -> Union[
    BlockSparseRingDilatedAttention,
    BlockSparseHierarchical,
    BlockSparseAdaptive,
    BlockSparseRingMultiheadDilatedAttention,
    BlockSparseRingDistributedDilatedAttention,
]:
    """
    Create a block-sparse attention module based on the specified variant.

    Args:
        variant: Which implementation to use:
            - "auto": Automatically select based on parameters
            - "base": Standard block-sparse attention
            - "hierarchical": Multi-scale attention patterns
            - "adaptive": Learned, content-adaptive patterns
            - "multihead": Drop-in replacement for nn.MultiheadAttention
            - "distributed": Enterprise-grade distributed implementation
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for dilated attention
        sparse_config: Sparse pattern configuration
        sparsity_ratio: Override sparsity ratio (0.1 = 90% sparse)
        block_size: Size of attention blocks
        hierarchical_config: Configuration for hierarchical variant
        adaptive_config: Configuration for adaptive variant
        distributed_config: Configuration for distributed variant
        embed_dim: Embedding dimension (required for multihead)
        num_heads: Number of attention heads (required for multihead)
        **kwargs: Additional arguments passed to the implementation

    Returns:
        Block-sparse attention module

    Examples:
        >>> # Auto-select based on context
        >>> attn = create_block_sparse_attention("auto")

        >>> # Create hierarchical attention
        >>> attn = create_block_sparse_attention(
        ...     "hierarchical",
        ...     hierarchical_config=get_hierarchical_presets()["long_range"]
        ... )

        >>> # Create multihead attention
        >>> attn = create_block_sparse_attention(
        ...     "multihead",
        ...     embed_dim=768,
        ...     num_heads=12,
        ...     sparsity_ratio=0.1
        ... )
    """
    # Default segment lengths and dilation rates
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    # Create sparse config if not provided
    if sparse_config is None and (sparsity_ratio is not None or block_size is not None):
        sparse_config = SparsePatternConfig(
            sparsity_ratio=sparsity_ratio or 0.1,
            block_size=block_size or 64,
        )

    # Auto-select variant based on parameters
    if variant == "auto":
        if distributed_config is not None:
            variant = "distributed"
        elif embed_dim is not None and num_heads is not None:
            variant = "multihead"
        elif adaptive_config is not None:
            variant = "adaptive"
        elif hierarchical_config is not None:
            variant = "hierarchical"
        else:
            variant = "base"

    # Create the appropriate implementation
    if variant == "base":
        return BlockSparseRingDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            **kwargs,
        )

    elif variant == "hierarchical":
        if hierarchical_config is None:
            # Use default hierarchical config
            hierarchical_config = HierarchicalConfig()

        return BlockSparseHierarchical(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            hierarchical_config=hierarchical_config,
            sparse_config=sparse_config,
            **kwargs,
        )

    elif variant == "adaptive":
        if adaptive_config is None:
            adaptive_config = AdaptiveConfig()

        return BlockSparseAdaptive(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            adaptive_config=adaptive_config,
            **kwargs,
        )

    elif variant == "multihead":
        if embed_dim is None or num_heads is None:
            raise ValueError(
                "embed_dim and num_heads are required for multihead variant"
            )

        return BlockSparseRingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            sparse_config=sparse_config,
            **kwargs,
        )

    elif variant == "distributed":
        if distributed_config is None:
            distributed_config = DistributedSparseConfig()

        # Note: distributed version has different API
        return BlockSparseRingDistributedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            distributed_config=distributed_config,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Choose from: auto, base, hierarchical, adaptive, multihead, distributed"
        )


def get_block_sparse_preset(
    preset_name: str,
    **override_kwargs,
) -> Union[
    BlockSparseRingDilatedAttention,
    BlockSparseHierarchical,
    BlockSparseAdaptive,
]:
    """
    Get a preset block-sparse configuration.

    Args:
        preset_name: Name of the preset:
            - "local": Local window attention only
            - "dilated": Multi-scale dilated attention
            - "global_local": Global tokens + local windows
            - "hierarchical_standard": Standard 3-level hierarchy
            - "hierarchical_fine": Fine-grained hierarchy
            - "hierarchical_long": Long-range hierarchy
            - "adaptive_standard": Standard adaptive config
            - "ultra_sparse": Extreme sparsity (99%+)
        **override_kwargs: Override preset parameters

    Returns:
        Configured block-sparse attention module
    """
    presets = {
        # Base presets
        "local": {
            "variant": "base",
            "sparse_config": SparsePatternConfig(
                pattern_type="local_window",
                sparsity_ratio=0.05,
                window_size=256,
                block_size=64,
            ),
        },
        "dilated": {
            "variant": "base",
            "sparse_config": SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.1,
                dilation_rates=[1, 2, 4, 8],
                block_size=64,
            ),
        },
        "global_local": {
            "variant": "base",
            "sparse_config": SparsePatternConfig(
                pattern_type="global_local",
                sparsity_ratio=0.1,
                global_tokens=64,
                window_size=256,
                block_size=64,
            ),
        },
        # Hierarchical presets
        "hierarchical_standard": {
            "variant": "hierarchical",
            "hierarchical_config": get_hierarchical_presets()["standard"],
        },
        "hierarchical_fine": {
            "variant": "hierarchical",
            "hierarchical_config": get_hierarchical_presets()["fine_grained"],
        },
        "hierarchical_long": {
            "variant": "hierarchical",
            "hierarchical_config": get_hierarchical_presets()["long_range"],
        },
        # Adaptive preset
        "adaptive_standard": {
            "variant": "adaptive",
            "adaptive_config": AdaptiveConfig(
                base_sparsity=0.9,
                temperature=1.0,
                learnable_temperature=True,
            ),
        },
        # Ultra sparse
        "ultra_sparse": {
            "variant": "base",
            "sparse_config": SparsePatternConfig(
                pattern_type="dilated_sparse",
                sparsity_ratio=0.01,  # 99% sparse
                block_size=128,
            ),
            "segment_lengths": [4096, 8192, 16384],
            "dilation_rates": [1, 4, 16],
        },
    }

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. Available presets: {available}"
        )

    # Get preset config and override with user kwargs
    config = presets[preset_name].copy()
    config.update(override_kwargs)

    # Extract variant and create
    variant = config.pop("variant")
    return create_block_sparse_attention(variant, **config)


# Convenience functions for specific variants
def create_hierarchical_block_sparse(
    preset: str = "standard",
    **kwargs,
) -> BlockSparseHierarchical:
    """Create hierarchical block-sparse attention with preset."""
    hierarchical_config = get_hierarchical_presets()[preset]
    return create_block_sparse_attention(
        "hierarchical",
        hierarchical_config=hierarchical_config,
        **kwargs,
    )


def create_adaptive_block_sparse(
    base_sparsity: float = 0.9,
    **kwargs,
) -> BlockSparseAdaptive:
    """Create adaptive block-sparse attention."""
    adaptive_config = AdaptiveConfig(base_sparsity=base_sparsity)
    return create_block_sparse_attention(
        "adaptive",
        adaptive_config=adaptive_config,
        **kwargs,
    )


def create_multihead_block_sparse(
    embed_dim: int,
    num_heads: int,
    sparsity_ratio: float = 0.1,
    **kwargs,
) -> BlockSparseRingMultiheadDilatedAttention:
    """Create multihead block-sparse attention."""
    return create_block_sparse_attention(
        "multihead",
        embed_dim=embed_dim,
        num_heads=num_heads,
        sparsity_ratio=sparsity_ratio,
        **kwargs,
    )
