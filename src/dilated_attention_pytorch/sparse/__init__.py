"""
Block-sparse attention implementations.

This module contains various block-sparse attention implementations for
efficient processing with reduced memory and computational requirements.
"""

from .block_sparse_attention import (
    BlockSparseAttention,
    SparsePatternConfig,
)
from .block_sparse_ring_dilated_attention_fixed import (
    BlockSparseRingDilatedAttentionFixed,
)
from .block_sparse_ring_dilated_attention_hilbert_post_pattern import (
    BlockSparseRingDilatedAttentionHilbertPostPattern,
)
from .block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
)
from .block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
)
from .block_sparse_adaptive import (
    BlockSparseAdaptive,
    AdaptiveConfig,
    AdaptiveSparsityTrainer,
)
from .block_sparse_adaptive_fixed import BlockSparseAdaptive as BlockSparseAdaptiveFixed
from .block_sparse_factory import (
    create_block_sparse_attention,
    create_adaptive_block_sparse,
    create_multihead_block_sparse,
)

__all__ = [
    # Core implementations
    "BlockSparseAttention",
    "BlockSparseRingDilatedAttentionFixed",
    "BlockSparseRingDilatedAttentionHilbertPostPattern",
    "BlockSparseRingMultiheadDilatedAttention",
    "BlockSparseRingDistributedDilatedAttention",
    # Adaptive implementations
    "BlockSparseAdaptive",
    "BlockSparseAdaptiveFixed",  # Alias for BlockSparseAdaptive from fixed module
    "AdaptiveConfig",
    "AdaptiveSparsityTrainer",
    # Configuration
    "SparsePatternConfig",
    # Factory functions
    "create_block_sparse_attention",
    "create_multihead_block_sparse",
    "create_adaptive_block_sparse",
]
