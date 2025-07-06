"""
Dilated Attention PyTorch Implementation

Unofficial PyTorch implementation of DilatedAttention from LongNet,
including Ring Attention for O(n) memory scaling.
"""

__version__ = "0.2.0"


# Block-Sparse Ring Attention implementations
from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from .block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
    DistributedSparsePattern,
)
from .block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
)

# Optimized Block-Sparse implementations
from .block_sparse_optimized import BlockSparseOptimized
from .block_sparse_torch_sparse import BlockSparseTorchSparse
from .block_sparse_hierarchical import (
    BlockSparseHierarchical,
    HierarchicalConfig,
    create_hierarchical_attention,
    get_hierarchical_presets,
)
from .block_sparse_adaptive import (
    BlockSparseAdaptive,
    AdaptiveConfig,
    ImportanceScorer,
    AdaptiveSparsityTrainer,
    create_adaptive_block_sparse,
)

# Factory functions for easy creation
from .core import (
    create_adaptive_sparse_attention,
    create_block_sparse_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
)
from .dilated_attention import DilatedAttention
from .improved_dilated_attention import ImprovedDilatedAttention
from .improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention
from .long_net import LongNet
from .multihead_dilated_attention import MultiheadDilatedAttention

# Hybrid Ring Attention - best of V2 and V3
from .ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
    RingDilatedAttentionTrue,  # Alias
)
from .ring_multihead_dilated_attention_hybrid import (
    RingMultiheadDilatedAttentionHybrid,
    create_ring_multihead_attention_hybrid,
)

# Ring Attention implementations
from .ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
    create_production_ring_attention,
)

# Hilbert Ring Attention - combines Ring Attention with Hilbert ordering
from .ring_dilated_attention_hilbert_optimized import (
    RingDilatedAttentionHilbertOptimized,
)

from .transformer import DilatedTransformerDecoderLayer, DilatedTransformerEncoderLayer
from .utils.sparse_pattern_utils import (
    PatternConfig,
    PatternOptimizer,
    PatternQualityAnalyzer,
    PatternType,
    SparsePatternGenerator,
)

# Alias for backward compatibility
RingDilatedAttention = RingDilatedAttentionHybrid


# Note: Optimizations have been integrated into the main block-sparse implementations

__all__ = [
    # Block-Sparse implementations
    "BlockSparseRingDilatedAttention",
    "BlockSparseRingDistributedDilatedAttention",
    "BlockSparseRingMultiheadDilatedAttention",
    "BlockSparseOptimized",
    "BlockSparseTorchSparse",
    "BlockSparseHierarchical",
    "HierarchicalConfig",
    "create_hierarchical_attention",
    "get_hierarchical_presets",
    "BlockSparseAdaptive",
    "AdaptiveConfig",
    "ImportanceScorer",
    "AdaptiveSparsityTrainer",
    "create_adaptive_block_sparse",
    # Ring Attention implementations
    "RingDilatedAttention",  # Alias for backward compatibility
    "RingDilatedAttentionProduction",
    "RingAttentionConfig",
    "create_production_ring_attention",
    # Hilbert Ring Attention - combines cache efficiency with ring attention
    "RingDilatedAttentionHilbertOptimized",  # Non-deprecated Hilbert implementation
    # Hybrid Ring Attention - true O(n/p) scaling with all features
    "RingDilatedAttentionHybrid",
    "RingDilatedAttentionTrue",  # Alias for Hybrid
    "RingMultiheadDilatedAttentionHybrid",
    "create_ring_multihead_attention_hybrid",
    # Original implementations
    "DilatedAttention",
    "DilatedTransformerDecoderLayer",
    "DilatedTransformerEncoderLayer",
    "DistributedSparseConfig",
    "DistributedSparsePattern",
    "ImprovedDilatedAttention",
    "ImprovedMultiheadDilatedAttention",
    "LongNet",
    "MultiheadDilatedAttention",
    "PatternConfig",
    "PatternOptimizer",
    "PatternQualityAnalyzer",
    "PatternType",
    # Configuration and utility classes
    "SparsePatternConfig",
    "SparsePatternGenerator",
    "create_adaptive_sparse_attention",
    "create_block_sparse_attention",
    # Factory functions (v0.2.0+)
    "create_dilated_attention",
    "create_multihead_dilated_attention",
]

# Flash optimizations are now integrated into V2Collective
HAS_FLASH_RING = True  # Flash is integrated into V2Collective
