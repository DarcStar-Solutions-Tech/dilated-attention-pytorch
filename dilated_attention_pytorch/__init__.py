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

# Factory functions for easy creation
from .core import (
    create_adaptive_sparse_attention,
    create_block_sparse_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
)
from .dilated_attention import DilatedAttention
from .improved_dilated_attention import ImprovedDilatedAttention
from .improved_distributed_dilated_attention import (
    DistributedImprovedDilatedAttention,
    DistributedImprovedMultiheadDilatedAttention,
)
from .improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention
from .long_net import LongNet
from .multihead_dilated_attention import MultiheadDilatedAttention

# Ring Attention implementations
from .ring_dilated_attention_v2 import RingDilatedAttentionV2
from .ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
    create_production_ring_attention,
)

# Note: Use create_multihead_dilated_attention('ring') for Ring Attention functionality
from .transformer import DilatedTransformerDecoderLayer, DilatedTransformerEncoderLayer
from .utils.sparse_pattern_utils import (
    PatternConfig,
    PatternOptimizer,
    PatternQualityAnalyzer,
    PatternType,
    SparsePatternGenerator,
)

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
    # Ring Attention implementations
    "RingDilatedAttentionV2",
    "RingDilatedAttentionProduction",
    "RingAttentionConfig",
    "create_production_ring_attention",
    # Original implementations
    "DilatedAttention",
    "DilatedTransformerDecoderLayer",
    "DilatedTransformerEncoderLayer",
    "DistributedImprovedDilatedAttention",
    "DistributedImprovedMultiheadDilatedAttention",
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
