"""
Dilated Attention PyTorch Implementation

Unofficial PyTorch implementation of DilatedAttention from LongNet,
including Ring Attention for O(n) memory scaling.
"""

__version__ = "0.2.0"


# Block-Sparse Ring Attention implementations
from .block_sparse_ring_dilated_attention import (
    BlockSparseMemoryPool, BlockSparseRingDilatedAttention,
    ContentAdaptiveSparsity, SparsePatternConfig)
from .block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention, DistributedSparseConfig,
    DistributedSparsePattern)
from .block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
    create_adaptive_sparse_multihead_attention,
    create_block_sparse_multihead_attention)
# Factory functions for easy creation
from .core import (create_adaptive_sparse_attention,
                   create_block_sparse_attention, create_dilated_attention,
                   create_multihead_dilated_attention)
from .dilated_attention import DilatedAttention
from .improved_dilated_attention import ImprovedDilatedAttention
from .long_net import LongNet
from .multihead_dilated_attention import MultiheadDilatedAttention
# from .distributed_dilated_attention import DistributedMultiheadDilatedAttention  # Old implementation
from .ring_dilated_attention import RingDilatedAttention
from .ring_dilated_attention_unfold_v2 import UnfoldRingDilatedAttention
from .ring_multihead_dilated_attention import RingMultiheadDilatedAttention
from .transformer import (DilatedTransformerDecoderLayer,
                          DilatedTransformerEncoderLayer)
from .utils.sparse_pattern_utils import (PatternConfig, PatternOptimizer,
                                         PatternQualityAnalyzer, PatternType,
                                         SparsePatternGenerator)

# Note: Optimizations have been integrated into the main block-sparse implementations

__all__ = [
    # Original implementations
    "DilatedAttention",
    "MultiheadDilatedAttention",
    "ImprovedDilatedAttention",
    "ImprovedMultiheadDilatedAttention",
    "DistributedImprovedDilatedAttention",
    "DistributedImprovedMultiheadDilatedAttention",
    # "DistributedMultiheadDilatedAttention",  # Old implementation
    "RingDilatedAttention",
    "UnfoldRingDilatedAttention",
    "RingMultiheadDilatedAttention",
    "DilatedTransformerEncoderLayer",
    "DilatedTransformerDecoderLayer",
    "LongNet",
    # Block-Sparse implementations
    "BlockSparseRingDilatedAttention",
    "BlockSparseRingMultiheadDilatedAttention",
    "BlockSparseRingDistributedDilatedAttention",
    # Configuration and utility classes
    "BlockSparseMemoryPool",
    "SparsePatternConfig",
    "DistributedSparseConfig",
    "DistributedSparsePattern",
    "PatternType",
    "PatternConfig",
    "ContentAdaptiveSparsity",
    "SparsePatternGenerator",
    "PatternQualityAnalyzer",
    "PatternOptimizer",
    # Convenience functions
    "create_block_sparse_multihead_attention",
    "create_adaptive_sparse_multihead_attention",
    # Factory functions (v0.2.0+)
    "create_dilated_attention",
    "create_multihead_dilated_attention",
    "create_block_sparse_attention",
    "create_adaptive_sparse_attention",
]
