"""
Dilated Attention PyTorch Implementation

Unofficial PyTorch implementation of DilatedAttention from LongNet, 
including Ring Attention for O(n) memory scaling.
"""

__version__ = "0.2.0"

from .dilated_attention import DilatedAttention
from .multihead_dilated_attention import MultiheadDilatedAttention
from .improved_dilated_attention import ImprovedDilatedAttention
# from .distributed_dilated_attention import DistributedMultiheadDilatedAttention  # Old implementation
from .ring_dilated_attention import RingDilatedAttention
from .ring_multihead_dilated_attention import RingMultiheadDilatedAttention
from .transformer import DilatedTransformerEncoderLayer, DilatedTransformerDecoderLayer
from .long_net import LongNet

# Block-Sparse Ring Attention implementations
from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    BlockSparseMemoryPool,
    SparsePatternConfig,
    ContentAdaptiveSparsity
)
from .block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
    create_block_sparse_multihead_attention,
    create_adaptive_sparse_multihead_attention
)
from .block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
    DistributedSparsePattern
)
from .utils.sparse_pattern_utils import (
    PatternType,
    PatternConfig,
    SparsePatternGenerator,
    PatternQualityAnalyzer,
    PatternOptimizer
)

# Factory functions for easy creation
from .core import (
    create_dilated_attention,
    create_multihead_dilated_attention,
    create_block_sparse_attention,
    create_adaptive_sparse_attention,
)

# Note: Optimizations have been integrated into the main block-sparse implementations

__all__ = [
    # Original implementations
    "DilatedAttention",
    "MultiheadDilatedAttention", 
    "ImprovedDilatedAttention",
    # "DistributedMultiheadDilatedAttention",  # Old implementation
    "RingDilatedAttention",
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