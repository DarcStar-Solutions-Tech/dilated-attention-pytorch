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

# DEPRECATED: These Ring Attention implementations are broken and will be removed in v0.3.0
# They incorrectly divide queries across devices. Use create_multihead_dilated_attention('ring') instead.
from .ring_dilated_attention import RingDilatedAttention  # DEPRECATED - broken implementation
from .ring_dilated_attention_unfold_v2 import UnfoldRingDilatedAttention  # DEPRECATED
from .ring_multihead_dilated_attention import RingMultiheadDilatedAttention  # DEPRECATED
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
    # "DistributedMultiheadDilatedAttention",  # Old implementation
    "RingDilatedAttention",
    "RingMultiheadDilatedAttention",
    # Configuration and utility classes
    "SparsePatternConfig",
    "SparsePatternGenerator",
    "UnfoldRingDilatedAttention",
    "create_adaptive_sparse_attention",
    "create_block_sparse_attention",
    # Factory functions (v0.2.0+)
    "create_dilated_attention",
    "create_multihead_dilated_attention",
]
