"""
Dilated Attention PyTorch Implementation

Unofficial PyTorch implementation of DilatedAttention from LongNet,
including Ring Attention for O(n) memory scaling.
"""

__version__ = "0.2.0"


# Block-Sparse Ring Attention implementations
from .sparse.block_sparse_ring_attention import (
    BlockSparseRingAttention,
    BlockSparseRingDilatedAttention,  # Backward compatibility alias
    SparsePatternConfig,
)
from .sparse.block_sparse_ring_distributed_dilated_attention import (
    BlockSparseRingDistributedDilatedAttention,
    DistributedSparseConfig,
    DistributedSparsePattern,
)
from .sparse.block_sparse_ring_multihead_dilated_attention import (
    BlockSparseRingMultiheadDilatedAttention,
)

# Optimized Block-Sparse implementations
# from .block_sparse_optimized import BlockSparseOptimized  # Merged into BlockSparseRingDilatedAttention
# from .block_sparse_torch_sparse import BlockSparseTorchSparse  # Removed - provided no benefit over base implementation
# Hierarchical variant removed due to poor memory efficiency
# Use create_block_sparse_attention with pattern_type='dilated_sparse' instead
from .sparse.block_sparse_adaptive import (
    BlockSparseAdaptive,
    AdaptiveConfig,
    ImportanceScorer,
    AdaptiveSparsityTrainer,
)

# Factory functions for easy creation
from .core import (
    create_dilated_attention,
    create_multihead_dilated_attention,
)

# Block-sparse factory
from .sparse.block_sparse_factory import (
    create_block_sparse_attention,
    get_block_sparse_preset,
    create_adaptive_block_sparse,
    create_multihead_block_sparse,
)
from .base.dilated_attention import DilatedAttention
from .base.improved_dilated_attention import ImprovedDilatedAttention
from .base.improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention
from .models.long_net import LongNet
from .base.multihead_dilated_attention import MultiheadDilatedAttention

# [Removed: Hybrid implementation deprecated due to poor performance]
# [Removed: RingMultiheadDilatedAttentionHybrid deprecated]

# [Removed: RingDilatedAttentionProduction - not actually ring attention]
# See docs/reports/ring-production-not-ring-attention-2025-07-08-0327-UTC.md

# Ring Attention implementations with O(n/k) memory scaling
from .ring import (
    StandardRingAttention,
    DistributedRingAttention,
    HilbertRingAttention,
    BlockSparseRingAttention as RingBlockSparseAttention,  # Avoid name conflict
    create_ring_attention,
    RingAttentionConfig,
)

# Additional ring implementations for advanced users
from .ring.base import (
    RingDilatedAttentionCorrect,  # Reference implementation that splits before projection
    RingDilatedAttentionSDPA,  # Uses PyTorch's scaled_dot_product_attention
)

# GPU-optimized Ring Hilbert Attention (legacy name for compatibility)
from .ring.hilbert.ring_dilated_attention_hilbert_gpu_optimized import (
    RingDilatedAttentionHilbertGPUOptimized,
)

# Enterprise distributed attention (formerly RingDistributedDilatedAttention)
from .ring.distributed import EnterpriseDistributedDilatedAttention

from .models.transformer import (
    DilatedTransformerDecoderLayer,
    DilatedTransformerEncoderLayer,
)
from .utils.sparse_pattern_utils import (
    PatternConfig,
    PatternOptimizer,
    PatternQualityAnalyzer,
    PatternType,
    SparsePatternGenerator,
)

# Dynamic segment sizing implementations
from .dynamic_dilated_attention import (
    DynamicDilatedAttention,
    DynamicMultiheadDilatedAttention,
    create_dynamic_dilated_attention,
)
from .utils.dynamic_segment_selector import (
    DynamicSegmentSelector,
    SegmentSelectionConfig,
)

# [Removed: RingDilatedAttention alias - implementation was not actually ring attention]


# Note: Optimizations have been integrated into the main block-sparse implementations

__all__ = [
    # Block-Sparse implementations
    "BlockSparseRingAttention",
    "BlockSparseRingDilatedAttention",  # Backward compatibility alias
    "BlockSparseRingDistributedDilatedAttention",
    "BlockSparseRingMultiheadDilatedAttention",
    # "BlockSparseOptimized",  # Merged into BlockSparseRingDilatedAttention
    # "BlockSparseTorchSparse",  # Removed - provided no benefit over base implementation
    # "BlockSparseHierarchical",  # Removed - poor memory efficiency
    # "HierarchicalConfig",  # Removed with hierarchical implementation
    # "create_hierarchical_attention",  # Removed with hierarchical implementation
    # "get_hierarchical_presets",  # Removed with hierarchical implementation
    "BlockSparseAdaptive",
    "AdaptiveConfig",
    "ImportanceScorer",
    "AdaptiveSparsityTrainer",
    "create_adaptive_block_sparse",
    # Ring Attention implementations (O(n/k) memory scaling)
    "StandardRingAttention",
    "DistributedRingAttention",
    "HilbertRingAttention",
    "RingBlockSparseAttention",
    "create_ring_attention",
    "RingAttentionConfig",
    # Additional ring implementations for advanced users
    "RingDilatedAttentionCorrect",
    "RingDilatedAttentionSDPA",
    # GPU-optimized Ring Hilbert Attention (legacy name)
    "RingDilatedAttentionHilbertGPUOptimized",
    # Enterprise distributed attention
    "EnterpriseDistributedDilatedAttention",
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
    # Block-sparse factory functions
    "create_block_sparse_attention",
    "get_block_sparse_preset",
    "create_adaptive_block_sparse",
    "create_multihead_block_sparse",
    # Factory functions (v0.2.0+)
    "create_dilated_attention",
    "create_multihead_dilated_attention",
    # Dynamic segment sizing (v0.3.0+)
    "DynamicDilatedAttention",
    "DynamicMultiheadDilatedAttention",
    "create_dynamic_dilated_attention",
    "DynamicSegmentSelector",
    "SegmentSelectionConfig",
]

# Flash optimizations are now integrated into V2Collective
HAS_FLASH_RING = True  # Flash is integrated into V2Collective
