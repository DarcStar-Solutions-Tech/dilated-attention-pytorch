"""
Utility modules for dilated attention implementations.

This package contains utility functions, helper classes, and tools that support
the main dilated attention implementations but are not core architectural components.
"""

# Validation utilities
# Attention computation utilities
from .attention_utils import (apply_head_specific_masks,
                              compute_position_embeddings,
                              create_4d_causal_mask,
                              get_attention_backend_info,
                              merge_attention_heads,
                              optimize_attention_computation,
                              split_attention_heads)
# Sparse pattern utilities
from .sparse_pattern_utils import (PatternOptimizer, PatternQualityAnalyzer,
                                   PatternType, PatternVisualizer,
                                   SparsePatternGenerator,
                                   analyze_pattern_statistics,
                                   load_sparse_pattern,
                                   optimize_pattern_for_hardware,
                                   save_sparse_pattern)
from .validation import ValidationMixin

__all__ = [
    # Validation
    "ValidationMixin",
    # Attention utilities
    "optimize_attention_computation",
    "create_4d_causal_mask",
    "apply_head_specific_masks",
    "merge_attention_heads",
    "split_attention_heads",
    "compute_position_embeddings",
    "get_attention_backend_info",
    # Sparse pattern utilities
    "PatternType",
    "SparsePatternGenerator",
    "PatternQualityAnalyzer",
    "PatternOptimizer",
    "PatternVisualizer",
    "save_sparse_pattern",
    "load_sparse_pattern",
    "analyze_pattern_statistics",
    "optimize_pattern_for_hardware",
]
