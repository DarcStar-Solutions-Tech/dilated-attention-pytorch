"""
Utility modules for dilated attention implementations.

This package contains utility functions, helper classes, and tools that support
the main dilated attention implementations but are not core architectural components.
"""

# Validation utilities
from .validation import ValidationMixin

# Attention computation utilities
from .attention_utils import (
    optimize_attention_computation,
    create_4d_causal_mask,
    apply_head_specific_masks,
    merge_attention_heads,
    split_attention_heads,
    compute_position_embeddings,
    get_attention_backend_info,
)

# Sparse pattern utilities
from .sparse_pattern_utils import (
    PatternType,
    SparsePatternGenerator,
    PatternQualityAnalyzer,
    PatternOptimizer,
    PatternVisualizer,
    save_sparse_pattern,
    load_sparse_pattern,
    analyze_pattern_statistics,
    optimize_pattern_for_hardware,
)

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