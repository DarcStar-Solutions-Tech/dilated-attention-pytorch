"""
Hilbert Attention kernel implementations.

This module provides optimized implementations of Hilbert-ordered attention
with support for dilated/sparse patterns.
"""

# Import the simplified implementation
from .hilbert_attention import HilbertAttention

# Import unified implementations
from .hilbert_attention_unified import UnifiedHilbertAttention
from .hilbert_attention_unified_optimized import UnifiedHilbertAttentionOptimized
from .hilbert_attention_unified_optimized_enhanced import (
    UnifiedHilbertAttentionOptimizedEnhanced,
)

# Import enhanced implementation with all optimizations
from .hilbert_attention_enhanced import HilbertAttentionEnhanced

# Import utilities
from .cache_manager import BoundedCache

# Try to import Triton-based implementations for backward compatibility
try:
    from .hilbert_attention_core import HilbertAttentionFunction

    TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    TRITON_AVAILABLE = False
    HilbertAttentionFunction = None

# Import the simple PyTorch fallback
from .hilbert_attention_simple import create_hilbert_mapping

__all__ = [
    # Main implementation
    "HilbertAttention",
    # Enhanced implementation with all optimizations
    "HilbertAttentionEnhanced",
    # Unified implementations
    "UnifiedHilbertAttention",
    "UnifiedHilbertAttentionOptimized",
    "UnifiedHilbertAttentionOptimizedEnhanced",
    # Utilities
    "BoundedCache",
    "create_hilbert_mapping",
    # Status flag
    "TRITON_AVAILABLE",
]
