"""
Hilbert Attention kernel implementations.

This module provides optimized implementations of Hilbert-ordered attention
with support for dilated/sparse patterns.
"""

# Import unified implementations
from .hilbert_attention_unified import UnifiedHilbertAttention
from .hilbert_attention_unified_optimized_enhanced import (
    UnifiedHilbertAttentionOptimizedEnhanced,
)

# Import utilities
from .cache_manager import BoundedCache

# Try to import Triton-based implementations for backward compatibility
try:
    from .hilbert_attention_unified import HilbertAttentionFunction

    TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    TRITON_AVAILABLE = False
    HilbertAttentionFunction = None

# Import the Hilbert mapping creation function
# Note: The function is a static method, so we need to access it from the class
create_hilbert_mapping = UnifiedHilbertAttention._create_hilbert_mapping

__all__ = [
    # Unified implementations (in order of increasing optimization)
    "UnifiedHilbertAttention",
    "UnifiedHilbertAttentionOptimizedEnhanced",
    # Utilities
    "BoundedCache",
    "create_hilbert_mapping",
    # Status flag
    "TRITON_AVAILABLE",
]
