"""
Production-ready kernel implementations for Dilated Attention with full gradient support.

This module contains optimized Triton kernels for Hilbert-ordered attention patterns.
All implementations support backward pass for training.
"""

# Try to import Triton-based implementations
try:
    # Import the unified core implementation with full gradient support
    from .hilbert_attention_core import HilbertAttentionCore, HilbertAttentionFunction

    # Import the wrapper that provides the standard q,k,v interface
    from .hilbert_attention_triton_wrapper import (
        HilbertAttentionTritonWrapper,
        HilbertAttentionTritonFixed,
    )

    TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    # Triton not available or compilation issues
    TRITON_AVAILABLE = False
    HilbertAttentionFunction = None
    HilbertAttentionTritonWrapper = None
    HilbertAttentionTritonFixed = None

# Always import the simplified PyTorch version
from .hilbert_attention_simple import (
    HilbertAttentionSimple,
    create_hilbert_mapping,
)

# Use simple version as fallback for core if Triton fails
if not TRITON_AVAILABLE:
    HilbertAttentionCore = HilbertAttentionSimple

__all__ = [
    # Core implementation
    "HilbertAttentionCore",
    "HilbertAttentionFunction",
    "HilbertAttentionSimple",
    "create_hilbert_mapping",
    # Wrapper for q,k,v interface
    "HilbertAttentionTritonWrapper",
    "HilbertAttentionTritonFixed",
    # Status flag
    "TRITON_AVAILABLE",
]
