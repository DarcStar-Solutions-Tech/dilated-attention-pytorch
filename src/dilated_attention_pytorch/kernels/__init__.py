"""
Production-ready kernel implementations for Dilated Attention with full gradient support.

This module contains optimized Triton kernels for Hilbert-ordered attention patterns.
All implementations support backward pass for training.
"""

# Import the unified core implementation with full gradient support
from .hilbert_attention_core import HilbertAttentionCore, HilbertAttentionFunction

# Import the wrapper that provides the standard q,k,v interface
from .hilbert_attention_triton_wrapper import (
    HilbertAttentionTritonWrapper,
    HilbertAttentionTritonFixed,
)

__all__ = [
    # Core implementation
    "HilbertAttentionCore",
    "HilbertAttentionFunction",
    # Wrapper for q,k,v interface
    "HilbertAttentionTritonWrapper",
    "HilbertAttentionTritonFixed",
]
