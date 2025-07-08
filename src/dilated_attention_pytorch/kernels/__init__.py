"""
Kernel implementations for Dilated Attention.
"""

# Import the unified core implementation
from .hilbert_attention_core import HilbertAttentionCore

# Import the wrapper that provides the standard interface
from .hilbert_attention_triton_wrapper import HilbertAttentionTritonFixed

# Maintain backward compatibility
HilbertAttentionTritonFixedOptimized = HilbertAttentionCore
HilbertAttentionOptimized = HilbertAttentionCore
HilbertAttentionTritonSimple = HilbertAttentionCore
HilbertAttentionTritonOptimized = HilbertAttentionCore

__all__ = [
    "HilbertAttentionCore",
    "HilbertAttentionTritonFixed",
    "HilbertAttentionTritonFixedOptimized",
    "HilbertAttentionOptimized",
    "HilbertAttentionTritonSimple",
    "HilbertAttentionTritonOptimized",
]
