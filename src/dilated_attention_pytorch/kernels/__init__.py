"""
Kernel implementations for Dilated Attention.
"""

# Import the wrapper that provides the standard interface
from .hilbert_attention_triton_wrapper import HilbertAttentionTritonFixed

__all__ = ["HilbertAttentionTritonFixed"]
