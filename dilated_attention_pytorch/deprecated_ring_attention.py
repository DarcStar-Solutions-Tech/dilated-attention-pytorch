"""
Deprecated Ring Attention implementations.

These implementations have a fundamental flaw: they divide queries across devices,
which prevents achieving the theoretical memory savings of Ring Attention.

Users should migrate to the corrected implementations:
- RingDilatedAttentionV2
- Use create_multihead_dilated_attention("ring") factory method
"""

import warnings
from typing import Optional
import torch
import torch.nn as nn


class DeprecationMixin:
    """Mixin to add deprecation warnings to broken implementations."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            f"{self.__class__.__name__} is deprecated due to incorrect implementation. "
            "This version divides queries across devices, preventing proper memory savings. "
            "Please use RingDilatedAttentionV2 or create_multihead_dilated_attention('ring') instead. "
            "This implementation will be removed in v0.3.0.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


def deprecated_forward_wrapper(original_forward):
    """Wrapper to add runtime warning on forward pass."""
    def wrapped_forward(self, *args, **kwargs):
        if not hasattr(self, '_deprecation_warning_shown'):
            warnings.warn(
                f"Using deprecated {self.__class__.__name__}.forward(). "
                "This implementation incorrectly divides queries and does not achieve "
                "the memory savings of true Ring Attention. Please migrate to the corrected version.",
                RuntimeWarning,
                stacklevel=2
            )
            self._deprecation_warning_shown = True
        return original_forward(self, *args, **kwargs)
    return wrapped_forward


def create_migration_guide():
    """Return migration guide as a string."""
    return """
    Migration Guide for Ring Attention
    ==================================
    
    The original Ring Attention implementations had a fundamental flaw: they divided
    queries across devices, which prevented achieving O(n/ring_size) memory scaling.
    
    Old (Broken) -> New (Correct) Migration:
    
    1. RingDilatedAttention -> RingDilatedAttentionV2
       ```python
       # Old (broken)
       from dilated_attention_pytorch import RingDilatedAttention
       attn = RingDilatedAttention(segment_lengths, dilation_rates)
       
       # New (correct)
       from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
       attn = RingDilatedAttentionV2(segment_lengths, dilation_rates, ring_size=4)
       ```
    
    2. RingMultiheadDilatedAttention -> create_multihead_dilated_attention("ring")
       ```python
       # Old (broken)
       from dilated_attention_pytorch import RingMultiheadDilatedAttention
       attn = RingMultiheadDilatedAttention(embed_dim, num_heads, ...)
       
       # New (correct) - using factory pattern
       from dilated_attention_pytorch import create_multihead_dilated_attention
       attn = create_multihead_dilated_attention(
           "ring",
           embed_dim=embed_dim,
           num_heads=num_heads,
           segment_lengths=segment_lengths,
           dilation_rates=dilation_rates,
           ring_size=4
       )
       ```
    
    3. BlockSparseRingDilatedAttention -> Will be updated in v0.3.0
       For now, use the standard block-sparse attention without ring.
    
    Key Differences:
    - Correct implementations keep full queries on each device
    - Only K/V are chunked and rotated
    - Memory scales as O(n/ring_size) as expected
    - Uses online softmax for proper normalization
    
    For more information, see docs/guides/ring-attention-migration.md
    """