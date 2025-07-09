"""
Ring attention implementations.

This module contains various ring attention implementations for distributed
processing of long sequences.
"""

# Import from submodules
from .base import (
    RingDilatedAttentionCorrect,
    RingAttentionWrapper,
    RingDilatedAttentionV3,
    RingDilatedAttentionMemoryEfficient,
    RingDilatedAttentionSDPA,
    RingDilatedAttentionFixedSimple,
)

from .distributed import RingDistributedDilatedAttention

from .hilbert import (
    RingDilatedAttentionHilbertCore,
    RingDilatedAttentionHilbertOptimizedFixed,
    RingDilatedAttentionHilbertProper,
    RingDilatedAttentionHilbertGPUOptimized,
)

from .utils import (
    all_ring_pass,
    split_by_rank,
    RingAttentionFunction,
    StableRingAccumulator,
    memory_efficient_ring_attention,
)

# Aliases for common imports
RingDilatedAttention = RingDilatedAttentionCorrect  # The correct implementation
RingDilatedAttentionProduction = RingDilatedAttentionCorrect  # Production alias

__all__ = [
    # Base implementations
    "RingDilatedAttentionCorrect",
    "RingAttentionWrapper",
    "RingDilatedAttentionV3",
    "RingDilatedAttentionMemoryEfficient",
    "RingDilatedAttentionSDPA",
    "RingDilatedAttentionFixedSimple",
    # Distributed
    "RingDistributedDilatedAttention",
    # Hilbert optimized
    "RingDilatedAttentionHilbertCore",
    "RingDilatedAttentionHilbertOptimizedFixed",
    "RingDilatedAttentionHilbertProper",
    "RingDilatedAttentionHilbertGPUOptimized",
    # Utils
    "all_ring_pass",
    "split_by_rank",
    "RingAttentionFunction",
    "StableRingAccumulator",
    "memory_efficient_ring_attention",
    # Aliases
    "RingDilatedAttention",
    "RingDilatedAttentionProduction",
]
