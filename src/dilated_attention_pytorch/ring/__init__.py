"""
Ring attention implementations.

This module contains various ring attention implementations for distributed
processing of long sequences.

New Standardized API (v0.3.0):
- Use create_ring_attention() factory function for new implementations
- See MIGRATION_GUIDE.md for migrating from deprecated implementations
"""

# Import standardized implementations
from .standard_ring_attention import StandardRingAttention
from .hilbert_ring_attention import HilbertRingAttention
from .distributed_ring_attention import DistributedRingAttention
from .block_sparse_ring_attention import BlockSparseRingAttention

# Import configuration and factory
from .base.ring_config import RingAttentionConfig, get_preset_config
from .factory import (
    create_ring_attention,
    create_ring_attention_from_preset,
    get_available_implementations,
    validate_ring_configuration,
)
from .utils import (
    all_ring_pass,
    split_by_rank,
    RingAttentionFunction,
    StableRingAccumulator,
    memory_efficient_ring_attention,
)

# Import from legacy submodules (deprecated - will be removed in v0.5.0)
# These are imported but not exported in __all__ to discourage use
_legacy_imports = {}
try:
    from .base import (
        RingDilatedAttentionCorrect,
        RingAttentionWrapper,
        RingDilatedAttentionV3,
        RingDilatedAttentionMemoryEfficient,
        RingDilatedAttentionSDPA,
        RingDilatedAttentionFixedSimple,
    )

    _legacy_imports.update(
        {
            "RingDilatedAttentionCorrect": RingDilatedAttentionCorrect,
            "RingAttentionWrapper": RingAttentionWrapper,
            "RingDilatedAttentionV3": RingDilatedAttentionV3,
            "RingDilatedAttentionMemoryEfficient": RingDilatedAttentionMemoryEfficient,
            "RingDilatedAttentionSDPA": RingDilatedAttentionSDPA,
            "RingDilatedAttentionFixedSimple": RingDilatedAttentionFixedSimple,
        }
    )
except ImportError:
    # Legacy implementations may not exist
    pass

try:
    from .distributed import RingDistributedDilatedAttention as LegacyRingDistributed

    _legacy_imports["LegacyRingDistributed"] = LegacyRingDistributed
except ImportError:
    LegacyRingDistributed = None

try:
    from .hilbert import (
        RingDilatedAttentionHilbertCore,
        RingDilatedAttentionHilbertOptimizedFixed,
        RingDilatedAttentionHilbertProper,
        RingDilatedAttentionHilbertGPUOptimized,
    )

    _legacy_imports.update(
        {
            "RingDilatedAttentionHilbertCore": RingDilatedAttentionHilbertCore,
            "RingDilatedAttentionHilbertOptimizedFixed": RingDilatedAttentionHilbertOptimizedFixed,
            "RingDilatedAttentionHilbertProper": RingDilatedAttentionHilbertProper,
            "RingDilatedAttentionHilbertGPUOptimized": RingDilatedAttentionHilbertGPUOptimized,
        }
    )
except ImportError:
    # Legacy implementations may not exist
    pass

# Aliases for backward compatibility
RingDilatedAttention = StandardRingAttention  # The new standard implementation
RingDilatedAttentionProduction = StandardRingAttention  # Production alias

__all__ = [
    # New standardized API
    "StandardRingAttention",
    "HilbertRingAttention",
    "DistributedRingAttention",
    "BlockSparseRingAttention",
    "RingAttentionConfig",
    "get_preset_config",
    "create_ring_attention",
    "create_ring_attention_from_preset",
    "get_available_implementations",
    "validate_ring_configuration",
    # Utils (still supported)
    "all_ring_pass",
    "split_by_rank",
    "RingAttentionFunction",
    "StableRingAccumulator",
    "memory_efficient_ring_attention",
    # Aliases
    "RingDilatedAttention",
    "RingDilatedAttentionProduction",
]
