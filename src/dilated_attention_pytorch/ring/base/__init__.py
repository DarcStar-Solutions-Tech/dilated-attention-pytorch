"""
Base ring attention implementations.
"""

from .ring_dilated_attention_correct import (
    RingDilatedAttentionCorrect,
    RingAttentionWrapper,
)
from .ring_dilated_attention_v3 import RingDilatedAttentionV3
from .ring_dilated_attention_memory_efficient import RingDilatedAttentionMemoryEfficient
from .ring_dilated_attention_sdpa import RingDilatedAttentionSDPA
from .ring_dilated_attention_fixed_simple import RingDilatedAttentionFixedSimple

# New standardized implementations
from .base_ring_attention import BaseRingAttention, RingAttentionState
from .ring_communication_mixin import RingCommunicationMixin
from .ring_config import RingAttentionConfig, create_ring_config, get_preset_config

__all__ = [
    # Legacy implementations
    "RingDilatedAttentionCorrect",
    "RingAttentionWrapper",
    "RingDilatedAttentionV3",
    "RingDilatedAttentionMemoryEfficient",
    "RingDilatedAttentionSDPA",
    "RingDilatedAttentionFixedSimple",
    # New standardized implementations
    "BaseRingAttention",
    "RingAttentionState",
    "RingCommunicationMixin",
    "RingAttentionConfig",
    "create_ring_config",
    "get_preset_config",
]
