"""
Distributed attention implementations.

NOTE: RingDistributedDilatedAttention is deprecated and does NOT implement
Ring Attention. Use EnterpriseDistributedDilatedAttention instead.
"""

from .ring_distributed_dilated_attention import (
    EnterpriseDistributedDilatedAttention,
    RingDistributedDilatedAttention,  # Deprecated alias
)

__all__ = [
    "EnterpriseDistributedDilatedAttention",
    "RingDistributedDilatedAttention",  # Deprecated, kept for compatibility
]
