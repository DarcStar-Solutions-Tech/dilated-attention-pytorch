"""
Configuration classes for distributed sparse attention.

This module contains configuration dataclasses and enums used by
block-sparse ring distributed attention implementations.
"""

from dataclasses import dataclass
from enum import Enum


# Check for optional enterprise dependencies
try:
    import deepspeed  # noqa: F401

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import apex  # noqa: F401

    HAS_APEX = True
except ImportError:
    HAS_APEX = False


class DistributedSparsePattern(Enum):
    """Types of distributed sparse patterns."""

    HIERARCHICAL = "hierarchical"
    NODE_LOCAL = "node_local"
    BANDWIDTH_AWARE = "bandwidth_aware"
    ADAPTIVE_LOAD_BALANCED = "adaptive_load_balanced"


@dataclass
class DistributedSparseConfig:
    """Configuration for distributed sparse attention patterns.
    
    This configuration controls how sparsity is applied across different
    levels of the distributed system hierarchy (within-node vs cross-node).
    
    Attributes:
        pattern_type: Type of distributed sparse pattern to use
        sparsity_ratio: Overall sparsity ratio (fraction of zeros)
        block_size: Size of attention blocks
        local_sparsity: Sparsity within the same node (higher density)
        global_sparsity: Sparsity for global attention patterns
        inter_node_sparsity: Sparsity for cross-node attention (minimal)
        compression_ratio: Gradient compression ratio for communication
        load_balance_threshold: Threshold for triggering load rebalancing
        adaptive_sparsity_rate: Rate of sparsity adaptation
        enable_async_communication: Enable asynchronous communication
        enable_gradient_compression: Enable gradient compression
        enable_load_balancing: Enable dynamic load balancing
        enable_memory_pool: Enable adaptive memory pooling
        enable_pinned_memory: Enable pinned memory for transfers
        gradient_bucket_size_mb: Size of gradient buckets in MB
        gradient_bucket_count: Max number of gradients per bucket
    """

    pattern_type: DistributedSparsePattern = DistributedSparsePattern.HIERARCHICAL
    sparsity_ratio: float = 0.25
    block_size: int = 128
    local_sparsity: float = 0.4  # Higher density for local attention
    global_sparsity: float = 0.1  # Lower density for global attention
    inter_node_sparsity: float = 0.05  # Minimal cross-node attention
    compression_ratio: float = 0.1  # Gradient compression ratio
    load_balance_threshold: float = 0.15  # Load imbalance threshold
    adaptive_sparsity_rate: float = 0.05  # Rate of sparsity adaptation
    enable_async_communication: bool = True
    enable_gradient_compression: bool = True
    enable_load_balancing: bool = True
    enable_memory_pool: bool = True
    enable_pinned_memory: bool = True
    gradient_bucket_size_mb: float = 25.0
    gradient_bucket_count: int = 32
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate sparsity ratios
        if not 0 <= self.sparsity_ratio <= 1:
            raise ValueError(f"sparsity_ratio must be in [0, 1], got {self.sparsity_ratio}")
        if not 0 <= self.local_sparsity <= 1:
            raise ValueError(f"local_sparsity must be in [0, 1], got {self.local_sparsity}")
        if not 0 <= self.global_sparsity <= 1:
            raise ValueError(f"global_sparsity must be in [0, 1], got {self.global_sparsity}")
        if not 0 <= self.inter_node_sparsity <= 1:
            raise ValueError(f"inter_node_sparsity must be in [0, 1], got {self.inter_node_sparsity}")
        
        # Validate compression ratio
        if not 0 < self.compression_ratio <= 1:
            raise ValueError(f"compression_ratio must be in (0, 1], got {self.compression_ratio}")
        
        # Validate block size
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(f"block_size must be a positive power of 2, got {self.block_size}")
            
        # Logical consistency checks
        if self.inter_node_sparsity > self.global_sparsity:
            raise ValueError(
                "inter_node_sparsity should not exceed global_sparsity for efficiency"
            )


__all__ = [
    "DistributedSparsePattern",
    "DistributedSparseConfig", 
    "HAS_DEEPSPEED",
    "HAS_APEX",
]