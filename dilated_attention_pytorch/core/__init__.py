"""
Core components for Dilated Attention PyTorch implementation.

This module provides base classes, configurations, and utilities that are
shared across all dilated attention implementations.
"""

from .base import BaseDilatedAttention, BaseMultiheadDilatedAttention
from .config import (
    DilatedAttentionConfig,
    MultiheadConfig,
    RingAttentionConfig,
    SparseAttentionConfig,
    DistributedConfig,
    MemoryPoolConfig,
)
from ..utils.validation import ValidationMixin
from .constants import (
    # Feature detection
    HAS_SDPA,
    HAS_SDPA_KERNEL,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_3,
    HAS_XFORMERS,
    HAS_DEEPSPEED,
    HAS_FAIRSCALE,
    HAS_APEX,
    # Version info
    TORCH_VERSION,
    FLASH_ATTN_VERSION,
    XFORMERS_VERSION,
    DEEPSPEED_VERSION,
    FAIRSCALE_VERSION,
    APEX_VERSION,
    # Hardware detection
    GPU_TYPE,
    CURRENT_OPTIMAL_SETTINGS,
)
from .memory_pool import (
    UnifiedMemoryPool,
    get_global_memory_pool,
    reset_global_memory_pool,
)
from ..utils.attention_utils import (
    compute_attention_scores,
    apply_dilated_attention_pattern,
    create_dilated_mask,
    create_block_diagonal_mask,
    optimize_attention_computation,
    standard_attention,
    compute_alibi_bias,
    compute_rotary_embeddings,
    apply_rotary_embeddings,
    merge_attention_heads,
    split_attention_heads,
)
from .factory import (
    register_attention,
    register_multihead_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
    create_block_sparse_attention,
    create_adaptive_sparse_attention,
)

__all__ = [
    # Base classes
    "BaseDilatedAttention",
    "BaseMultiheadDilatedAttention",
    # Configurations
    "DilatedAttentionConfig",
    "MultiheadConfig",
    "RingAttentionConfig",
    "SparseAttentionConfig",
    "DistributedConfig",
    "MemoryPoolConfig",
    # Mixins
    "ValidationMixin",
    # Feature detection
    "HAS_SDPA",
    "HAS_SDPA_KERNEL",
    "HAS_FLASH_ATTN",
    "HAS_FLASH_ATTN_3",
    "HAS_XFORMERS",
    "HAS_DEEPSPEED",
    "HAS_FAIRSCALE",
    "HAS_APEX",
    # Version info
    "TORCH_VERSION",
    "FLASH_ATTN_VERSION",
    "XFORMERS_VERSION",
    "DEEPSPEED_VERSION",
    "FAIRSCALE_VERSION",
    "APEX_VERSION",
    # Hardware detection
    "GPU_TYPE",
    "CURRENT_OPTIMAL_SETTINGS",
    # Memory pool
    "UnifiedMemoryPool",
    "get_global_memory_pool",
    "reset_global_memory_pool",
    # Attention utilities
    "compute_attention_scores",
    "apply_dilated_attention_pattern",
    "create_dilated_mask",
    "create_block_diagonal_mask",
    "optimize_attention_computation",
    "standard_attention",
    "compute_alibi_bias",
    "compute_rotary_embeddings",
    "apply_rotary_embeddings",
    "merge_attention_heads",
    "split_attention_heads",
    # Factory functions
    "register_attention",
    "register_multihead_attention",
    "create_dilated_attention",
    "create_multihead_dilated_attention",
    "create_block_sparse_attention",
    "create_adaptive_sparse_attention",
]
