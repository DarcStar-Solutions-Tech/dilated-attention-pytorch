"""
Core components for Dilated Attention PyTorch implementation.

This module provides base classes, configurations, and utilities that are
shared across all dilated attention implementations.
"""

from ..utils.attention_utils import (
    apply_dilated_attention_pattern,
    apply_rotary_embeddings,
    compute_alibi_bias,
    compute_attention_scores,
    compute_rotary_embeddings,
    create_block_diagonal_mask,
    create_dilated_mask,
    merge_attention_heads,
    optimize_attention_computation,
    split_attention_heads,
    standard_attention,
)
from ..utils.validation import ValidationMixin
from .base import BaseDilatedAttention, BaseMultiheadDilatedAttention
from .config import (
    DilatedAttentionConfig,
    DistributedConfig,
    MemoryPoolConfig,
    MultiheadConfig,
    RingAttentionConfig,
    SparseAttentionConfig,
)
from .constants import (  # Hardware detection; Feature detection; Version info
    APEX_VERSION,
    CURRENT_OPTIMAL_SETTINGS,
    DEEPSPEED_VERSION,
    FAIRSCALE_VERSION,
    FLASH_ATTN_VERSION,
    GPU_TYPE,
    HAS_APEX,
    HAS_DEEPSPEED,
    HAS_FAIRSCALE,
    HAS_FLASH_ATTN,
    HAS_FLASH_ATTN_3,
    HAS_SDPA,
    HAS_SDPA_KERNEL,
    HAS_XFORMERS,
    TORCH_VERSION,
    XFORMERS_VERSION,
)
from .factory import (
    create_adaptive_sparse_attention,
    create_block_sparse_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
    register_attention,
    register_multihead_attention,
)
from .memory_pool import UnifiedMemoryPool, get_global_memory_pool, reset_global_memory_pool

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
