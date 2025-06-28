"""
Attention-specific buffer manager for optimized memory allocation.

This module provides specialized buffer types and allocation strategies
for attention mechanisms, building on top of the enhanced memory pool.
"""

import enum
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import torch
from torch import Tensor

from .enhanced_memory_pool import get_enhanced_memory_pool

logger = logging.getLogger("dilated_attention_pytorch.attention_buffer_manager")


class BufferType(enum.Enum):
    """Types of buffers used in attention mechanisms."""

    QUERY = "query"  # Query tensor buffers
    KEY = "key"  # Key tensor buffers
    VALUE = "value"  # Value tensor buffers
    OUTPUT = "output"  # Attention output buffers
    SCORES = "scores"  # Attention scores (QK^T)
    WEIGHTS = "weights"  # Attention weights (softmax(scores))
    TEMP = "temp"  # Temporary buffers
    COMM = "comm"  # Communication buffers (for distributed)
    MASK = "mask"  # Attention masks (causal, padding, etc.)
    CACHE = "cache"  # KV cache for autoregressive models


@dataclass
class BufferConfig:
    """Configuration for a specific buffer type."""

    # Size characteristics
    typical_size_mb: float  # Typical size in MB
    reuse_frequency: str  # "high", "medium", "low"
    lifetime: str  # "ephemeral", "iteration", "persistent"

    # Allocation preferences
    prefer_bucketed: bool = True
    prefer_numa: bool = False
    prefer_pinned: bool = False
    zero_init: bool = True

    # Memory layout
    alignment: Optional[int] = None  # Byte alignment requirement
    contiguous: bool = True  # Requires contiguous memory


# Default configurations for each buffer type
BUFFER_CONFIGS: Dict[BufferType, BufferConfig] = {
    BufferType.QUERY: BufferConfig(
        typical_size_mb=1.0,
        reuse_frequency="high",
        lifetime="iteration",
        prefer_bucketed=True,
        zero_init=False,  # Will be filled immediately
    ),
    BufferType.KEY: BufferConfig(
        typical_size_mb=1.0,
        reuse_frequency="high",
        lifetime="iteration",
        prefer_bucketed=True,
        zero_init=False,
    ),
    BufferType.VALUE: BufferConfig(
        typical_size_mb=1.0,
        reuse_frequency="high",
        lifetime="iteration",
        prefer_bucketed=True,
        zero_init=False,
    ),
    BufferType.OUTPUT: BufferConfig(
        typical_size_mb=1.0,
        reuse_frequency="high",
        lifetime="iteration",
        prefer_bucketed=True,
        zero_init=True,  # Must be zeroed for accumulation
    ),
    BufferType.SCORES: BufferConfig(
        typical_size_mb=4.0,  # seq_len x seq_len can be large
        reuse_frequency="medium",
        lifetime="ephemeral",
        prefer_bucketed=False,  # Too large for buckets
        prefer_numa=True,  # Large allocations benefit from NUMA
        zero_init=False,
    ),
    BufferType.WEIGHTS: BufferConfig(
        typical_size_mb=4.0,
        reuse_frequency="low",
        lifetime="ephemeral",
        prefer_bucketed=False,
        prefer_numa=True,
        zero_init=False,
    ),
    BufferType.TEMP: BufferConfig(
        typical_size_mb=0.5,
        reuse_frequency="high",
        lifetime="ephemeral",
        prefer_bucketed=True,
        zero_init=False,
    ),
    BufferType.COMM: BufferConfig(
        typical_size_mb=2.0,
        reuse_frequency="high",
        lifetime="persistent",
        prefer_bucketed=True,
        prefer_pinned=True,  # Pin for faster CPU-GPU transfer
        zero_init=False,
    ),
    BufferType.MASK: BufferConfig(
        typical_size_mb=0.1,
        reuse_frequency="high",
        lifetime="persistent",
        prefer_bucketed=True,
        zero_init=False,
        alignment=64,  # Align for SIMD operations
    ),
    BufferType.CACHE: BufferConfig(
        typical_size_mb=8.0,
        reuse_frequency="medium",
        lifetime="persistent",
        prefer_bucketed=False,
        prefer_numa=True,
        zero_init=True,
    ),
}


class AttentionBufferManager:
    """
    Manager for attention-specific buffer allocation.

    This class provides optimized allocation strategies for different
    types of buffers used in attention mechanisms, with support for
    buffer reuse, pre-allocation, and type-specific optimization.
    """

    def __init__(
        self,
        enable_reuse: bool = True,
        enable_preallocation: bool = True,
        enable_profiling: bool = False,
        custom_configs: Optional[Dict[BufferType, BufferConfig]] = None,
        pool_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the attention buffer manager.

        Args:
            enable_reuse: Enable buffer reuse across iterations
            enable_preallocation: Pre-allocate common buffer sizes
            enable_profiling: Enable detailed profiling
            custom_configs: Custom buffer configurations
            pool_config: Configuration for underlying memory pool
        """
        self.enable_reuse = enable_reuse
        self.enable_preallocation = enable_preallocation
        self.enable_profiling = enable_profiling

        # Initialize configurations
        self.configs = BUFFER_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)

        # Initialize memory pool
        pool_config = pool_config or {}
        pool_config.setdefault("enable_profiling", enable_profiling)
        self.pool = get_enhanced_memory_pool(**pool_config)

        # Buffer caches for reuse
        self.buffer_cache: Dict[str, List[Tensor]] = defaultdict(list)
        self.cache_hits = 0
        self.cache_misses = 0

        # Pre-allocated buffers
        self.preallocated: Dict[str, Tensor] = {}

        # Statistics
        self.stats = defaultdict(
            lambda: {
                "allocations": 0,
                "deallocations": 0,
                "bytes_allocated": 0,
                "reuse_count": 0,
            }
        )

        logger.info(
            f"AttentionBufferManager initialized: "
            f"reuse={enable_reuse}, prealloc={enable_preallocation}"
        )

    def allocate(
        self,
        buffer_type: BufferType,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        zero_init: Optional[bool] = None,
    ) -> Tensor:
        """
        Allocate a buffer of the specified type.

        Args:
            buffer_type: Type of buffer to allocate
            shape: Buffer shape
            dtype: Data type
            device: Target device
            zero_init: Override zero initialization setting

        Returns:
            Allocated buffer
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config = self.configs[buffer_type]

        # Check for reusable buffer
        if self.enable_reuse:
            cache_key = self._get_cache_key(buffer_type, shape, dtype, device)
            if cache_key in self.buffer_cache and self.buffer_cache[cache_key]:
                buffer = self.buffer_cache[cache_key].pop()
                self.cache_hits += 1
                self.stats[buffer_type]["reuse_count"] += 1

                # Zero-initialize if needed (this may change data_ptr on some systems)
                if zero_init or (zero_init is None and config.zero_init):
                    buffer.zero_()

                return buffer

        # Cache miss - allocate new buffer
        self.cache_misses += 1

        # Determine allocation strategy
        strategy = self._select_strategy(buffer_type, shape, dtype)

        # Allocate through pool
        buffer = self.pool.allocate(shape, dtype, device, strategy)

        # Apply alignment if needed
        if config.alignment and hasattr(buffer, "data_ptr"):
            ptr = buffer.data_ptr()
            if ptr % config.alignment != 0:
                # Reallocate with proper alignment
                aligned_size = buffer.numel() * buffer.element_size()
                aligned_size = (
                    (aligned_size + config.alignment - 1)
                    // config.alignment
                    * config.alignment
                )
                # Note: PyTorch doesn't directly support aligned allocation,
                # so this is more of a best-effort approach
                logger.debug(f"Buffer not aligned to {config.alignment} bytes")

        # Zero-initialize if needed
        if zero_init or (zero_init is None and config.zero_init):
            buffer.zero_()

        # Update statistics
        self.stats[buffer_type]["allocations"] += 1
        self.stats[buffer_type]["bytes_allocated"] += (
            buffer.numel() * buffer.element_size()
        )

        return buffer

    def deallocate(self, buffer: Tensor, buffer_type: BufferType) -> None:
        """
        Return a buffer to the pool or cache.

        Args:
            buffer: Buffer to deallocate
            buffer_type: Type of buffer
        """
        config = self.configs[buffer_type]

        # Add to cache if reuse is enabled and buffer is reusable
        if self.enable_reuse and config.lifetime != "ephemeral":
            shape = tuple(buffer.shape)
            dtype = buffer.dtype
            device = buffer.device
            cache_key = self._get_cache_key(buffer_type, shape, dtype, device)

            # Limit cache size to prevent memory bloat
            max_cache_size = 10 if config.reuse_frequency == "high" else 5
            if len(self.buffer_cache[cache_key]) < max_cache_size:
                # Keep buffer in our cache - don't return to underlying pool
                self.buffer_cache[cache_key].append(buffer)
                self.stats[buffer_type]["deallocations"] += 1
                return

        # Return to pool only if not cached
        self.pool.deallocate(buffer)
        self.stats[buffer_type]["deallocations"] += 1

    def preallocate_buffers(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Pre-allocate common buffer sizes for attention.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_heads: Number of attention heads
            head_dim: Head dimension
            device: Target device
        """
        if not self.enable_preallocation:
            return

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pre-allocate Q, K, V, Output buffers
        qkv_shape = (batch_size, seq_len, num_heads, head_dim)
        for buffer_type in [
            BufferType.QUERY,
            BufferType.KEY,
            BufferType.VALUE,
            BufferType.OUTPUT,
        ]:
            key = f"{buffer_type.value}_{qkv_shape}"
            if key not in self.preallocated:
                self.preallocated[key] = self.allocate(
                    buffer_type, qkv_shape, torch.float32, device
                )
                logger.debug(f"Pre-allocated {buffer_type.value} buffer: {qkv_shape}")

        # Pre-allocate attention scores buffer (only if seq_len is reasonable)
        if seq_len <= 4096:  # Don't pre-allocate huge score matrices
            scores_shape = (batch_size, num_heads, seq_len, seq_len)
            key = f"scores_{scores_shape}"
            if key not in self.preallocated:
                self.preallocated[key] = self.allocate(
                    BufferType.SCORES, scores_shape, torch.float32, device
                )
                logger.debug(f"Pre-allocated scores buffer: {scores_shape}")

    def clear_cache(self, buffer_type: Optional[BufferType] = None) -> None:
        """
        Clear cached buffers.

        Args:
            buffer_type: Specific buffer type to clear, or None for all
        """
        if buffer_type:
            cache_keys = [
                k for k in self.buffer_cache if k.startswith(buffer_type.value)
            ]
            for key in cache_keys:
                buffers = self.buffer_cache.pop(key, [])
                for buffer in buffers:
                    self.pool.deallocate(buffer)
        else:
            # Clear all caches
            for buffers in self.buffer_cache.values():
                for buffer in buffers:
                    self.pool.deallocate(buffer)
            self.buffer_cache.clear()

        logger.info(f"Cleared buffer cache for {buffer_type or 'all types'}")

    def _get_cache_key(
        self,
        buffer_type: BufferType,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> str:
        """Generate cache key for buffer reuse."""
        # Ensure consistent device string representation
        device_str = str(device)
        if device.type == "cuda" and ":" not in device_str:
            device_str = f"cuda:{device.index if device.index is not None else 0}"
        return f"{buffer_type.value}_{shape}_{dtype}_{device_str}"

    def _select_strategy(
        self,
        buffer_type: BufferType,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> str:
        """
        Select allocation strategy based on buffer type and size.

        Args:
            buffer_type: Type of buffer
            shape: Buffer shape
            dtype: Data type

        Returns:
            Strategy name for memory pool
        """
        config = self.configs[buffer_type]

        # Calculate size in MB
        element_size = (
            torch.finfo(dtype).bits // 8
            if dtype.is_floating_point
            else torch.iinfo(dtype).bits // 8
        )
        size_mb = torch.prod(torch.tensor(shape)).item() * element_size / (1024 * 1024)

        # Select strategy based on configuration and size
        if config.prefer_numa and size_mb > 16:
            return "numa_aware"
        elif config.prefer_bucketed and size_mb < 1:
            return "bucketed"
        elif size_mb > config.typical_size_mb * 2:
            # Large allocation relative to typical size
            return "fragment_aware"
        else:
            return "auto"

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer manager statistics."""
        total_stats = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
            "buffer_types": dict(self.stats),
            "cached_buffers": sum(
                len(buffers) for buffers in self.buffer_cache.values()
            ),
            "preallocated_buffers": len(self.preallocated),
        }

        # Add pool statistics
        pool_stats = self.pool.get_stats()
        total_stats["pool_stats"] = pool_stats

        return total_stats

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.clear_cache()
            self.preallocated.clear()
        except Exception:
            pass  # Ignore errors during cleanup


# Convenience function for creating buffer manager
def create_attention_buffer_manager(
    enable_reuse: bool = True,
    enable_preallocation: bool = False,
    enable_profiling: bool = False,
    lightweight: bool = True,
) -> AttentionBufferManager:
    """
    Create an attention buffer manager with sensible defaults.

    Args:
        enable_reuse: Enable buffer reuse
        enable_preallocation: Enable pre-allocation
        enable_profiling: Enable profiling
        lightweight: Use lightweight pool configuration

    Returns:
        Configured AttentionBufferManager
    """
    pool_config = {
        "enable_fragment_aware": not lightweight,
        "enable_bucketed": True,
        "enable_numa": not lightweight,
        "enable_profiling": enable_profiling,
    }

    return AttentionBufferManager(
        enable_reuse=enable_reuse,
        enable_preallocation=enable_preallocation,
        enable_profiling=enable_profiling,
        pool_config=pool_config,
    )
