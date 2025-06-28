"""
Optimized Block-Sparse Ring Dilated Attention

This implementation addresses the performance bottlenecks identified in profiling:
1. Enhanced pattern caching with device-aware storage
2. Batched block operations to reduce kernel launch overhead
3. Preparation for sparse tensor integration
"""

import math
import threading
from collections import OrderedDict
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


class PersistentPatternCache:
    """Enhanced pattern cache that keeps patterns on device and tracks usage."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: OrderedDict[tuple, Tuple[Tensor, Tensor]] = OrderedDict()
        self.device_cache: Dict[torch.device, Dict[tuple, Tuple[Tensor, Tensor]]] = {}
        self.access_count: Dict[tuple, int] = {}
        self._lock = threading.Lock()

    def get(
        self, key: tuple, device: torch.device, generator_fn=None
    ) -> Tuple[Tensor, Tensor]:
        """Get pattern from cache or generate if not found."""
        with self._lock:
            # Check device cache first
            if device not in self.device_cache:
                self.device_cache[device] = {}

            device_patterns = self.device_cache[device]
            if key in device_patterns:
                self.access_count[key] = self.access_count.get(key, 0) + 1
                return device_patterns[key]

            # Check main cache
            if key in self.cache:
                cpu_row, cpu_col = self.cache[key]
                # Move to device and cache there
                device_row = cpu_row.to(device)
                device_col = cpu_col.to(device)
                device_patterns[key] = (device_row, device_col)

                # Update access count and move to end (LRU)
                self.access_count[key] = self.access_count.get(key, 0) + 1
                self.cache.move_to_end(key)

                return device_row, device_col

            # Generate new pattern if generator provided
            if generator_fn is not None:
                row_idx, col_idx = generator_fn()

                # Store in main cache (on CPU)
                self.cache[key] = (row_idx.cpu(), col_idx.cpu())

                # Store on device
                device_patterns[key] = (row_idx.to(device), col_idx.to(device))

                # Evict oldest if cache is full
                if len(self.cache) > self.max_size:
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    # Remove from all device caches
                    for dev_cache in self.device_cache.values():
                        dev_cache.pop(oldest_key, None)

                return row_idx.to(device), col_idx.to(device)

            raise KeyError(f"Pattern not found and no generator provided: {key}")

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self.cache.clear()
            self.device_cache.clear()
            self.access_count.clear()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_patterns = len(self.cache)
            total_accesses = sum(self.access_count.values())
            hit_rate = total_accesses / max(1, total_accesses + total_patterns)

            return {
                "total_patterns": total_patterns,
                "total_accesses": total_accesses,
                "hit_rate": hit_rate,
                "device_caches": len(self.device_cache),
                "most_accessed": sorted(
                    self.access_count.items(), key=lambda x: x[1], reverse=True
                )[:5],
            }


class BlockSparseOptimized(BlockSparseRingDilatedAttention):
    """
    Optimized Block-Sparse attention with performance improvements.

    Key optimizations:
    1. Persistent pattern caching across devices
    2. Batched block operations
    3. Reduced synchronization points
    4. Preparation for sparse tensor conversion
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: Optional[SparsePatternConfig] = None,
        enable_batched_ops: bool = True,
        cache_size: int = 100,
        **kwargs,
    ):
        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        # Replace standard cache with persistent cache
        self.pattern_cache = PersistentPatternCache(max_size=cache_size)
        self.enable_batched_ops = enable_batched_ops

        # Pre-allocate buffers for batched operations
        self._batch_buffers = {}

    def _get_sparse_block_indices(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Get indices with enhanced caching."""
        cache_key = (
            num_blocks,
            self.sparse_config.pattern_type,
            self.sparse_config.sparsity_ratio,
            self.sparse_config.block_size,
        )

        def generator():
            # Generate pattern based on type
            if self.sparse_config.pattern_type == "local_window":
                row_idx, col_idx = self._create_local_window_indices(num_blocks)
            elif self.sparse_config.pattern_type == "dilated_sparse":
                row_idx, col_idx = self._create_dilated_sparse_indices(num_blocks)
            elif self.sparse_config.pattern_type == "global_local":
                row_idx, col_idx = self._create_global_local_indices(num_blocks)
            else:
                raise ValueError(
                    f"Unknown pattern type: {self.sparse_config.pattern_type}"
                )

            # Convert to tensors
            row_idx = torch.tensor(row_idx, dtype=torch.long)
            col_idx = torch.tensor(col_idx, dtype=torch.long)

            return row_idx, col_idx

        return self.pattern_cache.get(cache_key, device, generator)

    def _compute_sparse_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Compute sparse attention with batched operations."""
        if self.enable_batched_ops:
            self._compute_sparse_attention_batched(
                q, k, v, output, block_indices, is_causal
            )
        else:
            # Fall back to parent implementation
            super()._compute_sparse_attention(q, k, v, output, block_indices, is_causal)

    def _compute_sparse_attention_batched(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Compute sparse attention using batched block operations."""
        batch, seq_len, num_heads, head_dim = q.shape
        num_blocks = seq_len // self.block_size
        scale = 1.0 / math.sqrt(head_dim)

        row_indices, col_indices = block_indices
        num_active_blocks = len(row_indices)

        # Reshape tensors for block access
        q_blocks = q.view(batch, num_blocks, self.block_size, num_heads, head_dim)
        k_blocks = k.view(batch, num_blocks, self.block_size, num_heads, head_dim)
        v_blocks = v.view(batch, num_blocks, self.block_size, num_heads, head_dim)

        # Gather all active blocks at once
        # Shape: [batch, num_active_blocks, block_size, num_heads, head_dim]
        q_active = q_blocks[:, row_indices]
        k_active = k_blocks[:, col_indices]
        v_active = v_blocks[:, col_indices]

        # Reshape for batched matmul
        # [batch * num_active_blocks * num_heads, block_size, head_dim]
        q_active = q_active.permute(0, 1, 3, 2, 4).reshape(
            -1, self.block_size, head_dim
        )
        k_active = k_active.permute(0, 1, 3, 2, 4).reshape(
            -1, self.block_size, head_dim
        )
        v_active = v_active.permute(0, 1, 3, 2, 4).reshape(
            -1, self.block_size, head_dim
        )

        # Batched attention computation
        # [batch * num_active_blocks * num_heads, block_size, block_size]
        scores = torch.bmm(q_active, k_active.transpose(-2, -1)) * scale

        # Apply causal mask if needed (batched)
        if is_causal:
            # Create batched causal mask
            causal_mask = self._get_batched_causal_mask(
                row_indices, col_indices, batch, num_heads
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Batched softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Batched value computation
        # [batch * num_active_blocks * num_heads, block_size, head_dim]
        block_outputs = torch.bmm(attn_weights, v_active)

        # Reshape back and scatter to output
        block_outputs = block_outputs.view(
            batch, num_active_blocks, num_heads, self.block_size, head_dim
        ).permute(0, 1, 3, 2, 4)

        # Scatter blocks back to output
        output_blocks = output.view(
            batch, num_blocks, self.block_size, num_heads, head_dim
        )

        # Accumulate results (handling overlapping blocks)
        for idx, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            output_blocks[:, row_idx] += block_outputs[:, idx]

    def _get_batched_causal_mask(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        batch: int,
        num_heads: int,
    ) -> Tensor:
        """Get batched causal mask for all active blocks."""
        num_active_blocks = len(row_indices)
        device = row_indices.device

        # Create base causal mask for a single block
        base_mask = torch.triu(
            torch.ones(
                self.block_size, self.block_size, device=device, dtype=torch.bool
            ),
            diagonal=1,
        )

        # Expand for all blocks and heads
        mask = base_mask.unsqueeze(0).expand(
            batch * num_active_blocks * num_heads, -1, -1
        )

        # Adjust mask based on block positions
        for idx, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            if col_idx > row_idx:
                # Future block - mask everything
                start_idx = idx * batch * num_heads
                end_idx = (idx + 1) * batch * num_heads
                mask[start_idx:end_idx] = True
            elif col_idx < row_idx:
                # Past block - unmask everything
                start_idx = idx * batch * num_heads
                end_idx = (idx + 1) * batch * num_heads
                mask[start_idx:end_idx] = False

        return mask

    def get_optimization_stats(self) -> dict:
        """Get statistics about optimization effectiveness."""
        stats = {
            "pattern_cache": self.pattern_cache.get_stats(),
            "batched_ops_enabled": self.enable_batched_ops,
            "buffer_count": len(self._batch_buffers),
        }

        # Add parent stats if available
        if hasattr(super(), "get_optimization_stats"):
            parent_stats = super().get_optimization_stats()
            stats.update(parent_stats)

        return stats
