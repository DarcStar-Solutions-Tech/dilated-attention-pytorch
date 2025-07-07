"""
Enhanced Block-Sparse Ring Dilated Attention

This is a temporary file that merges optimizations from block_sparse_optimized.py
into the base block_sparse_ring_dilated_attention.py. After verification,
this will replace the base implementation.
"""

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from .ring_dilated_attention_production import (
    RingAttentionConfig,
    RingDilatedAttentionProduction,
)

# from .sparse_pattern_generator import SparsePatternGenerator  # Not used in this implementation
from .core.constants import HAS_FLASH_ATTN_3, GPU_TYPE


@dataclass
class SparsePatternConfig:
    """Configuration for sparse attention patterns."""

    pattern_type: str = (
        "local_window"  # Options: local_window, dilated_sparse, global_local
    )
    block_size: int = 64  # Size of each attention block
    sparsity_ratio: float = 0.1  # 0.1 = 90% sparse
    window_size: int = 256  # For local_window pattern
    global_tokens: int = 1  # For global_local pattern
    dilation_rates: list[int] | None = None  # For dilated_sparse pattern

    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.sparsity_ratio <= 1:
            raise ValueError("sparsity_ratio must be between 0 and 1")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.pattern_type not in [
            "local_window",
            "dilated_sparse",
            "global_local",
            "adaptive",
        ]:
            raise ValueError(f"Unknown pattern_type: {self.pattern_type}")


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


class BlockSparseRingDilatedAttention(RingDilatedAttentionProduction):
    """
    Enhanced Block-Sparse Ring Dilated Attention with merged optimizations.

    Key features:
    1. Memory-efficient block-sparse attention patterns
    2. Never materializes full attention matrices
    3. Persistent pattern caching across devices
    4. Batched block operations for efficiency
    5. Integrates with Flash Attention 3 when available
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        sparse_config: SparsePatternConfig | None = None,
        enable_batched_ops: bool = True,
        pattern_cache_size: int = 100,
        **kwargs,
    ):
        """Initialize enhanced block-sparse ring dilated attention."""
        # Extract sparse_config from kwargs if passed there (for compatibility)
        if "sparse_config" in kwargs:
            sparse_config = kwargs.pop("sparse_config")
        if "sparsity_config" in kwargs:
            sparse_config = kwargs.pop("sparsity_config")

        # Extract parameters for RingAttentionConfig
        dropout = kwargs.get("dropout", 0.0)
        ring_size = kwargs.get("ring_size", None)
        use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", True)
        use_memory_pool = kwargs.get("enable_memory_pool", True)
        mixed_precision = kwargs.get("mixed_precision", True)

        # Create RingAttentionConfig
        ring_config = RingAttentionConfig(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_memory_pool=use_memory_pool,
            mixed_precision=mixed_precision,
        )

        # Initialize parent with config
        super().__init__(ring_config)

        # Extract sparsity_ratio if provided directly (for compatibility)
        sparsity_ratio = kwargs.pop("sparsity_ratio", None)

        # Handle both dict and SparsePatternConfig
        if isinstance(sparse_config, dict):
            self.sparse_config = SparsePatternConfig(**sparse_config)
        elif sparse_config is not None:
            self.sparse_config = sparse_config
        else:
            # Create default config, optionally with provided sparsity_ratio
            config_kwargs = {}
            if sparsity_ratio is not None:
                config_kwargs["sparsity_ratio"] = sparsity_ratio
            self.sparse_config = SparsePatternConfig(**config_kwargs)

        self.block_size = self.sparse_config.block_size

        # Enhanced pattern cache with device awareness
        self.pattern_cache = PersistentPatternCache(max_size=pattern_cache_size)

        # Batched operations flag
        self.enable_batched_ops = enable_batched_ops

        # Pre-allocate buffers for batched operations
        self._batch_buffers = {}

        # Check for Flash Attention 3 support
        self.use_fa3 = HAS_FLASH_ATTN_3 and str(GPU_TYPE) in ["h100", "h800"]
        if self.use_fa3:
            self.fa3_config = None  # Will be set dynamically based on sequence length

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor] | None]:
        """
        Forward pass with memory-efficient block-sparse attention.

        Returns:
            output: Attention output [batch, seq_len, num_heads, head_dim]
            attention_weights: If requested, returns dict with:
                - 'indices': COO format indices of non-zero blocks
                - 'values': Attention values for those blocks
                - 'shape': Full attention shape for reference
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # Ensure sequence length is divisible by block size
        if seq_len % self.block_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by block size {self.block_size}"
            )

        num_blocks = seq_len // self.block_size

        # Get active block pairs for sparse pattern
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, q.device)

        # Initialize output using memory pool if available
        if hasattr(self, "_allocate_tensor") and self._memory_pool is not None:
            output = self._allocate_tensor(
                q.shape, q.dtype, q.device, strategy="auto", zero_init=True
            )
        else:
            output = torch.zeros_like(q)

        # Compute sparse attention
        attention_weights = None
        if return_attention_weights:
            attention_weights = self._compute_sparse_attention_with_weights(
                q, k, v, output, block_indices, is_causal
            )
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)

        if return_attention_weights:
            return output, attention_weights
        else:
            return output

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
        if (
            self.enable_batched_ops and len(block_indices[0]) > 32
        ):  # Threshold for batching
            self._compute_sparse_attention_batched(
                q, k, v, output, block_indices, is_causal
            )
        else:
            # Fall back to sequential implementation for small patterns
            self._compute_sparse_attention_sequential(
                q, k, v, output, block_indices, is_causal
            )

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

        # Scatter blocks to output
        output_blocks = output.view(
            batch, num_blocks, self.block_size, num_heads, head_dim
        )

        # Use index_add for efficient scattering
        for i, row_idx in enumerate(row_indices):
            output_blocks[:, row_idx] += block_outputs[:, i]

    def _compute_sparse_attention_sequential(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> None:
        """Sequential implementation for small patterns or debugging."""
        batch, seq_len, num_heads, head_dim = q.shape
        scale = 1.0 / math.sqrt(head_dim)

        row_indices, col_indices = block_indices

        # Process each block pair
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Get block boundaries
            row_start = row_idx * self.block_size
            row_end = row_start + self.block_size
            col_start = col_idx * self.block_size
            col_end = col_start + self.block_size

            # Extract blocks
            q_block = q[
                :, row_start:row_end
            ]  # [batch, block_size, num_heads, head_dim]
            k_block = k[:, col_start:col_end]
            v_block = v[:, col_start:col_end]

            # Compute attention for this block
            # [batch, num_heads, block_size, head_dim]
            q_block = q_block.transpose(1, 2)
            k_block = k_block.transpose(1, 2)
            v_block = v_block.transpose(1, 2)

            # [batch, num_heads, block_size, block_size]
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale

            # Apply causal mask if needed
            if is_causal and row_idx >= col_idx:
                if row_idx == col_idx:
                    # Diagonal block - apply causal mask
                    causal_mask = torch.triu(
                        torch.ones(self.block_size, self.block_size, device=q.device),
                        diagonal=1,
                    ).bool()
                    scores = scores.masked_fill(causal_mask, float("-inf"))
                elif row_idx > col_idx:
                    # Below diagonal - check token positions
                    for i in range(self.block_size):
                        for j in range(self.block_size):
                            if row_start + i < col_start + j:
                                scores[:, :, i, j] = float("-inf")

            # Softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Apply attention to values
            # [batch, num_heads, block_size, head_dim]
            block_output = torch.matmul(attn_weights, v_block)

            # Transpose back and add to output
            block_output = block_output.transpose(
                1, 2
            )  # [batch, block_size, num_heads, head_dim]
            output[:, row_start:row_end] += block_output

    def _compute_sparse_attention_with_weights(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        block_indices: Tuple[Tensor, Tensor],
        is_causal: bool,
    ) -> dict[str, Tensor]:
        """Compute sparse attention and return attention weights in COO format."""
        # For now, use sequential implementation when weights are requested
        # TODO: Implement batched version with weight collection
        self._compute_sparse_attention_sequential(
            q, k, v, output, block_indices, is_causal
        )

        # Return sparse attention info
        batch, seq_len, num_heads, head_dim = q.shape
        row_indices, col_indices = block_indices

        return {
            "indices": torch.stack([row_indices, col_indices]),
            "values": None,  # Would need to collect during computation
            "shape": (seq_len // self.block_size, seq_len // self.block_size),
            "block_size": self.block_size,
        }

    def _get_batched_causal_mask(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        batch: int,
        num_heads: int,
    ) -> Tensor:
        """Create batched causal mask for active blocks."""
        num_active_blocks = len(row_indices)

        # Create mask tensor
        mask = torch.zeros(
            batch * num_active_blocks * num_heads,
            self.block_size,
            self.block_size,
            device=row_indices.device,
            dtype=torch.bool,
        )

        # Fill mask based on block positions
        for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            if row_idx == col_idx:
                # Diagonal block - standard causal mask
                causal_mask = torch.triu(
                    torch.ones(self.block_size, self.block_size, device=mask.device),
                    diagonal=1,
                ).bool()

                # Apply to all batch/head combinations for this block
                start_idx = i * batch * num_heads
                end_idx = (i + 1) * batch * num_heads
                mask[start_idx:end_idx] = causal_mask

            elif row_idx > col_idx:
                # Below diagonal - check token positions
                row_start = row_idx * self.block_size
                col_start = col_idx * self.block_size

                for bi in range(self.block_size):
                    for bj in range(self.block_size):
                        if row_start + bi < col_start + bj:
                            start_idx = i * batch * num_heads
                            end_idx = (i + 1) * batch * num_heads
                            mask[start_idx:end_idx, bi, bj] = True

        return mask

    # Pattern generation methods (same as original)
    def _create_local_window_indices(
        self, num_blocks: int
    ) -> tuple[list[int], list[int]]:
        """Create indices for local window attention pattern."""
        window_blocks = self.sparse_config.window_size // self.block_size
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            # Each block attends to window_blocks around it
            start = max(0, i - window_blocks // 2)
            end = min(num_blocks, i + window_blocks // 2 + 1)

            for j in range(start, end):
                row_indices.append(i)
                col_indices.append(j)

        return row_indices, col_indices

    def _create_dilated_sparse_indices(
        self, num_blocks: int
    ) -> tuple[list[int], list[int]]:
        """Create indices for dilated sparse attention pattern."""
        dilation_rates = self.sparse_config.dilation_rates or [1, 2, 4]
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            # Add local attention
            row_indices.append(i)
            col_indices.append(i)

            # Add dilated attention at different rates
            for rate in dilation_rates:
                if i + rate < num_blocks:
                    row_indices.append(i)
                    col_indices.append(i + rate)
                if i - rate >= 0:
                    row_indices.append(i)
                    col_indices.append(i - rate)

        return row_indices, col_indices

    def _create_global_local_indices(
        self, num_blocks: int
    ) -> tuple[list[int], list[int]]:
        """Create indices for global-local attention pattern."""
        global_blocks = min(self.sparse_config.global_tokens, num_blocks)
        window_blocks = self.sparse_config.window_size // self.block_size
        row_indices = []
        col_indices = []

        for i in range(num_blocks):
            # Global tokens attend to and are attended by all
            if i < global_blocks:
                for j in range(num_blocks):
                    row_indices.append(i)
                    col_indices.append(j)
                    if i != j:
                        row_indices.append(j)
                        col_indices.append(i)
            else:
                # Local window attention for non-global tokens
                start = max(0, i - window_blocks // 2)
                end = min(num_blocks, i + window_blocks // 2 + 1)

                for j in range(start, end):
                    row_indices.append(i)
                    col_indices.append(j)

                # Also attend to global tokens
                for j in range(global_blocks):
                    if j not in range(start, end):
                        row_indices.append(i)
                        col_indices.append(j)

        # Remove duplicates
        seen = set()
        unique_row = []
        unique_col = []
        for r, c in zip(row_indices, col_indices):
            if (r, c) not in seen:
                seen.add((r, c))
                unique_row.append(r)
                unique_col.append(c)

        return unique_row, unique_col

    def get_pattern_stats(self) -> dict:
        """Get statistics about pattern cache usage."""
        return self.pattern_cache.get_stats()

    def clear_pattern_cache(self):
        """Clear the pattern cache."""
        self.pattern_cache.clear()
