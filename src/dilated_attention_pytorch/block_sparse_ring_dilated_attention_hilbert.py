"""
Block-Sparse Ring Dilated Attention with Hilbert Space-Filling Curve Optimization.

This implementation combines block-sparse attention patterns with Hilbert curve
ordering for improved cache locality and memory access patterns.
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Dict, List

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from .utils.hilbert_curve import (
    generate_hilbert_indices,
    generate_hilbert_indices_rectangular,
)


class BlockSparseRingDilatedAttentionHilbert(BlockSparseRingDilatedAttention):
    """
    Block-Sparse Ring Dilated Attention with Hilbert curve optimization.

    This implementation reorders blocks using Hilbert space-filling curves
    to improve cache locality during attention computation. The Hilbert
    ordering ensures that spatially adjacent blocks remain close in memory,
    reducing cache misses and improving GPU memory bandwidth utilization.

    Key improvements over standard block-sparse:
    1. Better cache locality through Hilbert ordering
    2. Reduced random memory access patterns
    3. Improved GPU memory bandwidth utilization
    4. Optional block-level Hilbert ordering for very long sequences
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: SparsePatternConfig,
        use_hilbert: bool = True,
        hilbert_block_level: bool = True,
        hilbert_within_blocks: bool = False,
        cache_hilbert_indices: bool = True,
        **kwargs,
    ):
        """
        Initialize Hilbert-optimized block-sparse attention.

        Args:
            segment_lengths: Segment lengths for dilated attention
            dilation_rates: Dilation rates for each segment
            sparse_config: Sparse pattern configuration
            use_hilbert: Whether to use Hilbert ordering
            hilbert_block_level: Apply Hilbert ordering at block level
            hilbert_within_blocks: Apply Hilbert ordering within blocks
            cache_hilbert_indices: Cache Hilbert mappings
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        self.use_hilbert = use_hilbert
        self.hilbert_block_level = hilbert_block_level
        self.hilbert_within_blocks = hilbert_within_blocks
        self.cache_hilbert_indices = cache_hilbert_indices

        # Caches for Hilbert indices
        if cache_hilbert_indices:
            self._hilbert_block_cache: Dict[Tuple[int, int], Tensor] = {}
            self._inverse_hilbert_block_cache: Dict[Tuple[int, int], Tensor] = {}
            self._hilbert_element_cache: Dict[int, Tensor] = {}
            self._inverse_hilbert_element_cache: Dict[int, Tensor] = {}

    def _compute_hilbert_block_indices(
        self, num_blocks_h: int, num_blocks_w: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute Hilbert ordering for blocks."""
        cache_key = (num_blocks_h, num_blocks_w)

        # Check cache
        if self.cache_hilbert_indices and cache_key in self._hilbert_block_cache:
            return self._hilbert_block_cache[
                cache_key
            ], self._inverse_hilbert_block_cache[cache_key]

        # Compute Hilbert indices for 2D block grid
        if num_blocks_h == num_blocks_w and (num_blocks_h & (num_blocks_h - 1)) == 0:
            # Square power-of-2 grid - use standard Hilbert
            n_levels = int(math.log2(num_blocks_h))
            indices = generate_hilbert_indices(n_levels)
            hilbert_indices = torch.tensor(indices, dtype=torch.long)
        else:
            # Non-square or non-power-of-2 - use rectangular variant
            indices = generate_hilbert_indices_rectangular(num_blocks_w, num_blocks_h)
            hilbert_indices = torch.tensor(indices, dtype=torch.long)

        # Compute inverse mapping
        inverse_indices = torch.zeros_like(hilbert_indices)
        inverse_indices[hilbert_indices] = torch.arange(len(hilbert_indices))

        # Cache if enabled
        if self.cache_hilbert_indices:
            self._hilbert_block_cache[cache_key] = hilbert_indices
            self._inverse_hilbert_block_cache[cache_key] = inverse_indices

        return hilbert_indices, inverse_indices

    def _apply_hilbert_ordering_to_blocks(
        self, row_indices: Tensor, col_indices: Tensor, num_blocks: int
    ) -> Tuple[Tensor, Tensor]:
        """Apply Hilbert ordering to block indices."""
        if not self.use_hilbert or not self.hilbert_block_level:
            return row_indices, col_indices

        # Convert to 2D grid coordinates
        grid_size = int(math.ceil(math.sqrt(num_blocks)))

        # Get Hilbert ordering
        hilbert_indices, _ = self._compute_hilbert_block_indices(grid_size, grid_size)
        hilbert_indices = hilbert_indices.to(row_indices.device)

        # Create mapping from linear to Hilbert order
        # We need to map the (row, col) pairs through Hilbert ordering
        linear_indices = row_indices * grid_size + col_indices

        # Find position in Hilbert curve
        hilbert_positions = torch.zeros_like(linear_indices)
        for i, linear_idx in enumerate(linear_indices):
            # Find where this linear index appears in Hilbert order
            hilbert_pos = (hilbert_indices == linear_idx).nonzero(as_tuple=True)[0]
            if len(hilbert_pos) > 0:
                hilbert_positions[i] = hilbert_pos[0]
            else:
                hilbert_positions[i] = i  # Fallback to original order

        # Sort by Hilbert position
        sorted_indices = hilbert_positions.argsort()

        return row_indices[sorted_indices], col_indices[sorted_indices]

    def _compute_hilbert_element_indices(self, size: int) -> Tuple[Tensor, Tensor]:
        """Compute Hilbert ordering for elements within blocks."""
        if self.cache_hilbert_indices and size in self._hilbert_element_cache:
            return self._hilbert_element_cache[
                size
            ], self._inverse_hilbert_element_cache[size]

        # Find suitable 2D dimensions
        grid_size = int(math.ceil(math.sqrt(size)))
        if grid_size * grid_size > size:
            # Use rectangular variant
            indices = generate_hilbert_indices_rectangular(
                grid_size, (size + grid_size - 1) // grid_size
            )
            indices = [i for i in indices if i < size]
        else:
            # Square grid
            n_levels = int(math.ceil(math.log2(grid_size)))
            indices = generate_hilbert_indices(n_levels)[:size]

        hilbert_indices = torch.tensor(indices, dtype=torch.long)

        # Compute inverse
        inverse_indices = torch.zeros(size, dtype=torch.long)
        for i, idx in enumerate(indices):
            if idx < size:
                inverse_indices[idx] = i

        # Cache
        if self.cache_hilbert_indices:
            self._hilbert_element_cache[size] = hilbert_indices
            self._inverse_hilbert_element_cache[size] = inverse_indices

        return hilbert_indices, inverse_indices

    def _reorder_tensor_hilbert(
        self, tensor: Tensor, block_size: int, inverse: bool = False
    ) -> Tensor:
        """Reorder tensor elements within blocks using Hilbert curve."""
        if not self.use_hilbert or not self.hilbert_within_blocks:
            return tensor

        batch, seq_len, num_heads, head_dim = tensor.shape

        # Get Hilbert indices for block size
        if inverse:
            _, indices = self._compute_hilbert_element_indices(block_size)
        else:
            indices, _ = self._compute_hilbert_element_indices(block_size)

        # Reshape to blocks
        num_blocks = seq_len // block_size
        tensor_blocks = tensor.view(batch, num_blocks, block_size, num_heads, head_dim)

        # Apply Hilbert ordering within each block
        indices = indices.to(tensor.device)
        tensor_blocks = tensor_blocks[:, :, indices]

        # Reshape back
        return tensor_blocks.view(batch, seq_len, num_heads, head_dim)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Tensor | Tuple[Tensor, dict]:
        """
        Forward pass with Hilbert-optimized block-sparse attention.

        The Hilbert optimization works at two levels:
        1. Block-level: Reorders block computation order for better cache locality
        2. Element-level: Optionally reorders elements within blocks
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # Apply element-level Hilbert ordering if enabled
        if self.hilbert_within_blocks:
            q = self._reorder_tensor_hilbert(q, self.block_size, inverse=False)
            k = self._reorder_tensor_hilbert(k, self.block_size, inverse=False)
            v = self._reorder_tensor_hilbert(v, self.block_size, inverse=False)

        # Get sparse block indices using parent method
        num_blocks = seq_len // self.block_size
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, q.device)

        # Apply block-level Hilbert ordering
        row_indices, col_indices = block_indices
        row_indices, col_indices = self._apply_hilbert_ordering_to_blocks(
            row_indices, col_indices, num_blocks
        )
        block_indices = (row_indices, col_indices)

        # Initialize output
        output = torch.zeros_like(q)

        # Compute attention using parent class methods
        if return_attention_weights:
            attention_info = self._compute_sparse_attention_with_weights(
                q, k, v, output, block_indices, is_causal
            )
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)

        # Reverse element-level Hilbert ordering if applied
        if self.hilbert_within_blocks:
            output = self._reorder_tensor_hilbert(output, self.block_size, inverse=True)

        # Return results
        if return_attention_weights:
            # Include Hilbert optimization info
            attention_info["hilbert_optimized"] = True
            attention_info["hilbert_block_level"] = self.hilbert_block_level
            attention_info["hilbert_within_blocks"] = self.hilbert_within_blocks
            return output, attention_info

        return output

    def get_pattern_stats(self) -> dict:
        """Get statistics including Hilbert optimization info."""
        stats = super().get_pattern_stats()

        # Add Hilbert-specific stats
        stats["hilbert_optimization"] = {
            "enabled": self.use_hilbert,
            "block_level": self.hilbert_block_level,
            "within_blocks": self.hilbert_within_blocks,
            "cached_block_mappings": len(self._hilbert_block_cache)
            if hasattr(self, "_hilbert_block_cache")
            else 0,
            "cached_element_mappings": len(self._hilbert_element_cache)
            if hasattr(self, "_hilbert_element_cache")
            else 0,
        }

        return stats


def create_block_sparse_hilbert(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparsity_ratio: float = 0.1,
    pattern_type: str = "dilated_sparse",
    block_size: int = 64,
    use_hilbert: bool = True,
    **kwargs,
) -> BlockSparseRingDilatedAttentionHilbert:
    """
    Convenience function to create Hilbert-optimized block-sparse attention.

    Args:
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for each segment
        sparsity_ratio: Sparsity ratio (0.1 = 90% sparse)
        pattern_type: Type of sparse pattern
        block_size: Size of attention blocks
        use_hilbert: Whether to use Hilbert optimization
        **kwargs: Additional arguments

    Returns:
        BlockSparseRingDilatedAttentionHilbert instance
    """
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    return BlockSparseRingDilatedAttentionHilbert(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_hilbert=use_hilbert,
        **kwargs,
    )
