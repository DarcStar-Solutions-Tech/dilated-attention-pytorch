"""
Block-Sparse Ring Dilated Attention with Cached Hilbert Space-Filling Curve Optimization.

This implementation pre-computes and caches Hilbert orderings during initialization
to minimize overhead in the forward pass. Key optimizations:
1. Pre-computes Hilbert orderings once during initialization
2. Caches the reordered indices for each sequence length
3. Avoids recomputing Hilbert indices on every forward pass
4. Only applies the pre-computed ordering to block indices
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, List

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from .utils.hilbert_curve import (
    generate_hilbert_indices,
    generate_hilbert_indices_rectangular,
)


class BlockSparseRingDilatedAttentionHilbertCached(BlockSparseRingDilatedAttention):
    """
    Block-Sparse Ring Dilated Attention with pre-computed cached Hilbert curve optimization.

    This implementation pre-computes all Hilbert orderings during initialization or
    first use, avoiding the overhead of computing them on every forward pass.
    The cached orderings are stored as simple lookup tables for O(1) access.

    Key optimizations over the standard Hilbert implementation:
    1. Pre-computation: All Hilbert orderings computed once
    2. Caching: Sequence length -> ordering mapping stored
    3. Minimal overhead: Simple index lookup in forward pass
    4. Memory efficient: Only stores unique orderings
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: SparsePatternConfig,
        use_hilbert: bool = True,
        hilbert_block_level: bool = True,
        precompute_common_sizes: bool = True,
        common_seq_lengths: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize Cached Hilbert-optimized block-sparse attention.

        Args:
            segment_lengths: Segment lengths for dilated attention
            dilation_rates: Dilation rates for each segment
            sparse_config: Sparse pattern configuration
            use_hilbert: Whether to use Hilbert ordering
            hilbert_block_level: Apply Hilbert ordering at block level
            precompute_common_sizes: Pre-compute orderings for common sequence lengths
            common_seq_lengths: List of common sequence lengths to pre-compute
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        self.use_hilbert = use_hilbert
        self.hilbert_block_level = hilbert_block_level

        # Caches for pre-computed Hilbert orderings
        # Maps (num_blocks_h, num_blocks_w) -> (forward_ordering, inverse_ordering)
        self._hilbert_block_orderings: Dict[Tuple[int, int], Tuple[Tensor, Tensor]] = {}

        # Maps seq_length -> block grid dimensions for quick lookup
        self._seq_len_to_grid: Dict[int, Tuple[int, int]] = {}

        # Pre-compute orderings for common sequence lengths
        if use_hilbert and precompute_common_sizes:
            if common_seq_lengths is None:
                # Common sequence lengths in practice
                common_seq_lengths = [
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    131072,
                ]

            for seq_len in common_seq_lengths:
                if seq_len % self.block_size == 0:
                    self._precompute_hilbert_ordering(seq_len)

    def _precompute_hilbert_ordering(self, seq_length: int) -> None:
        """Pre-compute and cache Hilbert ordering for a given sequence length."""
        if not self.use_hilbert or not self.hilbert_block_level:
            return

        num_blocks = seq_length // self.block_size
        grid_size = int(math.ceil(math.sqrt(num_blocks)))

        # Store the mapping from sequence length to grid dimensions
        self._seq_len_to_grid[seq_length] = (grid_size, grid_size)

        # Check if we already have this grid size cached
        cache_key = (grid_size, grid_size)
        if cache_key not in self._hilbert_block_orderings:
            # Compute Hilbert ordering for this grid size
            forward_order, inverse_order = self._compute_hilbert_ordering(
                grid_size, grid_size
            )
            self._hilbert_block_orderings[cache_key] = (forward_order, inverse_order)

    def _compute_hilbert_ordering(
        self, num_blocks_h: int, num_blocks_w: int
    ) -> Tuple[Tensor, Tensor]:
        """Compute Hilbert ordering for a block grid."""
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

        return hilbert_indices, inverse_indices

    def _get_cached_hilbert_ordering(
        self, seq_length: int
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Get cached Hilbert ordering for a sequence length, computing if necessary."""
        if not self.use_hilbert or not self.hilbert_block_level:
            return None

        # Check if we need to compute the ordering for this sequence length
        if seq_length not in self._seq_len_to_grid:
            self._precompute_hilbert_ordering(seq_length)

        # Get grid dimensions
        grid_h, grid_w = self._seq_len_to_grid[seq_length]
        cache_key = (grid_h, grid_w)

        # Return the cached ordering
        return self._hilbert_block_orderings.get(cache_key)

    def _apply_cached_hilbert_ordering(
        self, row_indices: Tensor, col_indices: Tensor, seq_length: int, num_blocks: int
    ) -> Tuple[Tensor, Tensor]:
        """Apply pre-computed Hilbert ordering to block indices with minimal overhead."""
        if not self.use_hilbert or not self.hilbert_block_level:
            return row_indices, col_indices

        # Get cached ordering
        ordering_pair = self._get_cached_hilbert_ordering(seq_length)
        if ordering_pair is None:
            return row_indices, col_indices

        hilbert_indices, _ = ordering_pair
        hilbert_indices = hilbert_indices.to(row_indices.device)

        # Convert to 2D grid coordinates
        grid_size = int(math.ceil(math.sqrt(num_blocks)))

        # Create mapping from linear to Hilbert order
        linear_indices = row_indices * grid_size + col_indices

        # Fast lookup: Create a mapping tensor
        # This avoids the slow loop in the original implementation
        max_idx = min(linear_indices.max().item() + 1, len(hilbert_indices))
        _ = torch.arange(max_idx, device=row_indices.device)

        # Find positions in Hilbert curve using the pre-computed ordering
        # Create reverse mapping for O(1) lookup
        hilbert_to_pos = torch.zeros(
            len(hilbert_indices), dtype=torch.long, device=row_indices.device
        )
        hilbert_to_pos[hilbert_indices[:max_idx]] = torch.arange(
            max_idx, device=row_indices.device
        )

        # Apply the mapping
        valid_mask = linear_indices < max_idx
        hilbert_positions = torch.zeros_like(linear_indices)
        hilbert_positions[valid_mask] = hilbert_to_pos[linear_indices[valid_mask]]
        hilbert_positions[~valid_mask] = linear_indices[
            ~valid_mask
        ]  # Fallback for out-of-range

        # Sort by Hilbert position
        sorted_indices = hilbert_positions.argsort()

        return row_indices[sorted_indices], col_indices[sorted_indices]

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
        Forward pass with cached Hilbert-optimized block-sparse attention.

        This implementation minimizes overhead by using pre-computed Hilbert
        orderings stored in memory. The only computation in the forward pass
        is a simple index lookup and reordering operation.
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # Get sparse block indices using parent method
        num_blocks = seq_len // self.block_size
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, q.device)

        # Apply cached Hilbert ordering with minimal overhead
        row_indices, col_indices = block_indices
        row_indices, col_indices = self._apply_cached_hilbert_ordering(
            row_indices, col_indices, seq_len, num_blocks
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

        # Return results
        if return_attention_weights:
            # Include optimization info
            attention_info["hilbert_optimized"] = True
            attention_info["hilbert_cached"] = True
            attention_info["cached_orderings"] = len(self._hilbert_block_orderings)
            return output, attention_info

        return output

    def get_pattern_stats(self) -> dict:
        """Get statistics including cached Hilbert optimization info."""
        stats = super().get_pattern_stats()

        # Add Hilbert-specific stats
        stats["hilbert_optimization"] = {
            "enabled": self.use_hilbert,
            "block_level": self.hilbert_block_level,
            "cached_orderings": len(self._hilbert_block_orderings),
            "cached_seq_lengths": len(self._seq_len_to_grid),
            "memory_usage_bytes": sum(
                ordering[0].numel() * ordering[0].element_size()
                + ordering[1].numel() * ordering[1].element_size()
                for ordering in self._hilbert_block_orderings.values()
            ),
        }

        return stats

    def clear_cache(self) -> None:
        """Clear all cached Hilbert orderings to free memory."""
        self._hilbert_block_orderings.clear()
        self._seq_len_to_grid.clear()


def create_cached_block_sparse_hilbert(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparsity_ratio: float = 0.1,
    pattern_type: str = "dilated_sparse",
    block_size: int = 64,
    use_hilbert: bool = True,
    precompute_seq_lengths: Optional[List[int]] = None,
    **kwargs,
) -> BlockSparseRingDilatedAttentionHilbertCached:
    """
    Convenience function to create cached Hilbert-optimized block-sparse attention.

    Args:
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for each segment
        sparsity_ratio: Sparsity ratio (0.1 = 90% sparse)
        pattern_type: Type of sparse pattern
        block_size: Size of attention blocks
        use_hilbert: Whether to use Hilbert optimization
        precompute_seq_lengths: Specific sequence lengths to pre-compute
        **kwargs: Additional arguments

    Returns:
        BlockSparseRingDilatedAttentionHilbertCached instance
    """
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    return BlockSparseRingDilatedAttentionHilbertCached(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_hilbert=use_hilbert,
        common_seq_lengths=precompute_seq_lengths,
        **kwargs,
    )
