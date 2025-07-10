"""
Block-Sparse Attention with Post-Pattern Hilbert Optimization.

This implementation applies Hilbert ordering AFTER the sparse pattern is determined,
optimizing only the order in which the selected blocks are processed, not which
blocks are selected.
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, List
import numpy as np

from .block_sparse_attention import (
    BlockSparseAttention,
    SparsePatternConfig,
)
from ..utils.hilbert_curve import generate_hilbert_indices


class PostPatternHilbertOptimizer:
    """
    Optimizes the processing order of sparse blocks using Hilbert curves.

    Key insight: Don't change WHICH blocks interact, just optimize the ORDER
    in which we process them for better cache locality.
    """

    def __init__(self, cache_orderings: bool = True):
        self.cache_orderings = cache_orderings
        self._ordering_cache: Dict[Tuple[int, int, str], Tensor] = {}

    def optimize_block_processing_order(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        num_blocks: int,
        pattern_hash: Optional[str] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Reorder block indices to optimize cache locality during processing.

        This doesn't change which blocks interact, only the order of processing.
        """
        # Check cache
        if self.cache_orderings and pattern_hash:
            cache_key = (len(row_indices), num_blocks, pattern_hash)
            if cache_key in self._ordering_cache:
                order = self._ordering_cache[cache_key].to(row_indices.device)
                return row_indices[order], col_indices[order]

        # Analyze the access pattern
        access_pattern = self._analyze_access_pattern(
            row_indices, col_indices, num_blocks
        )

        # Compute optimal processing order
        optimal_order = self._compute_optimal_order(
            row_indices, col_indices, num_blocks, access_pattern
        )

        # Cache if enabled
        if self.cache_orderings and pattern_hash:
            self._ordering_cache[cache_key] = optimal_order.cpu()

        # Apply ordering
        return row_indices[optimal_order], col_indices[optimal_order]

    def _analyze_access_pattern(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        num_blocks: int,
    ) -> Dict[str, any]:
        """Analyze the sparse pattern to understand access characteristics."""
        # Convert to numpy for analysis
        rows = row_indices.cpu().numpy()
        cols = col_indices.cpu().numpy()

        # Compute statistics
        pattern_info = {
            "num_connections": len(rows),
            "density": len(rows) / (num_blocks * num_blocks),
            "diagonal_blocks": np.sum(rows == cols),
            "avg_distance": np.mean(np.abs(rows - cols)),
        }

        # Identify pattern type
        if pattern_info["diagonal_blocks"] > len(rows) * 0.8:
            pattern_info["type"] = "mostly_diagonal"
        elif pattern_info["avg_distance"] > num_blocks * 0.3:
            pattern_info["type"] = "long_range"
        else:
            pattern_info["type"] = "mixed"

        return pattern_info

    def _compute_optimal_order(
        self,
        row_indices: Tensor,
        col_indices: Tensor,
        num_blocks: int,
        pattern_info: Dict[str, any],
    ) -> Tensor:
        """
        Compute optimal processing order based on pattern characteristics.
        """
        device = row_indices.device
        num_pairs = len(row_indices)

        if pattern_info["type"] == "mostly_diagonal":
            # For diagonal-dominant patterns, process in block order
            # This maintains spatial locality
            block_sum = row_indices + col_indices  # Approximate position
            return block_sum.argsort()

        elif pattern_info["type"] == "long_range":
            # For long-range patterns, use Hilbert curve to optimize
            # Create a virtual 2D space of block interactions
            grid_size = int(math.ceil(math.sqrt(num_blocks)))

            # Ensure grid is power of 2 for Hilbert
            hilbert_size = 1
            while hilbert_size < grid_size:
                hilbert_size *= 2

            # Generate Hilbert curve
            n_levels = int(math.log2(hilbert_size))
            hilbert_curve = generate_hilbert_indices(n_levels)

            # Map block pairs to Hilbert positions
            hilbert_positions = torch.zeros(num_pairs, device=device, dtype=torch.long)

            for i in range(num_pairs):
                row = row_indices[i].item()
                col = col_indices[i].item()

                # Map to 2D position
                linear_pos = row * grid_size + col
                if linear_pos < len(hilbert_curve):
                    hilbert_pos = hilbert_curve.index(linear_pos)
                else:
                    # Fallback for out-of-range
                    hilbert_pos = linear_pos

                hilbert_positions[i] = hilbert_pos

            return hilbert_positions.argsort()

        else:  # mixed pattern
            # Use a hybrid approach: group by row, then optimize within groups
            # This balances between row-wise processing and cache optimization

            # Sort primarily by row, secondarily by Hilbert within row
            row_order = row_indices.argsort(stable=True)
            sorted_rows = row_indices[row_order]
            sorted_cols = col_indices[row_order]

            # Identify row groups
            unique_rows, row_starts = torch.unique_consecutive(
                sorted_rows, return_inverse=False, return_counts=True
            )

            # Optimize within each row group
            final_order = torch.zeros(num_pairs, device=device, dtype=torch.long)
            current_pos = 0

            for row_idx, count in enumerate(row_starts):
                if count > 1:
                    # Multiple blocks in this row - optimize their order
                    group_start = current_pos
                    group_end = current_pos + count
                    group_cols = sorted_cols[group_start:group_end]

                    # Sort by column within group for cache locality
                    col_order = group_cols.argsort()

                    # Apply sub-ordering
                    for i, pos in enumerate(col_order):
                        final_order[group_start + i] = row_order[group_start + pos]
                else:
                    # Single block - keep as is
                    final_order[current_pos] = row_order[current_pos]

                current_pos += count

            return final_order


class BlockSparseAttentionHilbert(BlockSparseAttention):
    """
    Block-Sparse attention with post-pattern Hilbert optimization.

    This implementation:
    1. Determines the sparse pattern normally
    2. Optimizes the processing order of selected blocks
    3. Doesn't change which blocks interact
    """

    def __init__(
        self,
        sparse_config: SparsePatternConfig,
        use_post_pattern_optimization: bool = True,
        cache_orderings: bool = True,
        **kwargs,
    ):
        super().__init__(sparse_config, **kwargs)

        self.use_post_pattern_optimization = use_post_pattern_optimization
        self.optimizer = PostPatternHilbertOptimizer(cache_orderings)

        # For pattern hashing
        self._pattern_hash_cache: Dict[Tuple[int, int], str] = {}

    def _get_pattern_hash(self, num_blocks: int, num_heads: int) -> str:
        """Generate a hash for the current sparse pattern configuration."""
        cache_key = (num_blocks, num_heads)
        if cache_key in self._pattern_hash_cache:
            return self._pattern_hash_cache[cache_key]

        # Create hash from configuration
        pattern_hash = f"{self.sparse_config.pattern_type}_{self.sparse_config.sparsity_ratio}_{self.sparse_config.block_size}_{num_blocks}"
        self._pattern_hash_cache[cache_key] = pattern_hash
        return pattern_hash

    def _get_optimized_sparse_block_indices(
        self,
        num_blocks: int,
        num_heads: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get sparse block indices with post-pattern optimization.
        """
        # First get standard sparse block indices
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, device)

        if not self.use_post_pattern_optimization:
            return block_indices

        # Get pattern hash for caching
        pattern_hash = self._get_pattern_hash(num_blocks, num_heads)

        row_indices, col_indices = block_indices

        # Handle multi-head format
        if len(row_indices.shape) == 2:  # [num_heads, num_blocks_per_head]
            # Optimize each head separately
            optimized_rows = torch.zeros_like(row_indices)
            optimized_cols = torch.zeros_like(col_indices)

            for head in range(num_heads):
                opt_row, opt_col = self.optimizer.optimize_block_processing_order(
                    row_indices[head],
                    col_indices[head],
                    num_blocks,
                    f"{pattern_hash}_h{head}",
                )
                optimized_rows[head] = opt_row
                optimized_cols[head] = opt_col

            return optimized_rows, optimized_cols
        else:
            # Single pattern shared across heads
            return self.optimizer.optimize_block_processing_order(
                row_indices, col_indices, num_blocks, pattern_hash
            )

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
        Forward pass with post-pattern optimization.

        The sparse pattern is determined normally, but blocks are processed
        in an optimized order for better cache locality.
        """
        batch, seq_len, num_heads, head_dim = q.shape
        num_blocks = seq_len // self.block_size

        # Get optimized block indices
        block_indices = self._get_optimized_sparse_block_indices(
            num_blocks, num_heads, q.device
        )

        # Initialize output
        output = torch.zeros_like(q)

        # Compute attention using parent class methods
        if return_attention_weights:
            attention_info = self._compute_sparse_attention_with_weights(
                q, k, v, output, block_indices, is_causal
            )
            # Add optimization info
            attention_info["optimization"] = "post_pattern_hilbert"
            attention_info["pattern_optimized"] = True
            return output, attention_info
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)
            return output

    def analyze_optimization_impact(self, seq_len: int) -> Dict[str, any]:
        """
        Analyze how the optimization changes processing order.
        """
        num_blocks = seq_len // self.block_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get both standard and optimized orderings
        _ = self._get_sparse_block_indices(num_blocks, 1, device)

        self.use_post_pattern_optimization = False
        unoptimized = self._get_optimized_sparse_block_indices(num_blocks, 1, device)

        self.use_post_pattern_optimization = True
        optimized = self._get_optimized_sparse_block_indices(num_blocks, 1, device)

        # Analyze differences
        row_std, col_std = unoptimized
        row_opt, col_opt = optimized

        # Compute metrics
        def compute_cache_metric(rows, cols):
            """Estimate cache efficiency based on access pattern."""
            # Simple metric: sum of distances between consecutive accesses
            distances = []
            for i in range(len(rows) - 1):
                # Manhattan distance in block space
                dist = abs(rows[i + 1] - rows[i]) + abs(cols[i + 1] - cols[i])
                distances.append(dist.item() if hasattr(dist, "item") else dist)
            return sum(distances) / len(distances) if distances else 0

        standard_metric = compute_cache_metric(
            row_std[0] if len(row_std.shape) > 1 else row_std,
            col_std[0] if len(col_std.shape) > 1 else col_std,
        )
        optimized_metric = compute_cache_metric(
            row_opt[0] if len(row_opt.shape) > 1 else row_opt,
            col_opt[0] if len(col_opt.shape) > 1 else col_opt,
        )

        return {
            "num_blocks": num_blocks,
            "num_connections": len(row_std[0] if len(row_std.shape) > 1 else row_std),
            "standard_cache_metric": standard_metric,
            "optimized_cache_metric": optimized_metric,
            "improvement": (standard_metric - optimized_metric) / standard_metric * 100,
            "pattern_preserved": torch.allclose(
                torch.sort(row_std)[0], torch.sort(row_opt)[0]
            )
            and torch.allclose(torch.sort(col_std)[0], torch.sort(col_opt)[0]),
        }


def create_post_pattern_hilbert_attention(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparsity_ratio: float = 0.1,
    pattern_type: str = "dilated_sparse",
    block_size: int = 64,
    **kwargs,
) -> BlockSparseAttentionHilbert:
    """
    Factory function for post-pattern Hilbert attention.
    """
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    return BlockSparseAttentionHilbert(
        sparse_config=sparse_config,
        use_post_pattern_optimization=True,
        **kwargs,
    )
