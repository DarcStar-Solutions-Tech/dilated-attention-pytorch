"""
Block-Sparse Ring Dilated Attention with Dilation-Aware Hilbert Ordering.

This implementation applies Hilbert space-filling curves in a way that respects
the dilated access patterns, grouping blocks that will be accessed together
and optimizing within those groups.
"""

import math
import torch
from torch import Tensor
from typing import Tuple, Dict, List

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from .utils.hilbert_curve import generate_hilbert_indices


class DilationAwareHilbertOrdering:
    """
    Manages Hilbert ordering that respects dilated attention patterns.

    Key insight: With dilation rate D, blocks are accessed in groups based on
    their position modulo D. We should apply Hilbert ordering within these
    groups, not across the entire sequence.
    """

    def __init__(self, cache_patterns: bool = True):
        self.cache_patterns = cache_patterns
        self._pattern_cache: Dict[Tuple[int, int, int], Tensor] = {}
        self._access_group_cache: Dict[Tuple[int, int], List[List[int]]] = {}

    def get_dilation_access_groups(
        self, num_blocks: int, dilation_rate: int, pattern_type: str = "dilated_sparse"
    ) -> List[List[int]]:
        """
        Group blocks based on dilated access patterns.

        For dilation rate D, blocks that are D positions apart will be
        accessed together, so we group them.
        """
        cache_key = (num_blocks, dilation_rate)
        if self.cache_patterns and cache_key in self._access_group_cache:
            return self._access_group_cache[cache_key]

        if pattern_type == "dilated_sparse":
            # With dilation, blocks access other blocks at multiples of dilation_rate
            # Group blocks that will interact due to dilation
            groups = []

            # Create groups based on dilation pattern
            # Blocks at positions 0, D, 2D, 3D... will access each other
            for offset in range(min(dilation_rate, num_blocks)):
                group = []
                pos = offset
                while pos < num_blocks:
                    group.append(pos)
                    pos += dilation_rate
                if len(group) > 1:  # Only include groups with multiple blocks
                    groups.append(group)

            # Also include local groups (adjacent blocks)
            # These represent the non-dilated local attention
            local_window = 3  # How many adjacent blocks to group
            for start in range(0, num_blocks, local_window):
                end = min(start + local_window, num_blocks)
                local_group = list(range(start, end))
                if len(local_group) > 1:
                    groups.append(local_group)

        else:
            # For other patterns, fall back to sequential grouping
            group_size = max(2, num_blocks // 8)  # Create ~8 groups
            groups = []
            for start in range(0, num_blocks, group_size):
                end = min(start + group_size, num_blocks)
                groups.append(list(range(start, end)))

        if self.cache_patterns:
            self._access_group_cache[cache_key] = groups

        return groups

    def apply_hilbert_within_groups(
        self,
        block_indices: Tuple[Tensor, Tensor],
        access_groups: List[List[int]],
        num_blocks: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply Hilbert ordering within each access group.

        This preserves the dilated access pattern while optimizing
        memory locality within each group.
        """
        row_indices, col_indices = block_indices
        device = row_indices.device

        # Create a mapping of block position to indices in the sparse pattern
        block_to_indices = {}
        for idx, (row, col) in enumerate(
            zip(row_indices.tolist(), col_indices.tolist())
        ):
            key = (row, col)
            if key not in block_to_indices:
                block_to_indices[key] = []
            block_to_indices[key].append(idx)

        # Collect indices to reorder
        reordered_indices = []

        for group in access_groups:
            if len(group) <= 1:
                continue

            # Find all block pairs within this group
            group_indices = []
            for i, row in enumerate(group):
                for j, col in enumerate(group):
                    key = (row, col)
                    if key in block_to_indices:
                        group_indices.extend(block_to_indices[key])

            if len(group_indices) <= 1:
                continue

            # Apply Hilbert ordering to this group
            group_size = int(math.ceil(math.sqrt(len(group))))
            if group_size > 1 and (group_size & (group_size - 1)) == 0:
                # Power of 2 - use standard Hilbert
                n_levels = int(math.log2(group_size))
                hilbert_order = generate_hilbert_indices(n_levels)

                # Map group indices through Hilbert curve
                hilbert_positions = []
                for idx in range(len(group_indices)):
                    if idx < len(hilbert_order):
                        hilbert_positions.append(hilbert_order[idx])
                    else:
                        hilbert_positions.append(idx)

                # Sort group indices by Hilbert position
                sorted_group = [
                    group_indices[i]
                    for i in sorted(
                        range(len(group_indices)), key=lambda x: hilbert_positions[x]
                    )
                ]
                reordered_indices.extend(sorted_group)
            else:
                # Non-power-of-2 - keep original order within group
                reordered_indices.extend(group_indices)

        # Add any remaining indices not in groups
        seen = set(reordered_indices)
        for idx in range(len(row_indices)):
            if idx not in seen:
                reordered_indices.append(idx)

        # Apply reordering
        reorder_tensor = torch.tensor(
            reordered_indices, device=device, dtype=torch.long
        )
        return row_indices[reorder_tensor], col_indices[reorder_tensor]


class BlockSparseRingDilatedAttentionHilbertDilationAware(
    BlockSparseRingDilatedAttention
):
    """
    Block-Sparse attention with dilation-aware Hilbert optimization.

    This implementation groups blocks based on dilated access patterns and
    applies Hilbert ordering within each group, preserving the benefits of
    both optimizations.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: SparsePatternConfig,
        use_dilation_aware_hilbert: bool = True,
        cache_patterns: bool = True,
        **kwargs,
    ):
        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        self.use_dilation_aware_hilbert = use_dilation_aware_hilbert
        self.hilbert_ordering = DilationAwareHilbertOrdering(cache_patterns)

        # Track which dilation rate is active
        self.current_dilation_rate = max(dilation_rates) if dilation_rates else 1

    def _get_sparse_block_indices_with_hilbert(
        self,
        num_blocks: int,
        num_heads: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get sparse block indices with dilation-aware Hilbert optimization.
        """
        # First get standard sparse block indices
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, device)

        if not self.use_dilation_aware_hilbert:
            return block_indices

        # Get access groups based on dilation pattern
        access_groups = self.hilbert_ordering.get_dilation_access_groups(
            num_blocks, self.current_dilation_rate, self.sparse_config.pattern_type
        )

        # Apply Hilbert ordering within each group
        # Process each head separately to maintain the multi-head structure
        row_indices, col_indices = block_indices

        if len(row_indices.shape) == 2:  # Multi-head format
            new_row_indices = torch.zeros_like(row_indices)
            new_col_indices = torch.zeros_like(col_indices)

            for head in range(num_heads):
                head_row = row_indices[head]
                head_col = col_indices[head]

                # Apply dilation-aware Hilbert to this head
                reordered_row, reordered_col = (
                    self.hilbert_ordering.apply_hilbert_within_groups(
                        (head_row, head_col), access_groups, num_blocks
                    )
                )

                new_row_indices[head] = reordered_row
                new_col_indices[head] = reordered_col

            return new_row_indices, new_col_indices
        else:
            # Single pattern shared across heads
            return self.hilbert_ordering.apply_hilbert_within_groups(
                block_indices, access_groups, num_blocks
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
        Forward pass with dilation-aware Hilbert optimization.
        """
        batch, seq_len, num_heads, head_dim = q.shape
        num_blocks = seq_len // self.block_size

        # Get optimized block indices
        block_indices = self._get_sparse_block_indices_with_hilbert(
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
            attention_info["optimization"] = "dilation_aware_hilbert"
            attention_info["dilation_rate"] = self.current_dilation_rate
            attention_info["access_groups"] = len(
                self.hilbert_ordering.get_dilation_access_groups(
                    num_blocks, self.current_dilation_rate
                )
            )
            return output, attention_info
        else:
            self._compute_sparse_attention(q, k, v, output, block_indices, is_causal)
            return output

    def visualize_access_pattern(self, seq_len: int) -> Dict[str, Tensor]:
        """
        Visualize the access pattern to understand optimization impact.
        """
        num_blocks = seq_len // self.block_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get standard pattern
        standard_indices = self._get_sparse_block_indices(num_blocks, 1, device)

        # Get optimized pattern
        self.use_dilation_aware_hilbert = True
        optimized_indices = self._get_sparse_block_indices_with_hilbert(
            num_blocks, 1, device
        )

        # Create visualization matrices
        standard_matrix = torch.zeros(num_blocks, num_blocks, device=device)
        optimized_matrix = torch.zeros(num_blocks, num_blocks, device=device)

        # Fill matrices
        for row, col in zip(*standard_indices):
            standard_matrix[row, col] = 1

        for row, col in zip(*optimized_indices):
            optimized_matrix[row, col] = 1

        # Get access groups for visualization
        access_groups = self.hilbert_ordering.get_dilation_access_groups(
            num_blocks, self.current_dilation_rate
        )

        group_matrix = torch.zeros(num_blocks, num_blocks, device=device)
        for group_id, group in enumerate(access_groups):
            for i in group:
                for j in group:
                    if i < num_blocks and j < num_blocks:
                        group_matrix[i, j] = group_id + 1

        return {
            "standard_pattern": standard_matrix,
            "optimized_pattern": optimized_matrix,
            "access_groups": group_matrix,
            "num_groups": len(access_groups),
        }


def create_dilation_aware_hilbert_attention(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparsity_ratio: float = 0.1,
    pattern_type: str = "dilated_sparse",
    block_size: int = 64,
    **kwargs,
) -> BlockSparseRingDilatedAttentionHilbertDilationAware:
    """
    Factory function for dilation-aware Hilbert attention.
    """
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    return BlockSparseRingDilatedAttentionHilbertDilationAware(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_dilation_aware_hilbert=True,
        **kwargs,
    )
