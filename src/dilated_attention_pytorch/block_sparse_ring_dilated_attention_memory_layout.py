"""
Block-Sparse Ring Dilated Attention with Memory Layout Optimization.

This implementation rearranges data in memory to match dilated access patterns,
improving cache efficiency by ensuring that data accessed together is stored
contiguously.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict, List
import numpy as np

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


class DilationAwareMemoryLayout:
    """
    Manages memory layout optimization for dilated attention patterns.

    Key insight: Instead of reordering computation, reorder the data layout
    so that elements accessed together are stored contiguously in memory.
    """

    def __init__(self, block_size: int, cache_layouts: bool = True):
        self.block_size = block_size
        self.cache_layouts = cache_layouts
        self._layout_cache: Dict[Tuple[int, int, int], Tuple[Tensor, Tensor]] = {}

    def compute_dilation_aware_layout(
        self,
        seq_len: int,
        dilation_rate: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute memory layout that groups elements accessed together.

        Returns:
            forward_mapping: Maps from original position to optimized position
            inverse_mapping: Maps from optimized position back to original
        """
        cache_key = (seq_len, dilation_rate, self.block_size)

        if self.cache_layouts and cache_key in self._layout_cache:
            fwd, inv = self._layout_cache[cache_key]
            return fwd.to(device), inv.to(device)

        # Compute layout based on dilation pattern
        if dilation_rate == 1:
            # No dilation - keep original layout
            forward_mapping = torch.arange(seq_len, device=device)
            inverse_mapping = torch.arange(seq_len, device=device)
        else:
            # Group positions by dilation pattern
            # Elements at positions [0, D, 2D, ...] will be accessed together
            forward_mapping = torch.zeros(seq_len, dtype=torch.long, device=device)
            inverse_mapping = torch.zeros(seq_len, dtype=torch.long, device=device)

            current_pos = 0

            # First, place elements by dilation groups
            for offset in range(dilation_rate):
                positions = list(range(offset, seq_len, dilation_rate))
                for i, pos in enumerate(positions):
                    forward_mapping[pos] = current_pos
                    inverse_mapping[current_pos] = pos
                    current_pos += 1

            # Further optimize within blocks
            if self.block_size > 1:
                # Reorder within each block for better cache line usage
                num_blocks = seq_len // self.block_size
                optimized_forward = forward_mapping.clone()

                for block_idx in range(num_blocks):
                    block_start = block_idx * self.block_size
                    block_end = block_start + self.block_size

                    # Get positions within this block
                    block_positions = forward_mapping[block_start:block_end]

                    # Sort to keep dilated groups together within blocks
                    sorted_indices = block_positions.argsort()

                    # Apply local optimization
                    for i, idx in enumerate(sorted_indices):
                        optimized_forward[block_start + idx] = block_start + i

                # Update inverse mapping
                for i in range(seq_len):
                    inverse_mapping[optimized_forward[i]] = i

                forward_mapping = optimized_forward

        if self.cache_layouts:
            self._layout_cache[cache_key] = (
                forward_mapping.cpu(),
                inverse_mapping.cpu(),
            )

        return forward_mapping, inverse_mapping

    def reorder_tensor_to_optimized_layout(
        self,
        tensor: Tensor,
        dilation_rate: int,
    ) -> Tensor:
        """Reorder tensor to optimized memory layout."""
        batch, seq_len = tensor.shape[:2]
        _ = tensor.shape[2:]

        forward_mapping, _ = self.compute_dilation_aware_layout(
            seq_len, dilation_rate, tensor.device
        )

        # Reorder along sequence dimension
        return tensor.index_select(1, forward_mapping)

    def reorder_tensor_from_optimized_layout(
        self,
        tensor: Tensor,
        dilation_rate: int,
    ) -> Tensor:
        """Reorder tensor from optimized layout back to original."""
        batch, seq_len = tensor.shape[:2]

        _, inverse_mapping = self.compute_dilation_aware_layout(
            seq_len, dilation_rate, tensor.device
        )

        # Reorder back to original layout
        return tensor.index_select(1, inverse_mapping)


class BlockSparseRingDilatedAttentionMemoryLayout(BlockSparseRingDilatedAttention):
    """
    Block-Sparse attention with memory layout optimization.

    This implementation:
    1. Reorders input data to match dilated access patterns
    2. Performs attention computation on reordered data
    3. Reorders output back to original layout

    The reordering overhead is amortized over multiple attention operations.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: SparsePatternConfig,
        use_memory_layout_optimization: bool = True,
        cache_layouts: bool = True,
        **kwargs,
    ):
        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        self.use_memory_layout_optimization = use_memory_layout_optimization
        self.layout_manager = DilationAwareMemoryLayout(self.block_size, cache_layouts)

        # Current dilation rate for layout optimization
        self.current_dilation_rate = max(dilation_rates) if dilation_rates else 1

        # Optional: learnable projection to adapt to reordered layout
        self.layout_adapter = None
        if use_memory_layout_optimization:
            # Small MLP to adapt features after reordering
            hidden_dim = kwargs.get("hidden_dim", 256)
            self.layout_adapter = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
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
        Forward pass with memory layout optimization.
        """
        batch, seq_len, num_heads, head_dim = q.shape

        if self.use_memory_layout_optimization and self.current_dilation_rate > 1:
            # Reorder inputs to optimized layout
            q_reordered = self.layout_manager.reorder_tensor_to_optimized_layout(
                q, self.current_dilation_rate
            )
            k_reordered = self.layout_manager.reorder_tensor_to_optimized_layout(
                k, self.current_dilation_rate
            )
            v_reordered = self.layout_manager.reorder_tensor_to_optimized_layout(
                v, self.current_dilation_rate
            )

            # Optional: apply layout adapter
            if self.layout_adapter is not None:
                # Reshape for adapter
                q_flat = q_reordered.view(batch * seq_len, -1)
                k_flat = k_reordered.view(batch * seq_len, -1)
                v_flat = v_reordered.view(batch * seq_len, -1)

                # Apply adapter
                q_flat = q_flat + self.layout_adapter(q_flat)
                k_flat = k_flat + self.layout_adapter(k_flat)
                v_flat = v_flat + self.layout_adapter(v_flat)

                # Reshape back
                q_reordered = q_flat.view(batch, seq_len, num_heads, head_dim)
                k_reordered = k_flat.view(batch, seq_len, num_heads, head_dim)
                v_reordered = v_flat.view(batch, seq_len, num_heads, head_dim)
        else:
            # No reordering for dilation_rate=1 or if disabled
            q_reordered = q
            k_reordered = k
            v_reordered = v

        # Get block indices
        num_blocks = seq_len // self.block_size
        block_indices = self._get_sparse_block_indices(num_blocks, num_heads, q.device)

        # Initialize output
        output = torch.zeros_like(q_reordered)

        # Compute attention on reordered data
        if return_attention_weights:
            attention_info = self._compute_sparse_attention_with_weights(
                q_reordered, k_reordered, v_reordered, output, block_indices, is_causal
            )
        else:
            self._compute_sparse_attention(
                q_reordered, k_reordered, v_reordered, output, block_indices, is_causal
            )

        # Reorder output back to original layout
        if self.use_memory_layout_optimization and self.current_dilation_rate > 1:
            output = self.layout_manager.reorder_tensor_from_optimized_layout(
                output, self.current_dilation_rate
            )

        if return_attention_weights:
            attention_info["optimization"] = "memory_layout"
            attention_info["dilation_rate"] = self.current_dilation_rate
            attention_info["layout_optimized"] = self.use_memory_layout_optimization
            return output, attention_info
        else:
            return output

    def analyze_memory_access_pattern(self, seq_len: int) -> Dict[str, any]:
        """
        Analyze how memory layout optimization affects access patterns.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dummy data
        _ = torch.arange(seq_len, device=device).float()

        # Get layout mappings
        forward_map, inverse_map = self.layout_manager.compute_dilation_aware_layout(
            seq_len, self.current_dilation_rate, device
        )

        # Analyze access pattern improvement
        # Simulate dilated access pattern
        access_distances_original = []
        access_distances_optimized = []

        for i in range(
            0, seq_len - self.current_dilation_rate, self.current_dilation_rate
        ):
            # Original layout: accessing i and i+dilation_rate
            dist_original = self.current_dilation_rate
            access_distances_original.append(dist_original)

            # Optimized layout: where are these positions now?
            pos1_optimized = forward_map[i].item()
            pos2_optimized = forward_map[i + self.current_dilation_rate].item()
            dist_optimized = abs(pos2_optimized - pos1_optimized)
            access_distances_optimized.append(dist_optimized)

        avg_distance_original = np.mean(access_distances_original)
        avg_distance_optimized = np.mean(access_distances_optimized)

        # Cache line analysis (assuming 64-byte cache lines, 4-byte floats = 16 elements)
        cache_line_size = 16
        cache_misses_original = sum(
            d > cache_line_size for d in access_distances_original
        )
        cache_misses_optimized = sum(
            d > cache_line_size for d in access_distances_optimized
        )

        return {
            "sequence_length": seq_len,
            "dilation_rate": self.current_dilation_rate,
            "avg_access_distance_original": avg_distance_original,
            "avg_access_distance_optimized": avg_distance_optimized,
            "improvement_factor": avg_distance_original / avg_distance_optimized,
            "cache_misses_original": cache_misses_original,
            "cache_misses_optimized": cache_misses_optimized,
            "cache_miss_reduction": (cache_misses_original - cache_misses_optimized)
            / cache_misses_original
            * 100,
        }


def create_memory_layout_optimized_attention(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparsity_ratio: float = 0.1,
    pattern_type: str = "dilated_sparse",
    block_size: int = 64,
    **kwargs,
) -> BlockSparseRingDilatedAttentionMemoryLayout:
    """
    Factory function for memory layout optimized attention.
    """
    sparse_config = SparsePatternConfig(
        pattern_type=pattern_type,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
    )

    return BlockSparseRingDilatedAttentionMemoryLayout(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        use_memory_layout_optimization=True,
        **kwargs,
    )
