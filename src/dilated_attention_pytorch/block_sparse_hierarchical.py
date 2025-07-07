"""
Block-Sparse Hierarchical Attention Patterns

This module implements hierarchical attention patterns that provide multi-scale
coverage with different levels of granularity:

1. Fine-grained local attention (high resolution, small receptive field)
2. Medium-grained regional attention (medium resolution, medium receptive field)
3. Coarse-grained global attention (low resolution, large receptive field)

The hierarchical approach allows efficient capture of both local and global
dependencies while maintaining sparsity.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
from torch import Tensor

from .block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical attention patterns."""

    # Level configurations (from fine to coarse)
    level_configs: List[Dict[str, int]] = None

    # Example default 3-level hierarchy:
    # Level 0: Fine-grained local attention (every position attends to 256 neighbors)
    # Level 1: Medium regional attention (every 4th position attends to 1024 positions)
    # Level 2: Coarse global attention (every 16th position attends to all positions)

    def __post_init__(self):
        if self.level_configs is None:
            self.level_configs = [
                {
                    "stride": 64,
                    "window_size": 256,
                    "block_size": 64,
                },  # Fine - every 64 tokens
                {
                    "stride": 256,
                    "window_size": 1024,
                    "block_size": 128,
                },  # Medium - every 256 tokens
                {
                    "stride": 1024,
                    "window_size": -1,
                    "block_size": 256,
                },  # Coarse - every 1024 tokens (-1 = global)
            ]

    @property
    def num_levels(self) -> int:
        return len(self.level_configs)


class BlockSparseHierarchical(BlockSparseRingDilatedAttention):
    """
    Block-Sparse attention with hierarchical multi-scale patterns.

    This implementation extends BlockSparseRingDilatedAttention with hierarchical attention
    patterns that provide different levels of granularity for capturing both
    local and global dependencies efficiently.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        hierarchical_config: Optional[HierarchicalConfig] = None,
        sparse_config: Optional[SparsePatternConfig] = None,
        **kwargs,
    ):
        """Initialize hierarchical block-sparse attention."""
        # Initialize parent with default sparse config
        if sparse_config is None:
            sparse_config = SparsePatternConfig(
                pattern_type="hierarchical",
                sparsity_ratio=0.1,  # Will be overridden by hierarchical pattern
                block_size=64,
            )

        super().__init__(segment_lengths, dilation_rates, sparse_config, **kwargs)

        self.hierarchical_config = hierarchical_config or HierarchicalConfig()

        # Override pattern generation
        self.sparse_config.pattern_type = "hierarchical"

    def _generate_hierarchical_pattern(
        self, seq_len: int, num_heads: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate hierarchical attention pattern with multi-scale coverage.

        Returns:
            row_indices: Row indices of active blocks
            col_indices: Column indices of active blocks
        """
        active_blocks = set()
        num_blocks = seq_len // self.block_size

        for level_idx, level_config in enumerate(
            self.hierarchical_config.level_configs
        ):
            stride = level_config["stride"]
            window_size = level_config["window_size"]

            # Positions that participate at this level (in terms of blocks, not tokens)
            # Convert stride from token-level to block-level
            block_stride = max(1, stride // self.block_size)
            active_block_positions = list(range(0, num_blocks, block_stride))

            # For each active block at this level
            for block_pos in active_block_positions:
                # Determine attention window in blocks
                if window_size == -1:  # Global attention
                    start_block = 0
                    end_block = num_blocks
                else:
                    # Convert window size from tokens to blocks
                    window_blocks = max(1, window_size // self.block_size)
                    half_window_blocks = window_blocks // 2

                    start_block = max(0, block_pos - half_window_blocks)
                    end_block = min(num_blocks, block_pos + half_window_blocks + 1)

                # Add block pairs for this level
                for target_block in range(start_block, end_block):
                    active_blocks.add((block_pos, target_block))

        # Convert to tensors
        if active_blocks:
            indices = torch.tensor(list(active_blocks), device=device, dtype=torch.long)
            row_indices = indices[:, 0]
            col_indices = indices[:, 1]
        else:
            # Fallback to at least diagonal blocks
            row_indices = torch.arange(num_blocks, device=device, dtype=torch.long)
            col_indices = torch.arange(num_blocks, device=device, dtype=torch.long)

        return row_indices, col_indices

    def _get_sparse_block_indices(
        self, num_blocks: int, num_heads: int, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        """Override parent's pattern generation with hierarchical patterns."""
        # Create cache key - hierarchical patterns are independent of heads
        cache_key = (num_blocks, num_heads, device.type, device.index, "hierarchical")

        # Define generator function for hierarchical pattern
        def generator():
            seq_len = num_blocks * self.block_size
            return self._generate_hierarchical_pattern(seq_len, num_heads, device)

        # Use parent's pattern cache
        return self.pattern_cache.get(cache_key, device, generator)

    def get_pattern_stats(self, seq_len: int) -> Dict[str, any]:
        """Get statistics about the hierarchical pattern."""
        device = torch.device("cpu")
        row_indices, col_indices = self._generate_hierarchical_pattern(
            seq_len, 1, device
        )

        num_blocks = seq_len // self.block_size
        total_blocks = num_blocks * num_blocks
        active_blocks = len(row_indices)
        sparsity = 1.0 - (active_blocks / total_blocks)

        # Analyze coverage by level
        level_stats = []
        for level_idx, level_config in enumerate(
            self.hierarchical_config.level_configs
        ):
            stride = level_config["stride"]
            window_size = level_config["window_size"]

            active_positions = len(range(0, seq_len, stride))
            coverage = active_positions / seq_len

            level_stats.append(
                {
                    "level": level_idx,
                    "stride": stride,
                    "window_size": window_size,
                    "active_positions": active_positions,
                    "coverage": coverage,
                }
            )

        return {
            "total_blocks": total_blocks,
            "active_blocks": active_blocks,
            "sparsity": sparsity,
            "levels": level_stats,
            "memory_reduction": sparsity,
        }

    def visualize_pattern(self, seq_len: int = 1024) -> str:
        """Create ASCII visualization of hierarchical pattern."""
        num_blocks = seq_len // self.block_size

        # Create pattern matrix
        pattern = [[" " for _ in range(num_blocks)] for _ in range(num_blocks)]

        # Generate pattern
        device = torch.device("cpu")
        row_indices, col_indices = self._generate_hierarchical_pattern(
            seq_len, 1, device
        )

        # Mark active blocks with different symbols for each level
        symbols = ["▪", "▫", "◦"]  # Different symbols for different levels

        for level_idx, level_config in enumerate(
            self.hierarchical_config.level_configs
        ):
            stride = level_config["stride"]
            window_size = level_config["window_size"]
            symbol = symbols[min(level_idx, len(symbols) - 1)]

            active_positions = list(range(0, seq_len, stride))

            for pos in active_positions:
                if window_size == -1:
                    start = 0
                    end = seq_len
                else:
                    half_window = window_size // 2
                    start = max(0, pos - half_window)
                    end = min(seq_len, pos + half_window)

                pos_block = pos // self.block_size
                start_block = start // self.block_size
                end_block = (end - 1) // self.block_size + 1

                for target_block in range(start_block, end_block):
                    if pos_block < num_blocks and target_block < num_blocks:
                        # Only update if not already marked by finer level
                        if pattern[pos_block][target_block] == " ":
                            pattern[pos_block][target_block] = symbol

        # Convert to string
        lines = ["Hierarchical Attention Pattern:"]
        lines.append("Legend: ▪=Fine, ▫=Medium, ◦=Coarse")
        lines.append("─" * (num_blocks + 2))

        for row in pattern:
            lines.append("│" + "".join(row) + "│")

        lines.append("─" * (num_blocks + 2))

        # Add statistics
        stats = self.get_pattern_stats(seq_len)
        lines.append(f"Sparsity: {stats['sparsity']:.1%}")
        lines.append(f"Active blocks: {stats['active_blocks']}/{stats['total_blocks']}")

        return "\n".join(lines)


def create_hierarchical_attention(
    embed_dim: int,
    num_heads: int,
    segment_lengths: List[int] = None,
    dilation_rates: List[int] = None,
    hierarchical_config: Optional[HierarchicalConfig] = None,
    **kwargs,
) -> BlockSparseHierarchical:
    """
    Create hierarchical block-sparse attention module.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates for dilated attention
        hierarchical_config: Configuration for hierarchical levels
        **kwargs: Additional arguments passed to BlockSparseHierarchical

    Returns:
        BlockSparseHierarchical module
    """
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    return BlockSparseHierarchical(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        hierarchical_config=hierarchical_config,
        **kwargs,
    )


# Example usage patterns
def get_hierarchical_presets() -> Dict[str, HierarchicalConfig]:
    """Get preset hierarchical configurations for common use cases."""
    return {
        "standard": HierarchicalConfig(),  # Default 3-level
        "fine_grained": HierarchicalConfig(
            level_configs=[
                {"stride": 1, "window_size": 128, "block_size": 32},
                {"stride": 2, "window_size": 512, "block_size": 64},
                {"stride": 8, "window_size": 2048, "block_size": 128},
                {"stride": 32, "window_size": -1, "block_size": 256},
            ]
        ),
        "long_range": HierarchicalConfig(
            level_configs=[
                {"stride": 1, "window_size": 512, "block_size": 128},
                {"stride": 8, "window_size": 4096, "block_size": 256},
                {"stride": 64, "window_size": -1, "block_size": 512},
            ]
        ),
        "ultra_sparse": HierarchicalConfig(
            level_configs=[
                {"stride": 1, "window_size": 64, "block_size": 16},
                {"stride": 16, "window_size": 1024, "block_size": 128},
                {"stride": 256, "window_size": -1, "block_size": 512},
            ]
        ),
    }
