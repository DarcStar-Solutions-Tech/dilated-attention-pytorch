#!/usr/bin/env python3
"""
Configuration strategies for attention kernels.

This module extracts the complex configuration logic from the Enhanced kernel
into composable, testable strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptimizationLevel(Enum):
    """Optimization level for attention computation."""

    NONE = 0  # No optimizations
    BASIC = 1  # Standard optimizations
    AGGRESSIVE = 2  # All optimizations including special cases


@dataclass
class BlockConfig:
    """Block configuration for Triton kernels."""

    block_m: int
    block_n: int
    block_d: int
    num_warps: int


@dataclass
class AttentionConfig:
    """Complete attention configuration."""

    block_config: BlockConfig
    use_multi_row: bool = False
    rows_per_block: int = 1
    enable_prefetch: bool = False
    # Note: use_fused_softmax removed after normalization fix


class AttentionConstants:
    """Constants for attention computation."""

    # Block sizes
    BLOCK_SIZE_TINY = 32
    BLOCK_SIZE_SMALL = 64
    BLOCK_SIZE_MEDIUM = 96
    BLOCK_SIZE_LARGE = 128

    # Warp counts
    WARPS_MIN = 2
    WARPS_DEFAULT = 4
    WARPS_LARGE = 6
    WARPS_MAX = 8

    # Sequence thresholds
    SEQ_PYTORCH_THRESHOLD = 512
    SEQ_SMALL = 1024
    SEQ_MEDIUM = 2048
    SEQ_LARGE = 4096
    SEQ_XLARGE = 8192
    SEQ_XXLARGE = 10240

    # Sparsity thresholds
    SPARSITY_VERY_HIGH = 0.75  # d >= 4
    SPARSITY_HIGH = 0.5  # d = 2

    # GPU architecture
    COMPUTE_CAP_PASCAL = 6
    COMPUTE_CAP_VOLTA = 7
    COMPUTE_CAP_AMPERE = 8
    COMPUTE_CAP_HOPPER = 9


class ConfigStrategy(ABC):
    """Abstract base class for configuration strategies."""

    @abstractmethod
    def get_config(
        self,
        seq_len: int,
        head_dim: int,
        effective_len: Optional[int] = None,
        sparsity: Optional[float] = None,
    ) -> AttentionConfig:
        """Get configuration for given parameters."""
        pass


class DenseConfigStrategy(ConfigStrategy):
    """Configuration strategy for dense attention patterns."""

    def __init__(self, compute_capability: int, optimization_level: OptimizationLevel):
        self.compute_capability = compute_capability
        self.optimization_level = optimization_level
        self.is_pascal = compute_capability < AttentionConstants.COMPUTE_CAP_VOLTA

    def get_config(
        self,
        seq_len: int,
        head_dim: int,
        effective_len: Optional[int] = None,
        sparsity: Optional[float] = None,
    ) -> AttentionConfig:
        """Get configuration for dense attention."""

        if self.is_pascal:
            return self._get_pascal_config(seq_len, head_dim)
        else:
            return self._get_volta_plus_config(seq_len, head_dim)

    def _get_pascal_config(self, seq_len: int, head_dim: int) -> AttentionConfig:
        """Pascal GPU configuration (limited shared memory)."""

        if seq_len <= AttentionConstants.SEQ_SMALL:
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_TINY,
                block_n=AttentionConstants.BLOCK_SIZE_TINY,
                block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                num_warps=AttentionConstants.WARPS_MIN,
            )
        else:
            # Pascal struggles with larger blocks
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                block_n=AttentionConstants.BLOCK_SIZE_SMALL,
                block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                num_warps=AttentionConstants.WARPS_DEFAULT,
            )

        # Enable multi-row for larger sequences on Pascal too
        use_multi_row = (
            seq_len >= AttentionConstants.SEQ_LARGE
            and self.optimization_level != OptimizationLevel.NONE
        )

        return AttentionConfig(
            block_config=block_config,
            use_multi_row=use_multi_row,
            rows_per_block=2 if use_multi_row else 1,
            enable_prefetch=False,
        )

    def _get_volta_plus_config(self, seq_len: int, head_dim: int) -> AttentionConfig:
        """Volta+ GPU configuration (more shared memory)."""

        if seq_len <= AttentionConstants.SEQ_SMALL:
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                block_n=AttentionConstants.BLOCK_SIZE_SMALL,
                block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                num_warps=AttentionConstants.WARPS_DEFAULT,
            )
        elif seq_len <= AttentionConstants.SEQ_LARGE:
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_LARGE,
                block_n=AttentionConstants.BLOCK_SIZE_LARGE,
                block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                num_warps=AttentionConstants.WARPS_MAX,
            )
        elif (
            seq_len == AttentionConstants.SEQ_XLARGE
            and self.optimization_level == OptimizationLevel.AGGRESSIVE
        ):
            # Special 8K optimization
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                block_n=AttentionConstants.BLOCK_SIZE_LARGE,
                block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                num_warps=AttentionConstants.WARPS_MAX,
            )
        elif AttentionConstants.SEQ_XLARGE < seq_len <= AttentionConstants.SEQ_XXLARGE:
            # 8K-10K range
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_MEDIUM,
                block_n=AttentionConstants.BLOCK_SIZE_MEDIUM,
                block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                num_warps=AttentionConstants.WARPS_MAX,
            )
        else:
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_LARGE,
                block_n=AttentionConstants.BLOCK_SIZE_LARGE,
                block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                num_warps=AttentionConstants.WARPS_MAX,
            )

        # Enable multi-row for medium sequences on Volta+
        use_multi_row = (
            seq_len >= AttentionConstants.SEQ_LARGE
            and self.optimization_level != OptimizationLevel.NONE
        )

        return AttentionConfig(
            block_config=block_config,
            use_multi_row=use_multi_row,
            rows_per_block=2 if use_multi_row else 1,
            enable_prefetch=seq_len >= AttentionConstants.SEQ_MEDIUM,
        )


class SparseConfigStrategy(ConfigStrategy):
    """Configuration strategy for sparse attention patterns."""

    def __init__(self, compute_capability: int, optimization_level: OptimizationLevel):
        self.compute_capability = compute_capability
        self.optimization_level = optimization_level
        self.is_pascal = compute_capability < AttentionConstants.COMPUTE_CAP_VOLTA

    def get_config(
        self,
        seq_len: int,
        head_dim: int,
        effective_len: Optional[int] = None,
        sparsity: Optional[float] = None,
    ) -> AttentionConfig:
        """Get configuration for sparse attention."""

        if effective_len is None or sparsity is None:
            raise ValueError("Sparse config requires effective_len and sparsity")

        # Special 4K optimizations
        if (
            seq_len == AttentionConstants.SEQ_LARGE
            and self.optimization_level == OptimizationLevel.AGGRESSIVE
        ):
            config = self._get_4k_sparse_config(effective_len, head_dim)
            if config:
                return config

        # General sparse configuration
        if effective_len <= AttentionConstants.SEQ_PYTORCH_THRESHOLD:
            block_config = BlockConfig(
                block_m=AttentionConstants.BLOCK_SIZE_TINY,
                block_n=AttentionConstants.BLOCK_SIZE_TINY,
                block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                num_warps=AttentionConstants.WARPS_MIN,
            )
        elif effective_len <= AttentionConstants.SEQ_MEDIUM:
            if sparsity >= AttentionConstants.SPARSITY_VERY_HIGH:
                # Very sparse - small blocks
                block_config = BlockConfig(
                    block_m=AttentionConstants.BLOCK_SIZE_TINY,
                    block_n=AttentionConstants.BLOCK_SIZE_TINY,
                    block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                    num_warps=AttentionConstants.WARPS_MIN,
                )
            else:
                # Moderately sparse - asymmetric blocks
                block_config = BlockConfig(
                    block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                    block_n=AttentionConstants.BLOCK_SIZE_TINY,
                    block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                    num_warps=AttentionConstants.WARPS_DEFAULT,
                )
        else:
            # Large sparse sequences
            if sparsity >= AttentionConstants.SPARSITY_VERY_HIGH:
                block_config = BlockConfig(
                    block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                    block_n=AttentionConstants.BLOCK_SIZE_TINY,
                    block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                    num_warps=AttentionConstants.WARPS_DEFAULT,
                )
            else:
                if self.is_pascal:
                    block_config = BlockConfig(
                        block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                        block_n=AttentionConstants.BLOCK_SIZE_SMALL,
                        block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                        num_warps=AttentionConstants.WARPS_DEFAULT,
                    )
                else:
                    block_config = BlockConfig(
                        block_m=AttentionConstants.BLOCK_SIZE_MEDIUM,
                        block_n=AttentionConstants.BLOCK_SIZE_SMALL,
                        block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                        num_warps=AttentionConstants.WARPS_LARGE,
                    )

        # No multi-row or prefetch for sparse
        return AttentionConfig(
            block_config=block_config,
            use_multi_row=False,
            rows_per_block=1,
            enable_prefetch=False,
        )

    def _get_4k_sparse_config(
        self, effective_len: int, head_dim: int
    ) -> Optional[AttentionConfig]:
        """Special configurations for 4K sparse patterns."""

        if effective_len == 1024:  # 4K d=4
            return AttentionConfig(
                block_config=BlockConfig(
                    block_m=AttentionConstants.BLOCK_SIZE_TINY,
                    block_n=AttentionConstants.BLOCK_SIZE_TINY,
                    block_d=min(AttentionConstants.BLOCK_SIZE_TINY, head_dim),
                    num_warps=AttentionConstants.WARPS_MIN,
                ),
                use_multi_row=False,
                rows_per_block=1,
                enable_prefetch=False,
            )
        elif effective_len == 2048:  # 4K d=2
            return AttentionConfig(
                block_config=BlockConfig(
                    block_m=AttentionConstants.BLOCK_SIZE_SMALL,
                    block_n=AttentionConstants.BLOCK_SIZE_SMALL,
                    block_d=min(AttentionConstants.BLOCK_SIZE_SMALL, head_dim),
                    num_warps=AttentionConstants.WARPS_DEFAULT,
                ),
                use_multi_row=False,
                rows_per_block=1,
                enable_prefetch=False,
            )
        return None


class ConfigStrategyFactory:
    """Factory for creating configuration strategies."""

    @staticmethod
    def create_strategy(
        is_sparse: bool,
        compute_capability: int,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
    ) -> ConfigStrategy:
        """Create appropriate configuration strategy."""

        if is_sparse:
            return SparseConfigStrategy(compute_capability, optimization_level)
        else:
            return DenseConfigStrategy(compute_capability, optimization_level)
