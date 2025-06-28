"""
Configuration classes for Dilated Attention implementations.

This module provides dataclasses for configuring various dilated attention
implementations, ensuring type safety and validation of parameters.
"""

from dataclasses import dataclass

import torch

from ..utils.validation import ValidationMixin


@dataclass
class DilatedAttentionConfig(ValidationMixin):
    """
    Base configuration for dilated attention mechanisms.

    This configuration class defines the core parameters needed for all
    dilated attention implementations.

    Args:
        segment_lengths: List of segment lengths for each attention group
        dilation_rates: List of dilation rates corresponding to each segment
        dropout: Dropout probability (default: 0.0)
        use_tf32: Whether to use TF32 for matmul ops on Ampere GPUs (default: True)
        device: Device to place tensors on (default: auto-detect)
        dtype: Data type for tensors (default: torch.float32)

    Example:
        >>> config = DilatedAttentionConfig(
        ...     segment_lengths=[2048, 4096, 8192],
        ...     dilation_rates=[1, 2, 4],
        ...     dropout=0.1
        ... )
    """

    segment_lengths: list[int]
    dilation_rates: list[int]
    dropout: float = 0.0
    use_tf32: bool = True
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate segment lengths and dilation rates
        self.validate_segment_dilation_match(self.segment_lengths, self.dilation_rates)

        # Validate all values are positive
        self.validate_positive_values(self.segment_lengths, "segment_lengths")
        self.validate_positive_values(self.dilation_rates, "dilation_rates")

        # Validate dropout
        self.validate_dropout_prob(self.dropout)

        # Convert device if string
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default dtype
        if self.dtype is None:
            self.dtype = torch.float32

    @property
    def num_groups(self) -> int:
        """Number of attention groups."""
        return len(self.segment_lengths)

    @property
    def max_segment_length(self) -> int:
        """Maximum segment length."""
        return max(self.segment_lengths)


@dataclass
class MultiheadConfig(ValidationMixin):
    """
    Configuration for multihead attention components.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        bias: Whether to use bias in projections (default: True)
        layer_norm: Whether to use layer normalization (default: True)
        layer_norm_eps: Epsilon for layer normalization (default: 1e-5)
        gamma_init: Initialization scale for MAGNETO (default: 1.0)
        device: Device for parameters (default: auto-detect)
        dtype: Data type for parameters (default: torch.float32)
    """

    embed_dim: int
    num_heads: int
    bias: bool = True
    layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    gamma_init: float = 1.0
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        """Validate multihead configuration."""
        # Validate embed_dim is positive
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")

        # Validate and compute head dimension
        self.head_dim = self.validate_head_dim(self.embed_dim, self.num_heads)

        # Validate layer norm epsilon
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be positive, got {self.layer_norm_eps}"
            )

        # Validate gamma init
        if self.gamma_init <= 0:
            raise ValueError(f"gamma_init must be positive, got {self.gamma_init}")

        # Convert device if string
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        elif self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default dtype
        if self.dtype is None:
            self.dtype = torch.float32


@dataclass
class RingAttentionConfig(DilatedAttentionConfig):
    """
    Configuration for ring attention implementations.

    Extends base configuration with ring-specific parameters.

    Additional Args:
        block_size: Block size for ring attention computation (default: 1024)
        ring_size: Number of devices in ring (default: auto-detect)
        use_checkpointing: Whether to use gradient checkpointing (default: True)
        use_memory_pool: Whether to use memory pooling (default: True)
        max_cached_buffers: Maximum number of cached buffers (default: 50)
    """

    block_size: int = 1024
    ring_size: int | None = None
    use_checkpointing: bool = True
    use_memory_pool: bool = True
    max_cached_buffers: int = 50

    def __post_init__(self) -> None:
        """Validate ring attention configuration."""
        super().__post_init__()

        # Validate block size
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")

        # Validate ring size if provided
        if self.ring_size is not None and self.ring_size <= 0:
            raise ValueError(f"ring_size must be positive, got {self.ring_size}")

        # Validate max cached buffers
        if self.max_cached_buffers < 0:
            raise ValueError(
                f"max_cached_buffers must be non-negative, got {self.max_cached_buffers}"
            )


@dataclass
class SparseAttentionConfig(DilatedAttentionConfig):
    """
    Configuration for sparse attention patterns.

    Extends base configuration with sparsity parameters.

    Additional Args:
        pattern_type: Type of sparse pattern (default: "dilated_sparse")
        sparsity_ratio: Fraction of connections to keep (default: 0.25)
        block_size: Block size for sparse patterns (default: 128)
        enable_adaptive: Whether to use adaptive sparsity (default: False)
        min_sparsity: Minimum sparsity ratio (default: 0.1)
        max_sparsity: Maximum sparsity ratio (default: 0.9)
    """

    pattern_type: str = "dilated_sparse"
    sparsity_ratio: float = 0.25
    block_size: int = 128
    enable_adaptive: bool = False
    min_sparsity: float = 0.1
    max_sparsity: float = 0.9

    def __post_init__(self) -> None:
        """Validate sparse attention configuration."""
        super().__post_init__()

        # Validate pattern type
        valid_patterns = {
            "dilated_sparse",
            "local_window",
            "global_local",
            "axial",
            "random",
            "learned",
            "hybrid",
        }
        if self.pattern_type not in valid_patterns:
            raise ValueError(
                f"Invalid pattern_type '{self.pattern_type}'. Must be one of: {valid_patterns}"
            )

        # Validate sparsity ratios
        for name, value in [
            ("sparsity_ratio", self.sparsity_ratio),
            ("min_sparsity", self.min_sparsity),
            ("max_sparsity", self.max_sparsity),
        ]:
            if not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1 (exclusive), got {value}"
                )

        # Validate sparsity range
        if self.min_sparsity >= self.max_sparsity:
            raise ValueError(
                f"min_sparsity ({self.min_sparsity}) must be less than "
                f"max_sparsity ({self.max_sparsity})"
            )

        # Validate block size
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training features.

    Args:
        world_size: Total number of processes (default: auto-detect)
        rank: Rank of current process (default: auto-detect)
        backend: Distributed backend (default: "nccl")
        sequence_parallel: Enable sequence parallelism (default: False)
        model_parallel: Enable model parallelism (default: False)
        pipeline_parallel: Enable pipeline parallelism (default: False)
        gradient_checkpointing: Enable gradient checkpointing (default: False)
        communication_optimization: Enable communication optimizations (default: True)
        bucket_size_mb: Gradient bucket size in MB (default: 25)
        enable_async_communication: Enable async communication (default: True)
        gradient_compression_ratio: Compression ratio for gradients (default: None)
    """

    world_size: int | None = None
    rank: int | None = None
    backend: str = "nccl"
    sequence_parallel: bool = False
    model_parallel: bool = False
    pipeline_parallel: bool = False
    gradient_checkpointing: bool = False
    communication_optimization: bool = True
    bucket_size_mb: int = 25
    enable_async_communication: bool = True
    gradient_compression_ratio: float | None = None
    zero_stage: int = 0  # DeepSpeed ZeRO optimization stage (0-3)

    def __post_init__(self) -> None:
        """Validate distributed configuration."""
        # Validate backend
        valid_backends = {"nccl", "gloo", "mpi"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Invalid backend '{self.backend}'. Must be one of: {valid_backends}"
            )

        # Validate world size and rank
        if self.world_size is not None:
            if self.world_size <= 0:
                raise ValueError(f"world_size must be positive, got {self.world_size}")

        if self.rank is not None:
            if self.rank < 0:
                raise ValueError(f"rank must be non-negative, got {self.rank}")
            if self.world_size is not None and self.rank >= self.world_size:
                raise ValueError(
                    f"rank ({self.rank}) must be less than world_size ({self.world_size})"
                )

        # Validate bucket size
        if self.bucket_size_mb <= 0:
            raise ValueError(
                f"bucket_size_mb must be positive, got {self.bucket_size_mb}"
            )

        # Validate compression ratio
        if self.gradient_compression_ratio is not None:
            if not 0.0 < self.gradient_compression_ratio <= 1.0:
                raise ValueError(
                    f"gradient_compression_ratio must be between 0 and 1, "
                    f"got {self.gradient_compression_ratio}"
                )

        # Validate ZeRO stage
        if self.zero_stage not in {0, 1, 2, 3}:
            raise ValueError(f"zero_stage must be 0, 1, 2, or 3, got {self.zero_stage}")


@dataclass
class MemoryPoolConfig:
    """
    Configuration for unified memory pool.

    Args:
        hot_cache_size: Size of hot cache for frequent buffers (default: 50)
        hot_cache_threshold: Access count to promote to hot cache (default: 10)
        allow_buffer_slicing: Allow slicing larger buffers for smaller requests (default: True)
        cleanup_threshold_mb: Memory usage threshold for cleanup in MB (default: 100)
        max_pool_size_mb: Maximum pool size in MB (default: 1024)
        enable_statistics: Enable detailed statistics tracking (default: True)
        device: Default device for allocations (default: auto-detect)
    """

    hot_cache_size: int = 50
    hot_cache_threshold: int = 10
    allow_buffer_slicing: bool = True
    cleanup_threshold_mb: int = 100
    max_pool_size_mb: int = 1024
    enable_statistics: bool = True
    device: torch.device | None = None

    def __post_init__(self) -> None:
        """Validate memory pool configuration and set defaults."""
        # Validate sizes
        for name, value in [
            ("hot_cache_size", self.hot_cache_size),
            ("hot_cache_threshold", self.hot_cache_threshold),
            ("cleanup_threshold_mb", self.cleanup_threshold_mb),
            ("max_pool_size_mb", self.max_pool_size_mb),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # Set default device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
