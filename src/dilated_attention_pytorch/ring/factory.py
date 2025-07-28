"""Factory functions for creating ring attention variants."""

from typing import Optional, Literal, Dict, Any
import torch
import warnings

from .base.ring_config import RingAttentionConfig, get_preset_config
from .standard_ring_attention import StandardRingAttention
from .hilbert_ring_attention import HilbertRingAttention
from .distributed_ring_attention import DistributedRingAttention
from .block_sparse_ring_attention import BlockSparseRingAttention

# Type alias for implementation names
RingImplementation = Literal[
    "standard", "hilbert", "distributed", "block_sparse", "auto"
]


def create_ring_attention(
    implementation: RingImplementation = "auto",
    config: Optional[RingAttentionConfig] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> Any:
    """Create a ring attention module with the specified implementation.

    Args:
        implementation: Which implementation to use:
            - "standard": Basic ring attention
            - "hilbert": Ring attention with Hilbert curve optimization
            - "distributed": Enterprise features (DeepSpeed, monitoring)
            - "block_sparse": Combined with block-sparse patterns
            - "auto": Automatically select based on config
        config: Ring attention configuration (uses default if None)
        device: Device to place tensors on
        dtype: Data type for tensors
        **kwargs: Additional implementation-specific arguments

    Returns:
        Ring attention module instance

    Examples:
        >>> # Create standard ring attention
        >>> attention = create_ring_attention("standard")

        >>> # Create with custom config
        >>> config = RingAttentionConfig(segment_lengths=[2048, 4096])
        >>> attention = create_ring_attention("hilbert", config=config)

        >>> # Use preset configuration
        >>> attention = create_ring_attention(
        ...     "distributed",
        ...     config=get_preset_config("production")
        ... )
    """
    # Create default config if not provided
    if config is None:
        config = RingAttentionConfig()

    # Auto-select implementation based on config
    if implementation == "auto":
        implementation = _auto_select_implementation(config, **kwargs)

    # Create the appropriate implementation
    if implementation == "standard":
        return StandardRingAttention(config, device, dtype)

    elif implementation == "hilbert":
        return HilbertRingAttention(config, device, dtype)

    elif implementation == "distributed":
        # Check for optional features
        enable_deepspeed = kwargs.get("enable_deepspeed", True)
        enable_monitoring = kwargs.get("enable_monitoring", True)

        return DistributedRingAttention(
            config,
            device,
            dtype,
            enable_deepspeed=enable_deepspeed,
            enable_monitoring=enable_monitoring,
        )

    elif implementation == "block_sparse":
        # Get block-sparse specific parameters
        block_size = kwargs.get("block_size", 64)
        sparsity_ratio = kwargs.get("sparsity_ratio", 0.9)
        pattern_type = kwargs.get("pattern_type", "local")

        return BlockSparseRingAttention(
            config,
            block_size=block_size,
            sparsity_ratio=sparsity_ratio,
            pattern_type=pattern_type,
            device=device,
            dtype=dtype,
        )

    else:
        raise ValueError(
            f"Unknown implementation: {implementation}. "
            f"Choose from: standard, hilbert, distributed, block_sparse, auto"
        )


def _auto_select_implementation(
    config: RingAttentionConfig,
    **kwargs: Any,
) -> str:
    """Automatically select best implementation based on config and environment.

    Args:
        config: Ring attention configuration
        **kwargs: Additional parameters that might influence selection

    Returns:
        Selected implementation name
    """
    # Check if distributed training is active
    try:
        import torch.distributed as dist

        is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    except ImportError:
        is_distributed = False

    # Check for enterprise features
    has_deepspeed = False
    try:
        import deepspeed  # noqa: F401

        has_deepspeed = True
    except ImportError:
        pass

    # Decision logic
    if kwargs.get("sparsity_ratio") is not None:
        # User wants block-sparse
        return "block_sparse"

    elif config.use_hilbert:
        # Hilbert optimization requested
        return "hilbert"

    elif is_distributed and has_deepspeed:
        # Distributed environment with enterprise features
        return "distributed"

    else:
        # Default to standard implementation
        return "standard"


def get_available_implementations() -> Dict[str, bool]:
    """Get dictionary of available implementations and their status.

    Returns:
        Dictionary mapping implementation names to availability status
    """
    implementations = {
        "standard": True,  # Always available
        "hilbert": True,  # Always available
        "distributed": True,  # Base functionality always available
        "block_sparse": True,  # Always available
    }

    # Check optional features for distributed
    try:
        import deepspeed  # noqa: F401

        implementations["distributed_deepspeed"] = True
    except ImportError:
        implementations["distributed_deepspeed"] = False

    try:
        import wandb  # noqa: F401

        implementations["distributed_monitoring"] = True
    except ImportError:
        implementations["distributed_monitoring"] = False

    return implementations


def validate_ring_configuration(
    config: RingAttentionConfig,
    seq_len: int,
    world_size: Optional[int] = None,
) -> bool:
    """Validate that a configuration is compatible with the given constraints.

    Args:
        config: Ring attention configuration
        seq_len: Sequence length to validate
        world_size: Number of GPUs (if distributed)

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid with details
    """
    # Check segment lengths
    max_segment = max(config.segment_lengths)
    if seq_len % max_segment != 0:
        raise ValueError(
            f"Sequence length {seq_len} must be divisible by "
            f"largest segment length {max_segment}"
        )

    # Check world size compatibility
    if world_size is not None and world_size > 1:
        if seq_len % world_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} must be divisible by "
                f"world size {world_size} for ring attention"
            )

    # Check segment/dilation compatibility
    if len(config.segment_lengths) != len(config.dilation_rates):
        raise ValueError(
            f"segment_lengths and dilation_rates must have same length, "
            f"got {len(config.segment_lengths)} and {len(config.dilation_rates)}"
        )

    return True


# Convenience functions for common use cases
def create_ring_attention_from_preset(
    preset: str,
    implementation: RingImplementation = "auto",
    **kwargs: Any,
) -> Any:
    """Create ring attention using a preset configuration.

    Args:
        preset: Preset name ("development", "production", "large_scale")
        implementation: Which implementation to use
        **kwargs: Override preset parameters or add implementation-specific args

    Returns:
        Ring attention module instance
    """
    config = get_preset_config(preset)

    # Apply any config overrides
    for key in ["segment_lengths", "dilation_rates", "dropout"]:
        if key in kwargs:
            setattr(config, key, kwargs.pop(key))

    return create_ring_attention(implementation, config, **kwargs)


# Deprecation warnings for old implementations
def _create_deprecated_wrapper(old_name: str, new_impl: str):
    """Create a wrapper that warns about deprecation."""

    def deprecated_factory(*args, **kwargs):
        warnings.warn(
            f"{old_name} is deprecated and will be removed in v0.5.0. "
            f"Use create_ring_attention('{new_impl}') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return create_ring_attention(new_impl, *args, **kwargs)

    return deprecated_factory


# Create deprecated aliases
create_ring_dilated_attention_v2 = _create_deprecated_wrapper(
    "create_ring_dilated_attention_v2", "standard"
)
create_ring_dilated_attention_hilbert_fixed = _create_deprecated_wrapper(
    "create_ring_dilated_attention_hilbert_fixed", "hilbert"
)
