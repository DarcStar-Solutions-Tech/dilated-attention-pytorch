"""
Factory pattern for creating dilated attention modules.

This module provides factory functions for creating various dilated attention
implementations with sensible defaults and automatic optimization.
"""

from typing import Optional, Dict, Type
import logging

from .config import (
    DilatedAttentionConfig,
    MultiheadConfig,
    RingAttentionConfig,
    SparseAttentionConfig,
)
from .base import BaseDilatedAttention, BaseMultiheadDilatedAttention
from .constants import HAS_FLASH_ATTN, HAS_FLASH_ATTN_3, GPU_TYPE

logger = logging.getLogger("dilated_attention_pytorch.factory")


# Registry for attention implementations
_ATTENTION_REGISTRY: Dict[str, Type[BaseDilatedAttention]] = {}
_MULTIHEAD_REGISTRY: Dict[str, Type[BaseMultiheadDilatedAttention]] = {}


def register_attention(name: str, cls: Type[BaseDilatedAttention]) -> None:
    """Register a dilated attention implementation."""
    _ATTENTION_REGISTRY[name] = cls


def register_multihead_attention(name: str, cls: Type[BaseMultiheadDilatedAttention]) -> None:
    """Register a multihead dilated attention implementation."""
    _MULTIHEAD_REGISTRY[name] = cls


def create_dilated_attention(
    attention_type: str = "auto",
    segment_lengths: Optional[list] = None,
    dilation_rates: Optional[list] = None,
    **kwargs,
) -> BaseDilatedAttention:
    """
    Create a dilated attention module.

    Args:
        attention_type: Type of attention to create. Options:
            - "auto": Automatically select best implementation
            - "standard": Standard dilated attention
            - "improved": Improved dilated attention with optimizations
            - "ring": Ring attention (O(n) memory)
            - "distributed": Distributed dilated attention
            - "block_sparse_ring": Block-sparse ring attention
        segment_lengths: List of segment lengths (default: [2048, 4096, 8192])
        dilation_rates: List of dilation rates (default: [1, 2, 4])
        **kwargs: Additional configuration parameters

    Returns:
        Dilated attention module

    Example:
        >>> attention = create_dilated_attention(
        ...     "improved",
        ...     segment_lengths=[1024, 2048],
        ...     dilation_rates=[1, 2],
        ...     dropout=0.1
        ... )
    """
    # Set defaults
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    # Auto-select implementation
    if attention_type == "auto":
        attention_type = _select_best_attention_type()

    # Validate type
    if attention_type not in _ATTENTION_REGISTRY:
        available = list(_ATTENTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown attention type '{attention_type}'. " f"Available types: {available}"
        )

    # Create configuration
    config_class = _get_config_class(attention_type)
    filtered_kwargs = _filter_kwargs(config_class, kwargs)
    config = config_class(segment_lengths=segment_lengths, dilation_rates=dilation_rates, **filtered_kwargs)

    # Create and return module
    cls = _ATTENTION_REGISTRY[attention_type]
    module = cls(config)

    logger.info(f"Created {attention_type} dilated attention module")
    return module


def create_multihead_dilated_attention(
    attention_type: str = "auto",
    embed_dim: int = 768,
    num_heads: int = 12,
    segment_lengths: Optional[list] = None,
    dilation_rates: Optional[list] = None,
    **kwargs,
) -> BaseMultiheadDilatedAttention:
    """
    Create a multihead dilated attention module.

    Args:
        attention_type: Type of attention to create (see create_dilated_attention)
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        segment_lengths: List of segment lengths
        dilation_rates: List of dilation rates
        **kwargs: Additional configuration parameters

    Returns:
        Multihead dilated attention module

    Example:
        >>> attention = create_multihead_dilated_attention(
        ...     "improved",
        ...     embed_dim=768,
        ...     num_heads=12,
        ...     dropout=0.1
        ... )
    """
    # Set defaults
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    # Auto-select implementation
    if attention_type == "auto":
        attention_type = _select_best_attention_type()

    # Map to multihead version
    multihead_type = f"multihead_{attention_type}"

    # Validate type
    if multihead_type not in _MULTIHEAD_REGISTRY:
        available = [t.replace("multihead_", "") for t in _MULTIHEAD_REGISTRY.keys()]
        raise ValueError(
            f"Unknown attention type '{attention_type}'. " f"Available types: {available}"
        )

    # Check if configs were passed directly
    multihead_config_passed = kwargs.get("multihead_config", None)
    attention_config_passed = kwargs.get("attention_config", None)
    
    if multihead_config_passed is not None:
        multihead_config = multihead_config_passed
        # Override with any direct parameters
        if embed_dim != 768:  # Not default
            multihead_config.embed_dim = embed_dim
        if num_heads != 12:  # Not default
            multihead_config.num_heads = num_heads
    else:
        # Separate multihead and attention configs
        multihead_kwargs = {}
        attention_kwargs = {}

        # Keywords that belong to multihead config
        multihead_keys = {"bias", "layer_norm", "layer_norm_eps", "gamma_init", "device", "dtype"}

        for key, value in kwargs.items():
            if key in multihead_keys:
                multihead_kwargs[key] = value
            else:
                attention_kwargs[key] = value

        # Create configurations
        filtered_multihead_kwargs = _filter_kwargs(MultiheadConfig, multihead_kwargs)
        multihead_config = MultiheadConfig(embed_dim=embed_dim, num_heads=num_heads, **filtered_multihead_kwargs)

    if attention_config_passed is not None:
        attention_config = attention_config_passed
        # Override with any direct parameters
        if segment_lengths != [2048, 4096, 8192]:  # Not default
            attention_config.segment_lengths = segment_lengths
        if dilation_rates != [1, 2, 4]:  # Not default
            attention_config.dilation_rates = dilation_rates
    else:
        config_class = _get_config_class(attention_type)
        filtered_attention_kwargs = _filter_kwargs(config_class, attention_kwargs)
        attention_config = config_class(
            segment_lengths=segment_lengths, dilation_rates=dilation_rates, **filtered_attention_kwargs
        )
    
    # Filter out config objects from kwargs to avoid duplication
    remaining_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ["multihead_config", "attention_config", "bias", "layer_norm", 
                                    "layer_norm_eps", "gamma_init", "device", "dtype", "dropout", 
                                    "segment_lengths", "dilation_rates", "embed_dim", "num_heads"]}

    # Create and return module
    cls = _MULTIHEAD_REGISTRY[multihead_type]
    
    # Handle legacy constructors for improved implementations
    if attention_type in ["improved", "improved_distributed"]:
        # These implementations haven't been fully refactored yet
        module = cls(
            embed_dim=multihead_config.embed_dim,
            num_heads=multihead_config.num_heads,
            dilation_rates=attention_config.dilation_rates,
            segment_lengths=attention_config.segment_lengths,
            dropout=attention_config.dropout,
            bias=multihead_config.bias,
            layer_norm=multihead_config.layer_norm,
            layer_norm_eps=multihead_config.layer_norm_eps,
            gamma_init=multihead_config.gamma_init,
            device=multihead_config.device,
            dtype=multihead_config.dtype,
            **remaining_kwargs,  # Pass through any extra kwargs
        )
    else:
        # New style with configs
        module = cls(multihead_config, attention_config)

    logger.info(f"Created {attention_type} multihead dilated attention module")
    return module


def create_block_sparse_attention(
    sparsity_ratio: float = 0.9,
    pattern_type: str = "dilated_sparse",
    embed_dim: int = 768,
    num_heads: int = 12,
    **kwargs,
) -> BaseMultiheadDilatedAttention:
    """
    Create a block-sparse dilated attention module.

    Args:
        sparsity_ratio: Fraction of connections to prune (0.9 = 90% sparse)
        pattern_type: Type of sparse pattern
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional configuration parameters

    Returns:
        Block-sparse multihead dilated attention module

    Example:
        >>> attention = create_block_sparse_attention(
        ...     sparsity_ratio=0.95,
        ...     pattern_type="global_local",
        ...     embed_dim=768,
        ...     num_heads=12
        ... )
    """
    # Force block-sparse type
    attention_type = "block_sparse_ring"

    # Add sparsity parameters
    kwargs.update({"sparsity_ratio": sparsity_ratio, "pattern_type": pattern_type})

    return create_multihead_dilated_attention(
        attention_type=attention_type, embed_dim=embed_dim, num_heads=num_heads, **kwargs
    )


def create_adaptive_sparse_attention(
    embed_dim: int = 768,
    num_heads: int = 12,
    min_sparsity: float = 0.1,
    max_sparsity: float = 0.9,
    **kwargs,
) -> BaseMultiheadDilatedAttention:
    """
    Create an adaptive sparse dilated attention module.

    The sparsity pattern is learned during training.

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        min_sparsity: Minimum sparsity ratio
        max_sparsity: Maximum sparsity ratio
        **kwargs: Additional configuration parameters

    Returns:
        Adaptive sparse multihead dilated attention module
    """
    # Force block-sparse type with adaptive enabled
    attention_type = "block_sparse_ring"

    # Add adaptive parameters
    kwargs.update(
        {
            "enable_adaptive": True,
            "min_sparsity": min_sparsity,
            "max_sparsity": max_sparsity,
            "pattern_type": "learned",
        }
    )

    return create_multihead_dilated_attention(
        attention_type=attention_type, embed_dim=embed_dim, num_heads=num_heads, **kwargs
    )


def _select_best_attention_type() -> str:
    """Automatically select the best attention type based on hardware."""
    gpu_type = str(GPU_TYPE)

    # H100/A100: Use most advanced implementation
    if gpu_type in ["h100", "a100"]:
        if HAS_FLASH_ATTN_3:
            return "improved"  # Best with FA3
        elif HAS_FLASH_ATTN:
            return "improved"  # Still good with FA2
        else:
            return "improved"  # Fallback to improved

    # V100 or older: Use simpler implementation
    elif gpu_type == "v100":
        return "improved"  # Use improved since standard has circular import

    # CPU: Use improved implementation
    elif gpu_type == "cpu":
        return "improved"  # Use improved since standard has circular import

    # Default: Use improved if Flash Attention available
    else:
        if HAS_FLASH_ATTN:
            return "improved"
        else:
            return "improved"  # Fallback to improved


def _filter_kwargs(config_class: Type, kwargs: dict) -> dict:
    """Filter kwargs to only include valid parameters for the config class."""
    import inspect
    from dataclasses import fields
    
    # Get valid field names for dataclass
    if hasattr(config_class, '__dataclass_fields__'):
        valid_fields = {f.name for f in fields(config_class)}
    else:
        # Fallback for non-dataclass configs
        sig = inspect.signature(config_class.__init__)
        valid_fields = set(sig.parameters.keys()) - {'self'}
    
    # Filter kwargs
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    
    # Log if any kwargs were filtered out
    filtered_out = set(kwargs.keys()) - set(filtered.keys())
    if filtered_out:
        logger.debug(f"Filtered out unknown kwargs: {filtered_out}")
    
    return filtered


def _get_config_class(attention_type: str) -> Type:
    """Get the appropriate config class for attention type."""
    config_mapping = {
        "standard": DilatedAttentionConfig,
        "improved": DilatedAttentionConfig,
        "ring": RingAttentionConfig,
        "distributed": DilatedAttentionConfig,
        "improved_distributed": DilatedAttentionConfig,
        "ring_distributed": RingAttentionConfig,
        "block_sparse_ring": SparseAttentionConfig,
        "block_sparse_ring_distributed": SparseAttentionConfig,
    }

    return config_mapping.get(attention_type, DilatedAttentionConfig)


# Import and register implementations when available
def _register_implementations():
    """Register all available implementations."""
    try:
        # Register standard implementations
        from ..dilated_attention import DilatedAttention
        from ..multihead_dilated_attention import MultiheadDilatedAttention

        register_attention("standard", DilatedAttention)
        register_multihead_attention("multihead_standard", MultiheadDilatedAttention)
        logger.debug("Registered standard dilated attention implementations")

    except ImportError as e:
        logger.warning(f"Failed to register standard implementations: {e}")

    try:
        # Register improved implementations
        from ..improved_dilated_attention import ImprovedDilatedAttention

        register_attention("improved", ImprovedDilatedAttention)
        logger.debug("Registered improved dilated attention implementation")

        from ..improved_multihead_dilated_attention import ImprovedMultiheadDilatedAttention

        register_multihead_attention("multihead_improved", ImprovedMultiheadDilatedAttention)
        logger.debug("Registered improved multihead dilated attention implementation")

    except ImportError as e:
        logger.warning(f"Failed to register improved implementations: {e}")

    try:
        # Register ring attention implementations
        from ..ring_dilated_attention import RingDilatedAttention

        register_attention("ring", RingDilatedAttention)
        logger.debug("Registered ring dilated attention implementation")

        try:
            from ..ring_multihead_dilated_attention import RingMultiheadDilatedAttention

            register_multihead_attention("multihead_ring", RingMultiheadDilatedAttention)
            logger.debug("Registered ring multihead dilated attention implementation")
        except ImportError:
            pass  # Multihead not refactored yet

    except ImportError as e:
        logger.warning(f"Failed to register ring implementations: {e}")

    try:
        # Register distributed implementations
        from ..improved_distributed_dilated_attention import DistributedImprovedDilatedAttention

        register_attention("distributed", DistributedImprovedDilatedAttention)
        register_attention("improved_distributed", DistributedImprovedDilatedAttention)
        logger.debug("Registered distributed dilated attention implementation")

        from ..improved_distributed_dilated_attention import (
            DistributedImprovedMultiheadDilatedAttention,
        )

        register_multihead_attention(
            "multihead_distributed", DistributedImprovedMultiheadDilatedAttention
        )
        register_multihead_attention(
            "multihead_improved_distributed", DistributedImprovedMultiheadDilatedAttention
        )
        logger.debug("Registered distributed multihead dilated attention implementation")

    except ImportError as e:
        logger.warning(f"Failed to register distributed implementations: {e}")

    try:
        # Register block-sparse implementations
        from ..block_sparse_ring_dilated_attention import BlockSparseRingDilatedAttention
        from ..block_sparse_ring_multihead_dilated_attention import (
            BlockSparseRingMultiheadDilatedAttention,
        )

        register_attention("block_sparse_ring", BlockSparseRingDilatedAttention)
        register_multihead_attention(
            "multihead_block_sparse_ring", BlockSparseRingMultiheadDilatedAttention
        )
        logger.debug("Registered block-sparse attention implementations")

    except ImportError:
        pass  # Block-sparse not refactored yet


# Initialize registry
_register_implementations()
