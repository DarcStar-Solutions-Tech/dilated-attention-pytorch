"""
Factory pattern for creating dilated attention modules.

This module provides factory functions for creating various dilated attention
implementations with sensible defaults and automatic optimization.
"""

import logging

from torch import nn

from .base import BaseDilatedAttention, BaseMultiheadDilatedAttention
from .config import (
    DilatedAttentionConfig,
    MultiheadConfig,
    RingAttentionConfig,
    SparseAttentionConfig,
)
from .constants import GPU_TYPE, HAS_FLASH_ATTN, HAS_FLASH_ATTN_3

logger = logging.getLogger("dilated_attention_pytorch.factory")


# Registry for attention implementations
_ATTENTION_REGISTRY: dict[str, type[BaseDilatedAttention]] = {}
_MULTIHEAD_REGISTRY: dict[str, type[BaseMultiheadDilatedAttention]] = {}


def register_attention(name: str, cls: type[BaseDilatedAttention]) -> None:
    """Register a dilated attention implementation."""
    _ATTENTION_REGISTRY[name] = cls


def register_multihead_attention(
    name: str, cls: type[BaseMultiheadDilatedAttention]
) -> None:
    """Register a multihead dilated attention implementation."""
    _MULTIHEAD_REGISTRY[name] = cls


def create_dilated_attention(
    attention_type: str = "auto",
    segment_lengths: list | None = None,
    dilation_rates: list | None = None,
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
    # Ensure implementations are registered
    _ensure_implementations_registered()

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
            f"Unknown attention type '{attention_type}'. Available types: {available}"
        )

    # Auto-enable memory pool based on sequence length and implementation
    _auto_configure_memory_pool(attention_type, segment_lengths, kwargs)

    # Create configuration
    config_class = _get_config_class(attention_type)
    filtered_kwargs = _filter_kwargs(config_class, kwargs)
    config = config_class(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        **filtered_kwargs,
    )

    # Create and return module
    cls = _ATTENTION_REGISTRY[attention_type]

    # Handle legacy constructors that don't accept config objects
    if attention_type in ["block_sparse_ring", "block_sparse_ring_distributed"]:
        # These implementations expect individual parameters
        module = cls(
            segment_lengths=config.segment_lengths,
            dilation_rates=config.dilation_rates,
            dropout=config.dropout,
            use_tf32=config.use_tf32,
            device=config.device,
            dtype=config.dtype,
            **kwargs,  # Pass through remaining kwargs
        )
    elif attention_type in ["standard", "improved"]:
        # These implementations need individual parameters + memory pool settings
        # Filter kwargs to only include what each implementation accepts
        if attention_type == "standard":
            module = cls(
                segment_lengths=config.segment_lengths,
                dilation_rates=config.dilation_rates,
                attention_dropout=config.dropout,
                **kwargs,  # This includes memory pool settings
            )
        else:  # improved
            module = cls(
                segment_lengths=config.segment_lengths,
                dilation_rates=config.dilation_rates,
                dropout=config.dropout,
                use_tf32=config.use_tf32,
                **kwargs,  # This includes memory pool settings
            )
    elif attention_type in ["distributed", "improved_distributed"]:
        # Distributed implementations need individual parameters
        # Filter out memory pool kwargs since distributed doesn't support them yet
        distributed_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in ["enable_memory_pool", "lightweight_pool", "enable_buffer_manager"]
        }
        module = cls(
            segment_lengths=config.segment_lengths,
            dilation_rates=config.dilation_rates,
            dropout=config.dropout,
            use_tf32=config.use_tf32,
            device=config.device,
            dtype=config.dtype,
            **distributed_kwargs,
        )
    else:
        module = cls(config)

    logger.info(f"Created {attention_type} dilated attention module")
    return module


def _select_best_multihead_attention_type() -> str:
    """Automatically select the best multihead attention type based on hardware."""
    gpu_type = str(GPU_TYPE)

    # H100/H800: Optimized for Flash Attention 3
    if gpu_type in ["h100", "h800"]:
        if HAS_FLASH_ATTN_3:
            logger.info(
                "H100/H800 detected with FA3 - using block_sparse_ring for maximum performance"
            )
            return "block_sparse_ring"  # Best with FA3 block-sparse optimizations
        elif HAS_FLASH_ATTN:
            return "improved"  # Still good with FA2
        else:
            return "improved"  # Fallback to improved

    # For other GPUs, use the same as base attention
    return _select_best_attention_type()


def create_multihead_dilated_attention(  # noqa: PLR0912
    attention_type: str = "auto",
    embed_dim: int = 768,
    num_heads: int = 12,
    segment_lengths: list | None = None,
    dilation_rates: list | None = None,
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
    # Ensure implementations are registered
    _ensure_implementations_registered()
    # Set defaults
    if segment_lengths is None:
        segment_lengths = [2048, 4096, 8192]
    if dilation_rates is None:
        dilation_rates = [1, 2, 4]

    # Auto-select implementation
    if attention_type == "auto":
        attention_type = _select_best_multihead_attention_type()

    # Map to multihead version
    multihead_type = f"multihead_{attention_type}"

    # Validate type
    if multihead_type not in _MULTIHEAD_REGISTRY:
        available = [t.replace("multihead_", "") for t in _MULTIHEAD_REGISTRY]
        raise ValueError(
            f"Unknown attention type '{attention_type}'. Available types: {available}"
        )

    # Auto-enable memory pool based on sequence length and implementation
    _auto_configure_memory_pool(attention_type, segment_lengths, kwargs)

    # Check if configs were passed directly
    multihead_config_passed = kwargs.get("multihead_config")
    attention_config_passed = kwargs.get("attention_config")

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
        multihead_keys = {
            "bias",
            "layer_norm",
            "layer_norm_eps",
            "gamma_init",
            "device",
            "dtype",
        }

        for key, value in kwargs.items():
            if key in multihead_keys:
                multihead_kwargs[key] = value
            else:
                attention_kwargs[key] = value

        # Create configurations
        filtered_multihead_kwargs = _filter_kwargs(MultiheadConfig, multihead_kwargs)
        multihead_config = MultiheadConfig(
            embed_dim=embed_dim, num_heads=num_heads, **filtered_multihead_kwargs
        )

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
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            **filtered_attention_kwargs,
        )

    # Filter out config objects from kwargs to avoid duplication
    remaining_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k
        not in [
            "multihead_config",
            "attention_config",
            "bias",
            "layer_norm",
            "layer_norm_eps",
            "gamma_init",
            "device",
            "dtype",
            "dropout",
            "segment_lengths",
            "dilation_rates",
            "embed_dim",
            "num_heads",
            # Memory pool settings are handled internally
            "enable_memory_pool",
            "lightweight_pool",
            "enable_buffer_manager",
        ]
    }

    # Create and return module
    cls = _MULTIHEAD_REGISTRY[multihead_type]

    # Handle legacy constructors for improved implementations
    if attention_type in ["improved", "improved_distributed"]:
        # These implementations haven't been fully refactored yet
        module = cls(
            embed_dim=multihead_config.embed_dim,
            num_heads=multihead_config.num_heads,
            segment_lengths=attention_config.segment_lengths,
            dilation_rates=attention_config.dilation_rates,
            dropout=attention_config.dropout,
            bias=multihead_config.bias,
            layer_norm=multihead_config.layer_norm,
            layer_norm_eps=multihead_config.layer_norm_eps,
            gamma_init=multihead_config.gamma_init,
            device=multihead_config.device,
            dtype=multihead_config.dtype,
            **remaining_kwargs,  # Pass through any extra kwargs
        )
    elif attention_type == "ring":
        # Ring attention expects individual parameters
        module = cls(
            embed_dim=multihead_config.embed_dim,
            num_heads=multihead_config.num_heads,
            segment_lengths=attention_config.segment_lengths,
            dilation_rates=attention_config.dilation_rates,
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
        attention_type=attention_type,
        embed_dim=embed_dim,
        num_heads=num_heads,
        **kwargs,
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
        attention_type=attention_type,
        embed_dim=embed_dim,
        num_heads=num_heads,
        **kwargs,
    )


def _select_best_attention_type() -> str:  # noqa: PLR0911, PLR0912
    """Automatically select the best attention type based on hardware."""
    gpu_type = str(GPU_TYPE)

    # H100/H800: Optimized for Flash Attention 3
    if gpu_type in ["h100", "h800"]:
        if HAS_FLASH_ATTN_3:
            logger.info(
                "H100/H800 detected with FA3 - using improved for base attention"
            )
            # Note: block_sparse_ring is only available for multihead attention
            # For base attention, use improved which has Flash Attention support
            return "improved"  # Use improved for base attention
        elif HAS_FLASH_ATTN:
            return "improved"  # Still good with FA2
        else:
            return "improved"  # Fallback to improved

    # A100: Use most advanced implementation
    elif gpu_type == "a100":
        if HAS_FLASH_ATTN_3:
            return "improved"  # FA3 available but not optimized for A100
        elif HAS_FLASH_ATTN:
            return "improved"  # Best with FA2
        else:
            return "improved"  # Fallback

    # V100 or older: Use simpler implementation
    elif gpu_type in ("v100", "cpu"):
        return "improved"  # Use improved since standard has circular import

    # AMD MI300/MI250: Check for ROCm optimizations
    elif gpu_type == "amd_instinct":
        if HAS_FLASH_ATTN:
            return "improved"
        else:
            return "improved"  # Standard PyTorch

    # Default: Use improved if Flash Attention available
    elif HAS_FLASH_ATTN:
        return "improved"
    else:
        return "improved"  # Fallback to improved


def _filter_kwargs(config_class: type, kwargs: dict) -> dict:
    """Filter kwargs to only include valid parameters for the config class."""
    import inspect
    from dataclasses import fields

    # Get valid field names for dataclass
    if hasattr(config_class, "__dataclass_fields__"):
        valid_fields = {f.name for f in fields(config_class)}
    else:
        # Fallback for non-dataclass configs
        sig = inspect.signature(config_class.__init__)
        valid_fields = set(sig.parameters.keys()) - {"self"}

    # Filter kwargs
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}

    # Log if any kwargs were filtered out
    filtered_out = set(kwargs.keys()) - set(filtered.keys())
    if filtered_out:
        logger.debug(f"Filtered out unknown kwargs: {filtered_out}")

    return filtered


def _get_config_class(attention_type: str) -> type:
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


def _auto_configure_memory_pool(
    attention_type: str, segment_lengths: list, kwargs: dict
) -> None:
    """
    Automatically configure memory pool settings based on implementation and sequence lengths.

    Args:
        attention_type: Type of attention implementation
        segment_lengths: List of segment lengths
        kwargs: Keyword arguments dict to update
    """
    # Don't override if user explicitly set memory pool settings
    if "enable_memory_pool" in kwargs or "enable_buffer_manager" in kwargs:
        return

    # Get maximum sequence length
    max_seq_len = max(segment_lengths)

    # Determine if memory pool would be beneficial
    # Memory pools are beneficial for:
    # 1. Long sequences (>= 4096 tokens)
    # 2. Ring attention (always benefits from memory pools)
    # 3. Distributed implementations (reduce allocation overhead)
    # 4. Block-sparse implementations (complex allocation patterns)

    should_enable_pool = (
        max_seq_len >= 4096
        or "ring" in attention_type
        or "distributed" in attention_type
        or "block_sparse" in attention_type
    )

    if should_enable_pool:
        # Choose appropriate memory pool settings based on implementation
        # Note: ImprovedDilatedAttentionV2 would use buffer manager, but we're using
        # ImprovedDilatedAttention from factory which uses memory pools
        kwargs["enable_memory_pool"] = True
        kwargs["lightweight_pool"] = (
            max_seq_len < 8192
        )  # Use lightweight for medium sequences
        logger.info(
            f"Auto-enabled memory pool for {attention_type} "
            f"(max seq length: {max_seq_len}, lightweight: {kwargs.get('lightweight_pool', False)})"
        )
    else:
        # Explicitly disable for short sequences to avoid overhead
        kwargs["enable_memory_pool"] = False
        logger.debug(
            f"Memory pool disabled for {attention_type} with short sequences "
            f"(max seq length: {max_seq_len})"
        )


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

        from ..improved_multihead_dilated_attention import (
            ImprovedMultiheadDilatedAttention,
        )

        register_multihead_attention(
            "multihead_improved", ImprovedMultiheadDilatedAttention
        )
        logger.debug("Registered improved multihead dilated attention implementation")

    except ImportError as e:
        logger.warning(f"Failed to register improved implementations: {e}")

    try:
        # Register corrected ring attention V2 implementations
        from ..ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )

        # Create a wrapper to match the expected interface
        class RingDilatedAttentionV2Wrapper(BaseDilatedAttention):
            """Wrapper to make RingDilatedAttentionV2Collective compatible with factory."""

            def __init__(self, config):
                super().__init__(config)
                self.ring_attention = RingDilatedAttentionV2Collective(
                    segment_lengths=config.segment_lengths,
                    dilation_rates=config.dilation_rates,
                    dropout=config.dropout,
                    ring_size=getattr(config, "ring_size", None),
                    device=config.device,
                    dtype=config.dtype,
                )

            def forward(self, query, key, value, is_causal=False, attention_mask=None):
                return self.ring_attention(query, key, value, is_causal, attention_mask)

        register_attention("ring", RingDilatedAttentionV2Wrapper)
        logger.debug(
            "Registered corrected ring dilated attention V2 collective implementation"
        )

        # Register the new RingMultiheadDilatedAttention
        from ..ring_multihead_dilated_attention import RingMultiheadDilatedAttention

        register_multihead_attention("multihead_ring", RingMultiheadDilatedAttention)
        logger.debug(
            "Registered corrected ring multihead dilated attention V2 implementation"
        )

    except ImportError as e:
        logger.error(f"Failed to register ring V2 implementations: {e}")
        logger.error("Ring attention is not available. Please check installation.")

    try:
        # Register ring distributed implementation
        from ..ring_distributed_refactored import RingDistributedDilatedAttention

        register_multihead_attention(
            "multihead_ring_distributed", RingDistributedDilatedAttention
        )
        logger.debug("Registered ring distributed dilated attention implementation")

    except ImportError as e:
        logger.warning(f"Failed to register ring distributed implementation: {e}")

    try:
        # Register distributed implementations
        from ..improved_distributed_dilated_attention import (
            DistributedImprovedDilatedAttention,
        )

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
            "multihead_improved_distributed",
            DistributedImprovedMultiheadDilatedAttention,
        )
        logger.debug(
            "Registered distributed multihead dilated attention implementation"
        )

    except ImportError as e:
        logger.warning(f"Failed to register distributed implementations: {e}")

    try:
        # Register block-sparse implementations
        from ..block_sparse_ring_dilated_attention import (
            BlockSparseRingDilatedAttention,
        )
        from ..block_sparse_ring_multihead_dilated_attention import (
            BlockSparseRingMultiheadDilatedAttention,
        )

        # Create wrapper for BlockSparse to work with config objects
        class BlockSparseWrapper(BaseMultiheadDilatedAttention):
            """Wrapper to make BlockSparse work with config pattern."""

            def __init__(self, multihead_config, attention_config):
                # Don't call super().__init__ since we're wrapping
                nn.Module.__init__(self)
                self.multihead_config = multihead_config
                self.attention_config = attention_config

                # Extract values from configs
                sparse_config = getattr(attention_config, "sparse_config", None)
                if sparse_config is None:
                    # Create sparse config from attention config
                    from ..block_sparse_ring_dilated_attention import (
                        SparsePatternConfig,
                    )

                    sparse_config = SparsePatternConfig(
                        pattern_type=getattr(
                            attention_config, "pattern_type", "dilated_sparse"
                        ),
                        sparsity_ratio=getattr(
                            attention_config, "sparsity_ratio", 0.25
                        ),
                        block_size=getattr(attention_config, "block_size", 128),
                    )

                # Create the actual implementation
                self._impl = BlockSparseRingMultiheadDilatedAttention(
                    embed_dim=multihead_config.embed_dim,
                    num_heads=multihead_config.num_heads,
                    segment_lengths=attention_config.segment_lengths,
                    dilation_rates=attention_config.dilation_rates,
                    sparse_config=sparse_config,
                    dropout=attention_config.dropout,
                    bias=multihead_config.bias,
                    device=multihead_config.device,
                    dtype=multihead_config.dtype,
                )

                # Expose necessary attributes
                self.embed_dim = multihead_config.embed_dim
                self.num_heads = multihead_config.num_heads
                self.dropout = attention_config.dropout

            def _create_attention_module(self):
                # Not used in wrapper
                pass

            def _init_qkv_projections(self, factory_kwargs):
                # Not used in wrapper
                pass

            def forward(self, query, key=None, value=None, **kwargs):
                return self._impl(query, key, value, **kwargs)

        register_attention("block_sparse_ring", BlockSparseRingDilatedAttention)
        register_multihead_attention("multihead_block_sparse_ring", BlockSparseWrapper)
        logger.debug("Registered block-sparse attention implementations")

    except ImportError as e:
        logger.warning(f"Failed to register block-sparse implementations: {e}")


# Lazy registration flag to avoid circular imports
_implementations_registered = False


def _ensure_implementations_registered():
    """Ensure implementations are registered (lazy loading to avoid circular imports)."""
    global _implementations_registered  # noqa: PLW0603
    if not _implementations_registered:
        _register_implementations()
        _implementations_registered = True
