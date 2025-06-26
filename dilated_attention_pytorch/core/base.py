"""
Base classes for Dilated Attention implementations.

This module provides abstract base classes that define the common interface
and shared functionality for all dilated attention implementations.
"""

import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from ..utils.validation import ValidationMixin
from .config import DilatedAttentionConfig, MultiheadConfig
from .constants import CURRENT_OPTIMAL_SETTINGS


class BaseDilatedAttention(nn.Module, ValidationMixin, ABC):
    """
    Abstract base class for all dilated attention implementations.

    This class provides the common interface and shared functionality that all
    dilated attention implementations should inherit from. It handles:
    - Configuration validation and storage
    - Common parameter initialization
    - Device and dtype management
    - Caching for frequently computed values
    - Abstract interface for forward pass

    Subclasses must implement:
    - forward(): The actual attention computation

    Args:
        config: Dilated attention configuration

    Example:
        >>> class MyDilatedAttention(BaseDilatedAttention):
        ...     def forward(self, q, k, v, is_causal=False):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, config: DilatedAttentionConfig):
        super().__init__()

        # Store configuration
        self.config = config

        # Extract commonly used attributes
        self.segment_lengths = config.segment_lengths
        self.dilation_rates = config.dilation_rates
        self.dropout = config.dropout
        self.num_groups = config.num_groups
        self.device = config.device
        self.dtype = config.dtype

        # Set TF32 usage if requested
        if config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Create dropout layer if needed
        self.dropout_layer = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # Initialize caches for frequently computed values with size limits
        self._max_cache_size = 100  # Configurable cache size limit
        self._head_groups_cache: dict[int, tuple[list[int], list[tuple[int, int]]]] = OrderedDict()
        self._pattern_cache: dict[Any, Tensor] = OrderedDict()
        self._indices_cache: dict[Any, tuple[Tensor, Tensor]] = OrderedDict()

        # Thread safety locks
        self._cache_lock = threading.RLock()  # Reentrant lock for nested access

        # Hardware optimization flags
        self._use_flash_attn = CURRENT_OPTIMAL_SETTINGS.get("use_flash_attn", False)
        self._optimal_block_size = CURRENT_OPTIMAL_SETTINGS.get("block_size", 512)

    @abstractmethod
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass for dilated attention.

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            attention_mask: Optional attention mask

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        pass

    def _get_head_groups(self, num_heads: int) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Get cached head group distribution.

        Distributes attention heads across dilated attention groups, with caching
        for efficiency.

        Args:
            num_heads: Total number of attention heads

        Returns:
            Tuple of (group_sizes, head_ranges) where:
            - group_sizes: List of number of heads per group
            - head_ranges: List of (start, end) indices for each group
        """
        with self._cache_lock:
            # Check cache first
            if num_heads in self._head_groups_cache:
                # Move to end (LRU behavior)
                self._head_groups_cache.move_to_end(num_heads)
                return self._head_groups_cache[num_heads]

            # Validate number of heads
            self.validate_num_heads(num_heads, self.num_groups)

            # Compute distribution
            heads_per_group = num_heads // self.num_groups
            extra_heads = num_heads % self.num_groups

            group_sizes = [heads_per_group] * self.num_groups
            for i in range(extra_heads):
                group_sizes[i] += 1

            # Compute head ranges
            head_ranges = []
            start = 0
            for size in group_sizes:
                head_ranges.append((start, start + size))
                start += size

            # Cache result with size limit
            result = (group_sizes, head_ranges)
            self._head_groups_cache[num_heads] = result

            # Evict oldest if cache exceeds size limit
            if len(self._head_groups_cache) > self._max_cache_size:
                self._head_groups_cache.popitem(last=False)

            return result

    def _validate_forward_inputs(
        self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor | None = None
    ) -> None:
        """
        Validate inputs to forward pass.

        Args:
            q, k, v: Query, key, value tensors
            attention_mask: Optional attention mask

        Raises:
            ValueError: If inputs are invalid
        """
        # Check tensor dimensions
        self.validate_tensor_shape(q, 4, "query")
        self.validate_tensor_shape(k, 4, "key")
        self.validate_tensor_shape(v, 4, "value")

        # Check shapes match
        self.validate_tensor_shapes_match(
            [q, k, v],
            ["query", "key", "value"],
            dims_to_check=[0, 1, 2, 3],  # All dimensions must match
        )

        # Check device and dtype consistency
        self.validate_device_dtype_consistency([q, k, v], ["query", "key", "value"])

        # Validate sequence length
        seq_len = q.shape[1]
        self.validate_sequence_length(seq_len, self.segment_lengths)

        # Validate attention mask if provided
        if attention_mask is not None:
            batch_size = q.shape[0]
            self.validate_attention_mask(attention_mask, batch_size, seq_len)

    def _apply_dropout(self, tensor: Tensor) -> Tensor:
        """Apply dropout if configured."""
        if self.dropout_layer is not None and self.training:
            return self.dropout_layer(tensor)
        return tensor

    def _cache_get(
        self, cache_dict: OrderedDict, key: Any, compute_fn: Callable | None = None
    ) -> Any | None:
        """
        Thread-safe cache retrieval with LRU behavior.

        Args:
            cache_dict: The cache dictionary to use
            key: Cache key
            compute_fn: Optional function to compute value if not in cache

        Returns:
            Cached value or computed value if compute_fn provided
        """
        with self._cache_lock:
            if key in cache_dict:
                # Move to end for LRU
                cache_dict.move_to_end(key)
                return cache_dict[key]

            if compute_fn is not None:
                # Compute and cache the value
                value = compute_fn()
                cache_dict[key] = value

                # Evict oldest if needed
                if len(cache_dict) > self._max_cache_size:
                    cache_dict.popitem(last=False)

                return value

            return None

    def _clear_caches(self) -> None:
        """Clear all caches (thread-safe)."""
        with self._cache_lock:
            self._head_groups_cache.clear()
            self._pattern_cache.clear()
            self._indices_cache.clear()

    def clear_cache(self, force: bool = False) -> None:
        """Clear cached patterns and buffers to free memory.

        Args:
            force: If True, clear all caches immediately. Otherwise, only clear if needed.
        """
        self._clear_caches()

    def get_memory_info(self) -> dict:
        """Get memory usage information.

        Returns:
            Dictionary with memory usage statistics
        """
        return {
            "cache_size": len(self._head_groups_cache)
            + len(self._pattern_cache)
            + len(self._indices_cache),
            "max_cache_size": self._max_cache_size,
            "memory_complexity": "O(n) for sequence length n",
        }

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"segment_lengths={self.segment_lengths}, "
            f"dilation_rates={self.dilation_rates}, "
            f"dropout={self.dropout}"
        )

    def __getstate__(self):
        """Support for pickling - exclude unpickleable objects."""
        state = self.__dict__.copy()
        # Remove the unpickleable lock
        state["_cache_lock"] = None
        # Clear caches to reduce pickle size
        state["_head_groups_cache"] = {}
        state["_pattern_cache"] = {}
        state["_indices_cache"] = {}
        return state

    def __setstate__(self, state):
        """Support for unpickling - recreate lock."""
        self.__dict__.update(state)
        # Recreate the lock
        self._cache_lock = threading.RLock()


class BaseMultiheadDilatedAttention(nn.Module, ValidationMixin, ABC):
    """
    Abstract base class for multihead dilated attention implementations.

    This class provides the common interface for multihead wrappers around
    dilated attention mechanisms. It handles:
    - QKV projection initialization
    - MAGNETO-style parameter initialization
    - Layer normalization
    - Output projection

    Subclasses must implement:
    - _create_attention_module(): Create the underlying attention module
    - forward(): The actual forward pass

    Args:
        multihead_config: Multihead attention configuration
        attention_config: Dilated attention configuration
    """

    def __init__(
        self,
        multihead_config: MultiheadConfig,
        attention_config: DilatedAttentionConfig,
    ):
        super().__init__()

        # Store configurations
        self.multihead_config = multihead_config
        self.attention_config = attention_config

        # Extract commonly used attributes
        self.embed_dim = multihead_config.embed_dim
        self.num_heads = multihead_config.num_heads
        self.head_dim = multihead_config.head_dim
        self.bias = multihead_config.bias
        self.device = multihead_config.device
        self.dtype = multihead_config.dtype

        # Create attention module
        self.attention = self._create_attention_module()

        # Initialize projections
        factory_kwargs = {"device": self.device, "dtype": self.dtype}

        # QKV projections (can be separate or fused)
        self._init_qkv_projections(factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias, **factory_kwargs)

        # Layer normalization if enabled
        if multihead_config.layer_norm:
            self.q_ln = nn.LayerNorm(
                self.embed_dim, eps=multihead_config.layer_norm_eps, **factory_kwargs
            )
            self.k_ln = nn.LayerNorm(
                self.embed_dim, eps=multihead_config.layer_norm_eps, **factory_kwargs
            )
        else:
            self.q_ln = self.k_ln = None

        # Initialize parameters
        self._reset_parameters()

    @abstractmethod
    def _create_attention_module(self) -> BaseDilatedAttention:
        """
        Create the underlying dilated attention module.

        Returns:
            Dilated attention module instance
        """
        pass

    @abstractmethod
    def _init_qkv_projections(self, factory_kwargs: dict[str, Any]) -> None:
        """
        Initialize QKV projections.

        Subclasses can implement either separate q/k/v projections or a
        fused qkv projection.

        Args:
            factory_kwargs: Device and dtype specifications
        """
        pass

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """
        Forward pass for multihead dilated attention.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (uses query if None)
            value: Value tensor (uses query if None)
            key_padding_mask: Mask for padded key positions
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            is_causal: Whether to apply causal masking

        Returns:
            If need_weights is False:
                Attention output [batch, seq_len, embed_dim]
            If need_weights is True:
                Tuple of (output, attention_weights)
        """
        pass

    def _reset_parameters(self) -> None:
        """
        Initialize parameters using MAGNETO-style initialization.

        This follows the initialization scheme from the MAGNETO paper for
        improved training stability at scale.
        """
        gamma = self.multihead_config.gamma_init

        # Initialize QKV projections
        if hasattr(self, "qkv_proj"):
            # Fused QKV projection
            nn.init.xavier_normal_(self.qkv_proj.weight, gain=gamma)
            if self.qkv_proj.bias is not None:
                nn.init.zeros_(self.qkv_proj.bias)
        else:
            # Separate projections - check if attributes exist
            for attr_name in ["q_proj", "k_proj", "v_proj"]:
                if hasattr(self, attr_name):
                    proj = getattr(self, attr_name)
                    nn.init.xavier_normal_(proj.weight, gain=gamma)
                    if proj.bias is not None:
                        nn.init.zeros_(proj.bias)

        # Initialize output projection
        if hasattr(self, "out_proj"):
            nn.init.xavier_normal_(self.out_proj.weight, gain=gamma)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

        # Layer norm parameters are already initialized properly

    def _apply_layer_norm(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply layer normalization if enabled."""
        if self.q_ln is not None:
            q = self.q_ln(q)
        if self.k_ln is not None:
            k = self.k_ln(k)
        return q, k

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias}, "
            f"layer_norm={self.multihead_config.layer_norm}"
        )
