"""
Fixed API wrapper for BlockSparseAdaptive to match standard interface.

This wrapper provides API consistency with other block-sparse implementations.
"""

from typing import List, Optional, Tuple
import torch
from torch import Tensor

from .block_sparse_adaptive import (
    BlockSparseAdaptive as _BlockSparseAdaptiveOriginal,
    AdaptiveConfig,
)
from .block_sparse_attention import SparsePatternConfig


class BlockSparseAdaptive(_BlockSparseAdaptiveOriginal):
    """
    API-consistent wrapper for BlockSparseAdaptive.

    This wrapper allows BlockSparseAdaptive to be initialized with the same
    interface as other block-sparse implementations, inferring num_heads and
    head_dim from the input tensors during forward pass.
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        sparse_config: Optional[SparsePatternConfig] = None,
        adaptive_config: Optional[AdaptiveConfig] = None,
        **kwargs,
    ):
        """Initialize with consistent API."""
        # Extract any provided dimensions
        self._num_heads = kwargs.pop("num_heads", None)
        self._head_dim = kwargs.pop("head_dim", None)
        self._embed_dim = kwargs.pop("embed_dim", None)
        self._initialized = False

        # If embed_dim and num_heads provided, calculate head_dim
        if (
            self._embed_dim is not None
            and self._num_heads is not None
            and self._head_dim is None
        ):
            self._head_dim = self._embed_dim // self._num_heads

        # Store initialization parameters
        self._init_segment_lengths = segment_lengths
        self._init_dilation_rates = dilation_rates
        self._init_adaptive_config = adaptive_config or AdaptiveConfig()
        self._init_kwargs = kwargs

        # For compatibility, we need to defer full initialization
        # until we know the dimensions from the first forward pass
        if self._num_heads is not None and self._head_dim is not None:
            # If dimensions provided, initialize immediately
            super().__init__(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                num_heads=self._num_heads,
                head_dim=self._head_dim,
                adaptive_config=self._init_adaptive_config,
                **kwargs,
            )
            self._initialized = True
        else:
            # Defer initialization - create minimal parent
            # We need to bypass the parent init temporarily
            torch.nn.Module.__init__(self)

            # Set minimal attributes needed before full init
            self.sparse_config = sparse_config or SparsePatternConfig(
                pattern_type="adaptive",
                sparsity_ratio=self._init_adaptive_config.base_sparsity,
                block_size=kwargs.get("block_size", 64),
            )
            self.block_size = self.sparse_config.block_size

    @property
    def num_heads(self) -> Optional[int]:
        """Get number of heads."""
        if self._initialized and hasattr(self, "_adaptive_num_heads"):
            return self._adaptive_num_heads
        return self._num_heads

    @num_heads.setter
    def num_heads(self, value: int):
        """Set number of heads."""
        if self._initialized:
            self._adaptive_num_heads = value
        else:
            self._num_heads = value

    @property
    def head_dim(self) -> Optional[int]:
        """Get head dimension."""
        if self._initialized and hasattr(self, "_adaptive_head_dim"):
            return self._adaptive_head_dim
        return self._head_dim

    @head_dim.setter
    def head_dim(self, value: int):
        """Set head dimension."""
        if self._initialized:
            self._adaptive_head_dim = value
        else:
            self._head_dim = value

    def _lazy_init(self, q: Tensor):
        """Lazy initialization when dimensions are known."""
        if self._initialized:
            return

        # Infer dimensions from input
        if q.dim() == 4:
            # [batch, seq_len, num_heads, head_dim]
            _, _, num_heads, head_dim = q.shape
        else:
            raise ValueError(
                f"Expected 4D tensor [batch, seq_len, num_heads, head_dim], got {q.dim()}D"
            )

        # Add device to kwargs for initialization
        init_kwargs = self._init_kwargs.copy()
        init_kwargs["device"] = q.device

        # Now initialize the parent class properly
        _BlockSparseAdaptiveOriginal.__init__(
            self,
            segment_lengths=self._init_segment_lengths,
            dilation_rates=self._init_dilation_rates,
            num_heads=num_heads,
            head_dim=head_dim,
            adaptive_config=self._init_adaptive_config,
            **init_kwargs,
        )

        self._initialized = True
        self._num_heads = num_heads
        self._head_dim = head_dim

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        return_attention_weights: bool = False,
        **kwargs,
    ) -> Tensor | Tuple[Tensor, dict]:
        """Forward pass with lazy initialization."""
        # Initialize on first forward pass if needed
        self._lazy_init(q)

        # Map return_attention_weights to return_pattern for parent class
        return_pattern = kwargs.pop("return_pattern", return_attention_weights)

        # Call parent forward
        result = super().forward(
            q=q, k=k, v=v, is_causal=is_causal, return_pattern=return_pattern, **kwargs
        )

        # Handle return value consistency
        if return_attention_weights:
            if isinstance(result, tuple):
                output, pattern_info = result
                # Convert pattern info to attention weights format
                attention_weights = {
                    "patterns": pattern_info.get("patterns", []),
                    "type": "adaptive",
                    "sparsity": self.sparse_config.sparsity_ratio,
                }
                return output, attention_weights
            else:
                # Parent didn't return patterns
                return result, None
        else:
            if isinstance(result, tuple):
                return result[0]
            return result

    def to(self, *args, **kwargs):
        """Override to handle both initialized and uninitialized states."""
        # If initialized, use parent's to()
        if self._initialized:
            return super().to(*args, **kwargs)
        else:
            # Just update the stored kwargs for later initialization
            # Extract device from args/kwargs
            device = None
            if args and hasattr(args[0], "type"):
                device = args[0]
            elif "device" in kwargs:
                device = kwargs["device"]

            if device is not None:
                self._init_kwargs["device"] = device

            # Still call Module's to() for any parameters we might have
            torch.nn.Module.to(self, *args, **kwargs)
            return self

    def __repr__(self):
        if self._initialized:
            return (
                f"BlockSparseAdaptive("
                f"num_heads={self._num_heads}, "
                f"head_dim={self._head_dim}, "
                f"adaptive_config={self._init_adaptive_config})"
            )
        else:
            return "BlockSparseAdaptive(uninitialized)"


# Also create a factory function for easier use
def create_adaptive_block_sparse_consistent(
    segment_lengths: List[int],
    dilation_rates: List[int],
    sparse_config: Optional[SparsePatternConfig] = None,
    adaptive_config: Optional[AdaptiveConfig] = None,
    **kwargs,
) -> BlockSparseAdaptive:
    """
    Create BlockSparseAdaptive with consistent API.

    This factory ensures API consistency with other block-sparse implementations.
    """
    return BlockSparseAdaptive(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        sparse_config=sparse_config,
        adaptive_config=adaptive_config,
        **kwargs,
    )
