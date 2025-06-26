"""
Refactored Ring Distributed Dilated Attention using core architecture.

This module provides a cleaner implementation of distributed ring attention
using the new configuration system and base classes.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor, nn

from .core import (BaseMultiheadDilatedAttention, DistributedConfig,
                   MultiheadConfig, RingAttentionConfig,
                   create_multihead_dilated_attention)
from .ring_multihead_dilated_attention import RingMultiheadDilatedAttention

logger = logging.getLogger(__name__)


@dataclass
class RingDistributedConfig:
    """Combined configuration for ring distributed attention."""

    multihead_config: MultiheadConfig
    ring_config: RingAttentionConfig
    distributed_config: DistributedConfig

    # Additional distributed ring parameters
    enable_deepspeed: bool = True
    enable_fairscale: bool = True
    enable_monitoring: bool = False
    gradient_accumulation_steps: int = 1
    communication_backend: str = "nccl"

    def __post_init__(self):
        """Validate configuration."""
        if self.ring_config.ring_size and self.distributed_config.world_size:
            if self.ring_config.ring_size > self.distributed_config.world_size:
                raise ValueError(
                    f"ring_size ({self.ring_config.ring_size}) cannot exceed "
                    f"world_size ({self.distributed_config.world_size})"
                )


class RingDistributedDilatedAttention(BaseMultiheadDilatedAttention):
    """
    Distributed Ring Attention implementation using the new core architecture.

    This refactored version provides a cleaner interface while maintaining
    all the advanced features of the original implementation.

    Key improvements:
    - Configuration-based initialization
    - Reuses base class functionality
    - Cleaner separation of concerns
    - Better error handling and validation
    """

    def __init__(
        self,
        config: Optional[RingDistributedConfig] = None,
        multihead_config: Optional[MultiheadConfig] = None,
        ring_config: Optional[RingAttentionConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        **kwargs,
    ):
        """
        Initialize distributed ring attention.

        Args:
            config: Combined configuration object
            multihead_config: Multihead attention configuration
            ring_config: Ring attention configuration
            distributed_config: Distributed training configuration
            **kwargs: Legacy parameters for backward compatibility
        """
        # Handle configuration
        if config is None:
            # Build from individual configs or kwargs
            if multihead_config is None:
                multihead_config = MultiheadConfig(
                    embed_dim=kwargs.get("embed_dim", 768),
                    num_heads=kwargs.get("num_heads", 12),
                    bias=kwargs.get("bias", True),
                    layer_norm=kwargs.get("layer_norm", True),
                    layer_norm_eps=kwargs.get("layer_norm_eps", 1e-5),
                    gamma_init=kwargs.get("gamma_init", 1.0),
                    device=kwargs.get("device"),
                    dtype=kwargs.get("dtype"),
                )

            if ring_config is None:
                ring_config = RingAttentionConfig(
                    segment_lengths=kwargs.get("segment_lengths", [2048, 4096, 8192]),
                    dilation_rates=kwargs.get("dilation_rates", [1, 2, 4]),
                    dropout=kwargs.get("dropout", 0.0),
                    block_size=kwargs.get("block_size", 1024),
                    ring_size=kwargs.get("ring_size"),
                    use_checkpointing=kwargs.get("use_checkpointing", True),
                    use_memory_pool=kwargs.get("use_memory_pool", True),
                )

            if distributed_config is None:
                distributed_config = DistributedConfig(
                    world_size=dist.get_world_size() if dist.is_initialized() else 1,
                    rank=dist.get_rank() if dist.is_initialized() else 0,
                    backend=kwargs.get("communication_backend", "nccl"),
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    find_unused_parameters=False,
                )

            config = RingDistributedConfig(
                multihead_config=multihead_config,
                ring_config=ring_config,
                distributed_config=distributed_config,
                enable_deepspeed=kwargs.get("enable_deepspeed", True),
                enable_fairscale=kwargs.get("enable_fairscale", True),
                enable_monitoring=kwargs.get("enable_monitoring", False),
                gradient_accumulation_steps=kwargs.get(
                    "gradient_accumulation_steps", 1
                ),
            )

        self.config = config

        # Initialize base class
        super().__init__(config.multihead_config, config.ring_config)

        # Create the core attention module using factory
        self.attention_core = create_multihead_dilated_attention(
            "ring",
            multihead_config=config.multihead_config,
            attention_config=config.ring_config,
        )

        # Store ring size for reference
        if hasattr(self.attention_core, "attention"):
            self.ring_size = self.attention_core.attention.ring_size
        else:
            self.ring_size = config.ring_config.ring_size or 1

        # Setup distributed features if needed
        self._setup_distributed()

        logger.info(
            f"Initialized RingDistributedDilatedAttention with "
            f"ring_size={self.ring_size}, world_size={config.distributed_config.world_size}"
        )

    def _setup_distributed(self):
        """Setup distributed training features."""
        config = self.config

        # Check for distributed environment
        if not dist.is_initialized():
            warnings.warn("Distributed not initialized. Running in single-GPU mode.")
            return

        # Setup process groups for ring communication
        self._setup_ring_groups()

        # Setup monitoring if enabled
        if config.enable_monitoring:
            self._setup_monitoring()

    def _setup_ring_groups(self):
        """Setup process groups for ring communication."""
        world_size = self.config.distributed_config.world_size
        ring_size = self.ring_size

        if ring_size > world_size:
            raise ValueError(
                f"ring_size ({ring_size}) cannot exceed world_size ({world_size})"
            )

        # Create ring groups
        self.ring_groups = []
        for i in range(0, world_size, ring_size):
            ranks = list(range(i, min(i + ring_size, world_size)))
            if len(ranks) == ring_size:
                group = dist.new_group(ranks)
                self.ring_groups.append((ranks, group))

        # Find current process's ring group
        rank = self.config.distributed_config.rank
        self.ring_group = None
        self.ring_rank = None

        for ranks, group in self.ring_groups:
            if rank in ranks:
                self.ring_group = group
                self.ring_rank = ranks.index(rank)
                break

    def _setup_monitoring(self):
        """Setup monitoring and logging."""
        try:
            import wandb

            self.wandb = wandb
            self.monitoring_enabled = True
        except ImportError:
            warnings.warn("wandb not available for monitoring")
            self.monitoring_enabled = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        """
        Forward pass through distributed ring attention.

        This implementation delegates to the underlying ring attention module
        while adding distributed coordination.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            key_padding_mask: Padding mask
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights
            is_causal: Whether to use causal masking

        Returns:
            Output tensor and optionally attention weights
        """
        # Use the underlying attention module
        output = self.attention_core(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        # Log metrics if monitoring enabled
        if hasattr(self, "monitoring_enabled") and self.monitoring_enabled:
            self._log_metrics(output)

        return output

    def _log_metrics(self, output: Union[Tensor, Tuple[Tensor, Optional[Tensor]]]):
        """Log metrics to monitoring service."""
        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        metrics = {
            "attention/output_norm": output_tensor.norm().item(),
            "attention/output_mean": output_tensor.mean().item(),
            "attention/output_std": output_tensor.std().item(),
        }

        if self.wandb:
            self.wandb.log(metrics)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"embed_dim={self.config.multihead_config.embed_dim}, "
            f"num_heads={self.config.multihead_config.num_heads}, "
            f"ring_size={self.ring_size}, "
            f"world_size={self.config.distributed_config.world_size}"
        )


def create_ring_distributed_attention(
    embed_dim: int = 768,
    num_heads: int = 12,
    segment_lengths: Sequence[int] = (2048, 4096, 8192),
    dilation_rates: Sequence[int] = (1, 2, 4),
    ring_size: Optional[int] = None,
    **kwargs,
) -> RingDistributedDilatedAttention:
    """
    Convenience function to create ring distributed attention.

    Args:
        embed_dim: Model dimension
        num_heads: Number of attention heads
        segment_lengths: Segment lengths for dilated attention
        dilation_rates: Dilation rates
        ring_size: Size of ring groups
        **kwargs: Additional parameters

    Returns:
        Configured ring distributed attention module
    """
    multihead_config = MultiheadConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["bias", "layer_norm", "gamma_init", "device", "dtype"]
        },
    )

    ring_config = RingAttentionConfig(
        segment_lengths=list(segment_lengths),
        dilation_rates=list(dilation_rates),
        ring_size=ring_size,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["dropout", "block_size", "use_checkpointing"]
        },
    )

    distributed_config = DistributedConfig(
        **{k: v for k, v in kwargs.items() if k in ["world_size", "rank", "backend"]}
    )

    config = RingDistributedConfig(
        multihead_config=multihead_config,
        ring_config=ring_config,
        distributed_config=distributed_config,
        **{
            k: v
            for k, v in kwargs.items()
            if k in ["enable_deepspeed", "enable_fairscale", "enable_monitoring"]
        },
    )

    return RingDistributedDilatedAttention(config)
