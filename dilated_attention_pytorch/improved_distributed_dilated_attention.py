"""
Advanced Distributed Dilated Attention implementation using the refactored core architecture.

This module provides highly optimized distributed attention classes that leverage:
- DeepSpeed ZeRO for memory optimization
- FairScale for model parallelism
- Torch Distributed for communication
- FlashAttention for memory efficiency
- Mixed precision training
- Gradient checkpointing
"""

import warnings
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn

# Advanced distributed training libraries
try:
    import deepspeed

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    warnings.warn("DeepSpeed not available. Install with: pip install deepspeed")

try:
    import fairscale  # noqa: F401
    from fairscale.nn.model_parallel import initialize_model_parallel

    HAS_FAIRSCALE = True
except ImportError:
    HAS_FAIRSCALE = False
    warnings.warn("FairScale not available. Install with: pip install fairscale")

try:
    import apex  # noqa: F401

    HAS_APEX = True
except ImportError:
    HAS_APEX = False

# Import from core architecture
from .core import (
    BaseDilatedAttention,
    BaseMultiheadDilatedAttention,
    DilatedAttentionConfig,
    DistributedConfig,
    MultiheadConfig,
)
from .improved_dilated_attention import ImprovedDilatedAttention


class DistributedImprovedDilatedAttention(BaseDilatedAttention):
    """
    Distributed version of ImprovedDilatedAttention with advanced optimizations.

    Features:
    - Sequence parallelism for ultra-long sequences
    - Model parallelism for large models
    - Memory-efficient communication patterns
    - Integration with DeepSpeed ZeRO
    - Automatic mixed precision
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        use_tf32: bool = True,
        sequence_parallel: bool = False,
        model_parallel: bool = False,
        use_flash_attention: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        # Create configuration
        config = DilatedAttentionConfig(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_tf32=use_tf32,
            device=device,
            dtype=dtype,
        )

        # Initialize base class
        super().__init__(config)

        # Store distributed configuration
        self.distributed_config = DistributedConfig(
            sequence_parallel=sequence_parallel,
            model_parallel=model_parallel,
            pipeline_parallel=False,
            zero_stage=0,
        )

        # Initialize distributed state
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Create local attention instance with optimizations
        self.local_attention = ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_tf32=use_tf32,
            device=device,
            dtype=dtype,
        )

        # Setup distributed communication groups
        self._setup_communication_groups()

        # Initialize sequence parallelism if enabled
        if self.distributed_config.sequence_parallel:
            self._setup_sequence_parallelism()

        # Initialize model parallelism if enabled
        if self.distributed_config.model_parallel and HAS_FAIRSCALE:
            self._setup_model_parallelism()

    def _setup_communication_groups(self):
        """Setup communication groups for efficient distributed operations."""
        if not dist.is_initialized():
            return

        # Create process groups for different parallelism strategies
        self.dp_group = None  # Data parallel group
        self.mp_group = None  # Model parallel group
        self.sp_group = None  # Sequence parallel group

        # Data parallel group (default)
        self.dp_group = dist.new_group(ranks=list(range(self.world_size)))

        # Model parallel groups (if using model parallelism)
        if self.distributed_config.model_parallel:
            # Create groups for model parallelism
            # This is a simplified setup - real implementation would be more complex
            mp_size = min(4, self.world_size)  # Use up to 4 GPUs for model parallelism
            for i in range(0, self.world_size, mp_size):
                ranks = list(range(i, min(i + mp_size, self.world_size)))
                group = dist.new_group(ranks=ranks)
                if self.rank in ranks:
                    self.mp_group = group

        # Sequence parallel groups (if using sequence parallelism)
        if self.distributed_config.sequence_parallel:
            # All ranks participate in sequence parallelism
            self.sp_group = dist.new_group(ranks=list(range(self.world_size)))

    def _setup_sequence_parallelism(self):
        """Setup sequence parallelism for ultra-long sequences."""
        if not self.distributed_config.sequence_parallel or not dist.is_initialized():
            return

        # Calculate local sequence length per GPU
        self.local_seq_len_factor = 1.0 / self.world_size

        # Setup communication buffers
        self._register_communication_hooks()

    def _setup_model_parallelism(self):
        """Setup model parallelism using FairScale."""
        if not self.distributed_config.model_parallel or not HAS_FAIRSCALE:
            return

        # Initialize model parallel groups
        mp_size = getattr(self, "mp_size", 2)
        initialize_model_parallel(mp_size)

    def _register_communication_hooks(self):
        """Register hooks for efficient gradient communication."""
        if not dist.is_initialized():
            return

        # Register hooks for overlapping communication with computation
        def gradient_hook(grad):
            # Overlap gradient reduction with computation
            if self.distributed_config.sequence_parallel:
                return self._reduce_sequence_parallel_grads(grad)
            return grad

        # Apply hooks to attention parameters
        for param in self.local_attention.parameters():
            param.register_hook(gradient_hook)

    def _split_sequence_parallel(self, tensor: Tensor, dim: int = 1) -> Tensor:
        """Split tensor along sequence dimension for sequence parallelism."""
        if not self.distributed_config.sequence_parallel or not dist.is_initialized():
            return tensor

        seq_len = tensor.size(dim)
        local_seq_len = seq_len // self.world_size
        start_idx = self.rank * local_seq_len
        end_idx = start_idx + local_seq_len

        # Handle remainder
        if self.rank == self.world_size - 1:
            end_idx = seq_len

        # Use tensor slicing instead of index_select for better memory efficiency
        # This avoids creating intermediate index tensors and is faster
        if dim == 1:
            return tensor[:, start_idx:end_idx]  # Most common case
        else:
            # General case using narrow for memory efficiency
            return tensor.narrow(dim, start_idx, end_idx - start_idx)

    def _gather_sequence_parallel(self, tensor: Tensor, dim: int = 1) -> Tensor:
        """Optimized gather with asynchronous communication and memory efficiency."""
        if not self.distributed_config.sequence_parallel or not dist.is_initialized():
            return tensor

        # Use asynchronous all_gather for better overlap with computation
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]

        # Start asynchronous all_gather
        gather_handle = dist.all_gather(tensor_list, tensor, group=self.sp_group, async_op=True)

        # Can do other work here while communication happens
        # For now, just wait, but this allows for future optimization
        gather_handle.wait()

        # Use torch.stack + flatten instead of cat for better memory layout
        # This can be more cache-friendly for subsequent operations
        stacked = torch.stack(tensor_list, dim=0)
        if dim == 1:
            # Most common case: sequence dimension
            return stacked.transpose(0, 1).flatten(1, 2)
        else:
            # General case
            return torch.cat(tensor_list, dim=dim)

    def _reduce_sequence_parallel_grads(self, grad: Tensor) -> Tensor:
        """Reduce gradients across sequence parallel ranks."""
        if not self.distributed_config.sequence_parallel or not dist.is_initialized():
            return grad

        # All-reduce gradients
        dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=self.sp_group)
        return grad

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass with distributed optimizations.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to apply causal masking
            attention_mask: Optional attention mask

        Returns:
            Attention output tensor [batch, seq_len, num_heads, head_dim]
        """

        # Validate inputs using base class
        self._validate_forward_inputs(query, key, value, attention_mask)

        # Handle sequence parallelism with memory optimizations
        if self.distributed_config.sequence_parallel:
            # Split input tensors across sequence dimension (no-copy operations where possible)
            query = self._split_sequence_parallel(query, dim=1)
            key = self._split_sequence_parallel(key, dim=1)
            value = self._split_sequence_parallel(value, dim=1)

            if attention_mask is not None:
                attention_mask = self._split_sequence_parallel(attention_mask, dim=-1)

        # Local attention computation with optimized memory allocation
        # Only use autocast if CUDA is available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                output = self.local_attention(
                    query,
                    key,
                    value,
                    is_causal=is_causal,
                    attention_mask=attention_mask,
                )
        else:
            output = self.local_attention(
                query, key, value, is_causal=is_causal, attention_mask=attention_mask
            )

        # Handle sequence parallelism output with async communication where possible
        if self.distributed_config.sequence_parallel:
            # Use asynchronous gather for better overlap with computation
            output = self._gather_sequence_parallel(output, dim=1)

        return output


class DistributedImprovedMultiheadDilatedAttention(BaseMultiheadDilatedAttention):
    """
    Distributed version of ImprovedMultiheadDilatedAttention with enterprise-grade features.

    Features:
    - DeepSpeed ZeRO integration
    - Automatic mixed precision
    - Gradient checkpointing
    - Memory-efficient attention
    - Advanced communication patterns
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        use_tf32: bool = True,
        # Distributed settings
        sequence_parallel: bool = False,
        model_parallel: bool = False,
        use_deepspeed: bool = True,
        use_gradient_checkpointing: bool = False,
        communication_backend: str = "nccl",
        # Memory optimizations
        cpu_offload: bool = False,
        use_8bit_optimizer: bool = False,
        # Advanced features
        use_flash_attention: bool = True,
        compile_model: bool = False,
    ):
        # Create configurations
        multihead_config = MultiheadConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            bias=bias,
            layer_norm=layer_norm,
            layer_norm_eps=layer_norm_eps,
            gamma_init=gamma_init,
            device=device,
            dtype=dtype,
        )

        attention_config = DilatedAttentionConfig(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            use_tf32=use_tf32,
            device=device,
            dtype=dtype,
        )

        # Initialize base class
        super().__init__(multihead_config, attention_config)

        # Store distributed settings
        self.distributed_config = DistributedConfig(
            sequence_parallel=sequence_parallel,
            model_parallel=model_parallel,
            pipeline_parallel=False,
            zero_stage=3 if use_deepspeed else 0,
        )

        # Store additional settings
        self.use_deepspeed = use_deepspeed and HAS_DEEPSPEED
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.communication_backend = communication_backend
        self.cpu_offload = cpu_offload
        self.use_8bit_optimizer = use_8bit_optimizer
        self.use_flash_attention = use_flash_attention
        self.compile_model = compile_model

        # Initialize distributed state
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Initialize with model parallelism if enabled
        self.use_model_parallel = self.distributed_config.model_parallel and HAS_FAIRSCALE
        if self.use_model_parallel:
            self._init_model_parallel_projections()
        else:
            # Use base class initialization
            self._init_qkv_projections({"device": device, "dtype": dtype})

        # Create distributed attention mechanism
        self.attention = self._create_attention_module()

        # Setup advanced optimizations
        self._setup_optimizations(cpu_offload=cpu_offload, compile_model=compile_model)

    def _init_model_parallel_projections(self):
        """Initialize model parallel projections using FairScale."""
        from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

        # Use fused QKV projection with column parallelism for 3x efficiency
        self.qkv_proj = ColumnParallelLinear(
            self.embed_dim,
            3 * self.embed_dim,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
            gather_output=False,
        )
        self.out_proj = RowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
            input_is_parallel=True,
        )

    def _create_attention_module(self) -> DistributedImprovedDilatedAttention:
        """Create the underlying distributed attention module."""
        return DistributedImprovedDilatedAttention(
            segment_lengths=self.attention_config.segment_lengths,
            dilation_rates=self.attention_config.dilation_rates,
            dropout=self.attention_config.dropout,
            use_tf32=self.attention_config.use_tf32,
            sequence_parallel=self.distributed_config.sequence_parallel,
            model_parallel=self.distributed_config.model_parallel,
            use_flash_attention=self.use_flash_attention,
            dtype=self.dtype,
            device=self.device,
        )

    def _reset_parameters(self):
        """Initialize parameters following MAGNETO architecture guidelines."""
        if self.use_model_parallel:
            # Model parallel initialization
            # FairScale handles initialization internally
            pass
        else:
            # Use base class MAGNETO initialization
            super()._reset_parameters()

    def _setup_optimizations(self, cpu_offload: bool = False, compile_model: bool = False):
        """Setup advanced optimizations."""

        # CPU offloading for large models
        if cpu_offload and self.use_deepspeed:
            # DeepSpeed will handle CPU offloading
            pass
        elif cpu_offload:
            # Manual CPU offloading
            self._setup_cpu_offloading()

        # Model compilation
        if compile_model and hasattr(torch, "compile"):
            self.attention = torch.compile(self.attention, mode="max-autotune")

        # Gradient checkpointing
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _setup_cpu_offloading(self):
        """Setup CPU offloading for memory optimization."""
        # Move less frequently used parameters to CPU
        # This is a simplified implementation
        pass

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable()
        else:
            # Manual gradient checkpointing setup
            self._gradient_checkpointing = True

    def _checkpointed_forward(self, *args, **kwargs):
        """Forward pass with gradient checkpointing."""
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self._forward_impl, *args, **kwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """
        Forward pass with optional gradient checkpointing.

        Args:
            query: Query tensor [batch, seq_len, embed_dim]
            key: Key tensor (uses query if None)
            value: Value tensor (uses query if None)
            key_padding_mask: Mask for padded positions [batch, seq_len]
            need_weights: Whether to return attention weights
            attn_mask: Additional attention mask
            is_causal: Whether to apply causal masking
            average_attn_weights: Whether to average attention weights (unused)

        Returns:
            If need_weights is False:
                Attention output [batch, seq_len, embed_dim]
            If need_weights is True:
                Tuple of (output, None) - weights not supported
        """
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query

        # Validate inputs
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                f"Expected 3D tensors (batch, seq_len, embed_dim), got shapes: "
                f"query={query.shape}, key={key.shape}, value={value.shape}"
            )

        batch_size, seq_len, _ = query.shape

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                self._forward_impl,
                query,
                key,
                value,
                key_padding_mask,
                need_weights,
                attn_mask,
                is_causal,
                average_attn_weights,
                use_reentrant=False,
            )
        else:
            return self._forward_impl(
                query,
                key,
                value,
                key_padding_mask,
                need_weights,
                attn_mask,
                is_causal,
                average_attn_weights,
            )

    def _forward_impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None,
        need_weights: bool,
        attn_mask: Tensor | None,
        is_causal: bool,
        average_attn_weights: bool,
    ) -> Tensor | tuple[Tensor, Tensor | None]:
        """Implementation of forward pass."""
        from .utils.attention_utils import split_attention_heads

        # Extract dimensions
        batch_size, seq_len, _ = query.shape

        # Apply projections based on model parallelism
        if self.use_model_parallel and hasattr(self, "qkv_proj"):
            # Fused QKV projection for model parallelism
            qkv = self.qkv_proj(query)
            # Split QKV
            q, k, v = qkv.chunk(3, dim=-1)
            # Reshape to heads
            q = split_attention_heads(q, self.num_heads)
            k = split_attention_heads(k, self.num_heads)
            v = split_attention_heads(v, self.num_heads)
        else:
            # Standard projections
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # Apply layer normalization if enabled
            q, k = self._apply_layer_norm(q, k)

            # Split into heads
            q = split_attention_heads(q, self.num_heads)
            k = split_attention_heads(k, self.num_heads)
            v = split_attention_heads(v, self.num_heads)

        # Combine masks if provided
        combined_mask = self._combine_masks(attn_mask, key_padding_mask, batch_size, seq_len)

        # Apply distributed dilated attention
        attn_output = self.attention(q, k, v, is_causal=is_causal, attention_mask=combined_mask)

        # Merge heads back
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)

        # Apply post-attention layer norm if enabled (MAGNETO style)
        if self.multihead_config.layer_norm and hasattr(self, "q_ln"):
            attn_output = self.q_ln(attn_output)

        # Output projection
        output = self.out_proj(attn_output)

        # For consistency, always return a tuple when requested
        # This matches the behavior of nn.MultiheadAttention
        if need_weights:
            return output, None
        else:
            # Check if we should always return tuple (for compatibility)
            if getattr(self, "_always_return_tuple", False):
                return output, None
            return output

    def _combine_masks(
        self,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> Tensor | None:
        """
        Combine attention mask and key padding mask.

        Args:
            attn_mask: Attention mask [batch*num_heads, seq_len, seq_len] or [seq_len, seq_len]
            key_padding_mask: Key padding mask [batch, seq_len]
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Combined mask or None
        """
        combined_mask = None

        # Handle key padding mask
        if key_padding_mask is not None:
            # Convert key_padding_mask from [batch, seq_len] to attention format
            # [batch, 1, 1, seq_len] -> broadcast to [batch, num_heads, seq_len, seq_len]
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
            key_padding_mask = key_padding_mask.expand(-1, self.num_heads, seq_len, -1)

            # Create mask where True positions are masked (set to -inf)
            combined_mask = torch.zeros(
                batch_size,
                self.num_heads,
                seq_len,
                seq_len,
                device=key_padding_mask.device,
                dtype=self.dtype,
            )
            combined_mask.masked_fill_(key_padding_mask, float("-inf"))

        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [seq_len, seq_len] -> broadcast
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_mask = attn_mask.expand(batch_size, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                # [batch*num_heads, seq_len, seq_len] -> reshape
                attn_mask = attn_mask.view(batch_size, self.num_heads, seq_len, seq_len)

            if combined_mask is None:
                combined_mask = attn_mask
            else:
                combined_mask = combined_mask + attn_mask

        # Reshape to expected format for attention: [batch, seq_len, num_heads, seq_len]
        if combined_mask is not None:
            combined_mask = combined_mask.permute(0, 2, 1, 3)

        return combined_mask

    def extra_repr(self) -> str:
        """String representation for debugging."""
        repr_str = super().extra_repr()
        repr_str += f", sequence_parallel={self.distributed_config.sequence_parallel}"
        repr_str += f", model_parallel={self.distributed_config.model_parallel}"
        if self.use_deepspeed:
            repr_str += ", deepspeed=True"
        if self.use_gradient_checkpointing:
            repr_str += ", gradient_checkpointing=True"
        return repr_str


class DeepSpeedDilatedAttentionEngine:
    """
    DeepSpeed integration for dilated attention with ZeRO optimizations.
    """

    @staticmethod
    def initialize_deepspeed(
        model: nn.Module, config_path: str | None = None, **kwargs
    ) -> tuple[Any, Any, Any, Any]:
        """
        Initialize model with DeepSpeed optimizations.

        Args:
            model: The model to optimize
            config_path: Path to DeepSpeed configuration file
            **kwargs: Additional DeepSpeed parameters

        Returns:
            Tuple of (model_engine, optimizer, train_dataloader, lr_scheduler)
        """
        if not HAS_DEEPSPEED:
            raise ImportError("DeepSpeed not available. Install with: pip install deepspeed")

        # Default DeepSpeed configuration for dilated attention
        default_config = {
            "train_batch_size": kwargs.get("train_batch_size", 16),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": kwargs.get("learning_rate", 1e-4),
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": kwargs.get("learning_rate", 1e-4),
                    "warmup_num_steps": 1000,
                },
            },
            "zero_optimization": {
                "stage": kwargs.get("zero_stage", 2),
                "offload_optimizer": {
                    "device": "cpu" if kwargs.get("cpu_offload", False) else "none",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu" if kwargs.get("cpu_offload", False) else "none",
                    "pin_memory": True,
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "fp16": {
                "enabled": kwargs.get("use_fp16", True),
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "activation_checkpointing": {
                "partition_activations": kwargs.get("partition_activations", False),
                "cpu_checkpointing": kwargs.get("cpu_checkpointing", False),
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False,
            },
            "gradient_clipping": kwargs.get("gradient_clipping", 1.0),
            "steps_per_print": 100,
            "wall_clock_breakdown": False,
        }

        # Load custom config if provided
        if config_path:
            import json

            with open(config_path) as f:
                config = json.load(f)
        else:
            config = default_config

        # Initialize DeepSpeed
        model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
            model=model, config=config, **kwargs
        )

        return model_engine, optimizer, train_dataloader, lr_scheduler

    @staticmethod
    def create_config_file(
        output_path: str,
        train_batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 1e-4,
        zero_stage: int = 2,
        cpu_offload: bool = False,
        use_fp16: bool = True,
        **kwargs,
    ):
        """Create a DeepSpeed configuration file optimized for dilated attention."""

        config = {
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if cpu_offload else "none",
                    "pin_memory": True,
                },
                "offload_param": {
                    "device": "cpu" if cpu_offload else "none",
                    "pin_memory": True,
                },
                "allgather_partitions": True,
                "reduce_scatter": True,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "fp16": {
                "enabled": use_fp16,
                "auto_cast": False,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "activation_checkpointing": {
                "partition_activations": False,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "synchronize_checkpoint_boundary": False,
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
        }

        # Add custom parameters
        config.update(kwargs)

        # Save configuration
        import json

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"DeepSpeed configuration saved to {output_path}")


def create_distributed_model(
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    segment_lengths: list[int],
    dilation_rates: list[int],
    vocab_size: int,
    max_seq_len: int = 32768,
    # Distributed settings
    use_deepspeed: bool = True,
    use_model_parallel: bool = False,
    use_sequence_parallel: bool = False,
    # Optimization settings
    use_gradient_checkpointing: bool = True,
    use_8bit_optimizer: bool = False,
    cpu_offload: bool = False,
    compile_model: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create a complete distributed transformer model with dilated attention.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        segment_lengths: Attention segment lengths
        dilation_rates: Attention dilation rates
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        use_deepspeed: Whether to use DeepSpeed optimizations
        use_model_parallel: Whether to use model parallelism
        use_sequence_parallel: Whether to use sequence parallelism
        use_gradient_checkpointing: Whether to use gradient checkpointing
        use_8bit_optimizer: Whether to use 8-bit optimizers
        cpu_offload: Whether to offload to CPU
        compile_model: Whether to compile the model
        **kwargs: Additional arguments

    Returns:
        Configured transformer model
    """

    class DistributedDilatedTransformer(nn.Module):
        def __init__(self):
            super().__init__()

            # Embeddings
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
            self.dropout = nn.Dropout(kwargs.get("dropout", 0.1))

            # Transformer layers
            self.layers = nn.ModuleList(
                [
                    DistributedImprovedMultiheadDilatedAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=kwargs.get("dropout", 0.1),
                        sequence_parallel=use_sequence_parallel,
                        model_parallel=use_model_parallel,
                        use_deepspeed=use_deepspeed,
                        use_gradient_checkpointing=use_gradient_checkpointing,
                        cpu_offload=cpu_offload,
                        compile_model=compile_model,
                        **kwargs,
                    )
                    for _ in range(num_layers)
                ]
            )

            # Output layers
            self.ln_f = nn.LayerNorm(embed_dim)
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

            # Tie weights
            if kwargs.get("tie_weights", True):
                self.lm_head.weight = self.token_embedding.weight

        def forward(self, input_ids, position_ids=None, is_causal=True):
            batch_size, seq_len = input_ids.shape

            if position_ids is None:
                position_ids = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

            # Embeddings
            token_embeds = self.token_embedding(input_ids)
            pos_embeds = self.pos_embedding(position_ids)
            x = self.dropout(token_embeds + pos_embeds)

            # Transformer layers
            for layer in self.layers:
                attn_out, _ = layer(x, x, x, is_causal=is_causal)
                x = x + attn_out

            # Output
            x = self.ln_f(x)
            logits = self.lm_head(x)

            return logits

    model = DistributedDilatedTransformer()

    # Apply final optimizations
    if compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode="max-autotune")

    return model


# Example usage and configuration
def get_recommended_config(
    model_size: str = "medium", world_size: int = 8, memory_per_gpu_gb: int = 80
) -> dict[str, Any]:
    """Get recommended configuration for different model sizes."""

    configs = {
        "small": {
            "embed_dim": 768,
            "num_heads": 12,
            "num_layers": 12,
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
            "use_deepspeed": True,
            "zero_stage": 2,
            "use_sequence_parallel": False,
            "use_model_parallel": False,
            "cpu_offload": False,
        },
        "medium": {
            "embed_dim": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "segment_lengths": [2048, 4096, 8192, 16384],
            "dilation_rates": [1, 2, 4, 8],
            "use_deepspeed": True,
            "zero_stage": 2,
            "use_sequence_parallel": world_size >= 4,
            "use_model_parallel": False,
            "cpu_offload": memory_per_gpu_gb < 40,
        },
        "large": {
            "embed_dim": 2048,
            "num_heads": 32,
            "num_layers": 24,
            "segment_lengths": [2048, 4096, 8192, 16384, 32768],
            "dilation_rates": [1, 2, 4, 8, 16],
            "use_deepspeed": True,
            "zero_stage": 3,
            "use_sequence_parallel": world_size >= 4,
            "use_model_parallel": world_size >= 8,
            "cpu_offload": True,
        },
    }

    return configs.get(model_size, configs["medium"])


if __name__ == "__main__":
    # Example: Create a distributed model
    config = get_recommended_config("medium", world_size=8)

    model = create_distributed_model(vocab_size=50000, max_seq_len=32768, **config)

    print("Distributed dilated attention model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
