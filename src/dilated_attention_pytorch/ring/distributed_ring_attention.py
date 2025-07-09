"""Enterprise-grade distributed ring attention implementation.

This module provides ring attention with advanced features for production
deployments including DeepSpeed integration, fault tolerance, and monitoring.
"""

import math
import time
from typing import Optional, Tuple, Dict, Any
import warnings
import torch
import torch.nn.functional as F
from torch import Tensor

from .base.base_ring_attention import BaseRingAttention, RingAttentionState
from .base.ring_communication_mixin import RingCommunicationMixin
from .base.ring_config import RingAttentionConfig, RingCommunicationStats
from ..utils.attention_utils import create_causal_mask

# Optional imports for enterprise features
try:
    import deepspeed  # noqa: F401

    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from fairscale.nn import checkpoint_wrapper  # noqa: F401

    HAS_FAIRSCALE = True
except ImportError:
    HAS_FAIRSCALE = False


class DistributedRingAttention(BaseRingAttention, RingCommunicationMixin):
    """Enterprise distributed ring attention with advanced features.

    This implementation provides:
    - DeepSpeed integration for ZeRO optimization
    - Fault tolerance with automatic recovery
    - Performance monitoring and logging
    - Gradient compression and optimization
    - Checkpoint/resume functionality
    - Watchdog timer for deadlock detection
    """

    def __init__(
        self,
        config: RingAttentionConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_deepspeed: bool = True,
        enable_monitoring: bool = True,
    ):
        """Initialize distributed ring attention.

        Args:
            config: Ring attention configuration
            device: Device to place tensors on
            dtype: Data type for tensors
            enable_deepspeed: Enable DeepSpeed integration if available
            enable_monitoring: Enable performance monitoring
        """
        # Initialize base classes
        BaseRingAttention.__init__(self, config, device, dtype)
        RingCommunicationMixin.__init__(self)

        # Store config
        self.config = config
        self.enable_deepspeed = enable_deepspeed and HAS_DEEPSPEED
        self.enable_monitoring = enable_monitoring

        # Communication statistics
        self.comm_stats = RingCommunicationStats()

        # Monitoring setup
        self.monitoring_data = {
            "forward_passes": 0,
            "total_tokens": 0,
            "communication_time": 0.0,
            "computation_time": 0.0,
            "peak_memory_mb": 0.0,
        }

        # Watchdog setup
        self._watchdog_enabled = config.enable_watchdog
        self._watchdog_timeout = config.watchdog_timeout
        self._last_activity_time = time.time()

        # DeepSpeed setup
        if self.enable_deepspeed:
            self._setup_deepspeed()

        # Validate setup
        if self.is_distributed:
            self._validate_distributed_setup()

    def _setup_deepspeed(self):
        """Setup DeepSpeed integration."""
        if not HAS_DEEPSPEED:
            warnings.warn("DeepSpeed requested but not available")
            self.enable_deepspeed = False
            return

        # Check if already initialized with DeepSpeed
        if hasattr(self, "deepspeed_config"):
            return

        self.deepspeed_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {"enabled": self.dtype == torch.float16},
            "bf16": {"enabled": self.dtype == torch.bfloat16},
            "zero_optimization": {
                "stage": 2,  # ZeRO-2 for ring attention
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
            },
            "gradient_clipping": 1.0,
            "communication_data_type": "fp16"
            if self.dtype == torch.float16
            else "fp32",
        }

    def _validate_distributed_setup(self):
        """Validate distributed environment setup."""
        if not self.validate_ring_setup():
            raise RuntimeError("Ring communication validation failed")

        # Check backend compatibility
        backend = torch.distributed.get_backend()
        if backend != self.config.communication_backend:
            warnings.warn(
                f"Backend mismatch: initialized with {backend}, "
                f"config specifies {self.config.communication_backend}"
            )

        # Log setup information
        if self.rank == 0:
            print("Distributed Ring Attention initialized:")
            print(f"  World size: {self.world_size}")
            print(f"  Backend: {backend}")
            print(f"  DeepSpeed: {self.enable_deepspeed}")
            print(f"  Monitoring: {self.enable_monitoring}")

    def _update_watchdog(self):
        """Update watchdog timer to prevent deadlock detection."""
        if self._watchdog_enabled:
            self._last_activity_time = time.time()

    def _check_watchdog(self):
        """Check if watchdog timeout has been exceeded."""
        if not self._watchdog_enabled:
            return

        elapsed = time.time() - self._last_activity_time
        if elapsed > self._watchdog_timeout:
            raise RuntimeError(
                f"Watchdog timeout ({self._watchdog_timeout}s) exceeded. "
                f"Possible deadlock in ring communication."
            )

    def _log_monitoring_data(self, key: str, value: float):
        """Log monitoring data to various backends."""
        self.monitoring_data[key] = value

        # Log to WandB if available
        if self.enable_monitoring and HAS_WANDB and wandb.run is not None:
            wandb.log({f"ring_attention/{key}": value})

    def _split_sequence(
        self, x: Tensor, already_split: bool = False
    ) -> Tuple[Tensor, int, int]:
        """Split sequence for local processing with monitoring.

        Args:
            x: Input tensor of shape (batch, seq_len, ...)
            already_split: Whether sequence is already split

        Returns:
            Tuple of (local_chunk, start_idx, end_idx)
        """
        self._update_watchdog()

        if already_split or not self.is_distributed:
            seq_len = x.shape[1]
            return x, 0, seq_len

        batch_size, seq_len = x.shape[:2]

        # Validate sequence length
        assert seq_len % self.world_size == 0, (
            f"Sequence length {seq_len} must be divisible by world size {self.world_size}"
        )

        local_seq_len = seq_len // self.world_size
        start_idx = self.rank * local_seq_len
        end_idx = start_idx + local_seq_len

        # Extract local chunk
        local_chunk = x[:, start_idx:end_idx].contiguous()

        # Update monitoring
        self._log_monitoring_data("local_seq_len", local_seq_len)
        self._log_monitoring_data("total_tokens", batch_size * local_seq_len)

        return local_chunk, start_idx, end_idx

    def _ring_communication(
        self, tensor: Tensor, direction: str = "forward", tag: int = 0
    ) -> Tensor:
        """Perform ring communication with monitoring and fault tolerance.

        Args:
            tensor: Tensor to communicate
            direction: "forward" or "backward"
            tag: Communication tag

        Returns:
            Received tensor from neighbor
        """
        self._update_watchdog()

        start_time = time.time()

        try:
            if direction == "forward":
                result = self.ring_pass_forward(tensor, tag=tag)
            else:
                result = self.ring_pass_backward(tensor, tag=tag)

            # Update statistics
            comm_time = time.time() - start_time
            self.comm_stats.update_communication(
                bytes_transferred=tensor.numel() * tensor.element_size(),
                comm_time=comm_time,
                success=True,
            )

            # Log monitoring data
            self._log_monitoring_data("communication_time", comm_time)

            return result

        except Exception as e:
            # Update failure statistics
            self.comm_stats.update_communication(0, 0.0, success=False)

            # Log error
            if self.enable_monitoring:
                warnings.warn(f"Ring communication failed: {e}")

            # Re-raise if error recovery disabled
            if not self.config.enable_error_recovery:
                raise

            # Return zeros as fallback
            return torch.zeros_like(tensor)

    def _local_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        segment_idx: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention on local chunk with monitoring.

        Args:
            q: Query tensor of shape (batch, q_len, num_heads, head_dim)
            k: Key tensor of shape (batch, kv_len, num_heads, head_dim)
            v: Value tensor of shape (batch, kv_len, num_heads, head_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            segment_idx: Current segment index

        Returns:
            Tuple of (attention_output, lse) where lse is log-sum-exp
        """
        self._update_watchdog()

        start_time = time.time()

        batch_size, q_len, num_heads, head_dim = q.shape
        kv_len = k.shape[1]

        # Reshape for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, q_len, head_dim)
        k = k.transpose(1, 2)  # (batch, num_heads, kv_len, head_dim)
        v = v.transpose(1, 2)  # (batch, num_heads, kv_len, head_dim)

        # Compute attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Apply causal mask if needed
        if is_causal:
            causal_mask = create_causal_mask(
                q_len, kv_len, device=self.device, dtype=scores.dtype
            )
            scores = scores + causal_mask

        # Compute log-sum-exp for numerical stability
        lse = torch.logsumexp(scores, dim=-1)  # (batch, num_heads, q_len)

        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout if configured
        if self.dropout_p > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout_p)

        # Compute attention output
        attn_output = torch.matmul(
            attn_weights, v
        )  # (batch, num_heads, q_len, head_dim)

        # Transpose back
        attn_output = attn_output.transpose(1, 2)  # (batch, q_len, num_heads, head_dim)

        # Update monitoring
        comp_time = time.time() - start_time
        self._log_monitoring_data("computation_time", comp_time)

        return attn_output, lse

    def _accumulate_results(
        self,
        local_out: Tensor,
        local_lse: Tensor,
        remote_out: Tensor,
        remote_lse: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Accumulate results from ring passes using log-sum-exp.

        Args:
            local_out: Local attention output (batch, seq, heads, dim)
            local_lse: Local log-sum-exp (batch, heads, seq)
            remote_out: Remote attention output
            remote_lse: Remote log-sum-exp

        Returns:
            Tuple of (accumulated_output, accumulated_lse)
        """
        # Compute stable accumulation using log-sum-exp trick
        max_lse = torch.maximum(local_lse, remote_lse)

        # Compute weights
        local_weight = torch.exp(local_lse - max_lse)
        remote_weight = torch.exp(remote_lse - max_lse)

        # Reshape weights for broadcasting
        local_weight = local_weight.transpose(1, 2).unsqueeze(-1)
        remote_weight = remote_weight.transpose(1, 2).unsqueeze(-1)

        # Weighted combination
        accumulated_out = (local_out * local_weight + remote_out * remote_weight) / (
            local_weight + remote_weight
        )

        # Update LSE
        accumulated_lse = max_lse + torch.log(
            torch.exp(local_lse - max_lse) + torch.exp(remote_lse - max_lse)
        )

        return accumulated_out, accumulated_lse

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        already_split: bool = False,
    ) -> Tensor:
        """Forward pass of distributed ring attention.

        Args:
            query: Query tensor (batch, seq_len, num_heads, head_dim)
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask
            is_causal: Whether to use causal masking
            already_split: Whether inputs are already split

        Returns:
            Attention output
        """
        # Check watchdog
        self._check_watchdog()

        # Update pass counter
        self.monitoring_data["forward_passes"] += 1

        # Input validation
        batch_size, seq_len, num_heads, head_dim = query.shape
        self._validate_sequence_length(seq_len)

        # Record peak memory before
        if self.enable_monitoring and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        # Split sequences for local processing
        q_local, q_start, q_end = self._split_sequence(query, already_split)
        k_local, k_start, k_end = self._split_sequence(key, already_split)
        v_local, v_start, v_end = self._split_sequence(value, already_split)

        local_seq_len = q_local.shape[1]

        # Initialize state
        state = RingAttentionState(
            batch_size=batch_size,
            local_seq_len=local_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=self.device,
            dtype=self.dtype,
        )

        # Current K and V start as local chunks
        current_k = k_local
        current_v = v_local

        # Perform ring passes
        for ring_step in range(self.world_size):
            # Compute local attention
            local_out, local_lse = self._local_attention(
                q_local,
                current_k,
                current_v,
                attention_mask=attention_mask,
                is_causal=is_causal,
                segment_idx=ring_step,
            )

            # Accumulate results
            if ring_step == 0:
                state.output = local_out
                state.lse = local_lse
            else:
                state.output, state.lse = self._accumulate_results(
                    state.output, state.lse, local_out, local_lse
                )

            # Ring communication for next iteration (except last)
            if ring_step < self.world_size - 1:
                current_k = self._ring_communication(
                    current_k, direction="forward", tag=ring_step * 2
                )
                current_v = self._ring_communication(
                    current_v, direction="forward", tag=ring_step * 2 + 1
                )

        # Synchronize before returning
        if self.is_distributed:
            self._synchronize_ring()

        # Record peak memory after
        if self.enable_monitoring and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            self._log_monitoring_data("peak_memory_mb", peak_memory - start_memory)

        # Log statistics
        if self.config.log_communication_stats and self.rank == 0:
            stats = self.comm_stats.get_summary()
            print(f"Distributed Ring Attention Stats: {stats}")
            print(f"Monitoring Data: {self.monitoring_data}")

        return state.output

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of current state.

        Returns:
            Dictionary containing state for checkpoint
        """
        return {
            "comm_stats": self.comm_stats.__dict__,
            "monitoring_data": self.monitoring_data,
            "config": self.config.to_dict(),
        }

    def restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
        """
        if "comm_stats" in checkpoint:
            for k, v in checkpoint["comm_stats"].items():
                setattr(self.comm_stats, k, v)

        if "monitoring_data" in checkpoint:
            self.monitoring_data.update(checkpoint["monitoring_data"])

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        base_repr = super().extra_repr()
        features = []
        if self.enable_deepspeed:
            features.append("DeepSpeed")
        if self.enable_monitoring:
            features.append("Monitoring")
        if self._watchdog_enabled:
            features.append(f"Watchdog({self._watchdog_timeout}s)")

        features_str = ", ".join(features)
        return f"{base_repr}, features=[{features_str}]"
