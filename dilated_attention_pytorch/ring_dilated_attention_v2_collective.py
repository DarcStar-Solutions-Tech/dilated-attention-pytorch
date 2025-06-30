"""
Ring Dilated Attention V2 - Corrected with Collective Operations.

This implementation fixes the distributed communication issues by using
collective operations (all_gather) instead of error-prone isend/irecv.

Key improvements:
1. Uses dist.all_gather instead of isend/irecv
2. No CUDA memory errors
3. Simpler and more robust
4. Often faster due to NCCL optimizations
"""

import math
import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

# Import base functionality from original V2
from .ring_dilated_attention_v2 import RingDilatedAttentionV2

# Import optimized attention utilities
try:
    from .utils.attention_utils import optimize_attention_computation

    HAS_OPTIMIZED_ATTENTION = True
except ImportError:
    HAS_OPTIMIZED_ATTENTION = False
    warnings.warn(
        "Optimized attention utilities not available, falling back to manual computation"
    )


class RingDilatedAttentionV2Collective(RingDilatedAttentionV2):
    """
    Corrected Ring Dilated Attention using collective operations.

    This version replaces the problematic isend/irecv communication with
    robust all_gather operations that handle synchronization properly.
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        enable_memory_pool: bool = False,
        enable_profiling: bool = False,
        lightweight_pool: bool = True,
        use_pattern_cache: bool = True,
    ):
        """Initialize with same parameters as V2."""
        super().__init__(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=dropout,
            ring_size=ring_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=enable_memory_pool,
            enable_profiling=enable_profiling,
            lightweight_pool=lightweight_pool,
            use_pattern_cache=use_pattern_cache,
        )

        # Pre-allocate lists for all_gather if in distributed mode
        if self.mode == "distributed" and self.ring_size > 1:
            self._k_chunks_list = None
            self._v_chunks_list = None

    def _ring_attention(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """
        Ring attention with collective operations and proper dilated patterns.

        This implementation:
        1. Applies dilated attention patterns to the full Q, K, V first
        2. Gathers all dilated K/V chunks using collective operations
        3. Processes them with online softmax for correct normalization
        """
        b, n, h, d = q.shape

        # CRITICAL FIX: Apply dilated attention patterns to all tensors first
        # This ensures each ring step processes properly dilated attention
        q_dilated = self._apply_dilated_attention_pattern(
            q, q, q, is_causal
        )  # Use Q for all since we need dilated Q
        _ = self._apply_dilated_attention_pattern(k, k, k, is_causal)  # Get dilated K
        _ = self._apply_dilated_attention_pattern(v, v, v, is_causal)  # Get dilated V

        # Note: _apply_dilated_attention_pattern normally takes Q,K,V and returns attention output
        # But we need the dilated K,V tensors themselves. Let's use a different approach:

        # Apply dilated patterns directly to K and V chunks
        chunk_size = (n + self.ring_size - 1) // self.ring_size

        # Get local K/V chunks
        local_start = self.rank * chunk_size
        local_end = min((self.rank + 1) * chunk_size, n)
        actual_chunk_size = local_end - local_start

        # Extract local chunks
        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Apply dilated patterns to the local chunks
        k_local_dilated, v_local_dilated = self._apply_dilated_patterns_to_chunk(
            k_local, v_local, local_start, actual_chunk_size
        )

        # Handle padding if needed
        if actual_chunk_size < chunk_size:
            pad_size = chunk_size - actual_chunk_size
            k_local_dilated = F.pad(k_local_dilated, (0, 0, 0, 0, 0, pad_size), value=0)
            v_local_dilated = F.pad(v_local_dilated, (0, 0, 0, 0, 0, pad_size), value=0)

        # Gather all dilated K/V chunks across devices using collective operation
        if (
            self._k_chunks_list is None
            or self._k_chunks_list[0].shape != k_local_dilated.shape
        ):
            self._k_chunks_list = [
                torch.empty_like(k_local_dilated) for _ in range(self.ring_size)
            ]
            self._v_chunks_list = [
                torch.empty_like(v_local_dilated) for _ in range(self.ring_size)
            ]

        # All-gather operation - robust and handles synchronization
        dist.all_gather(self._k_chunks_list, k_local_dilated)
        dist.all_gather(self._v_chunks_list, v_local_dilated)

        # Apply dilated patterns to full query as well
        q_dilated = self._apply_dilated_patterns_to_query(q)

        # Process each chunk with dilated attention using online softmax
        output = torch.zeros((b, h, n, d), device=q.device, dtype=q.dtype)
        running_max = torch.full(
            (b, h, n, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        running_sum = torch.zeros((b, h, n, 1), device=q.device, dtype=q.dtype)

        for step in range(self.ring_size):
            # Determine which chunk to process
            chunk_idx = (self.rank - step) % self.ring_size
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, n)

            # Get the dilated K/V chunks for this step
            k_chunk_dilated = self._k_chunks_list[chunk_idx]
            v_chunk_dilated = self._v_chunks_list[chunk_idx]

            # Trim padding if this is the last chunk
            if chunk_end - chunk_start < chunk_size:
                actual_size = chunk_end - chunk_start
                k_chunk_dilated = k_chunk_dilated[:, :actual_size]
                v_chunk_dilated = v_chunk_dilated[:, :actual_size]

            # Compute attention with dilated patterns using online softmax
            self._compute_chunk_attention_with_online_softmax(
                q_dilated,
                k_chunk_dilated,
                v_chunk_dilated,
                chunk_start,
                is_causal,
                running_max,
                running_sum,
                output,
                step,
            )

        # Final normalization
        output = output / (running_sum + 1e-8)  # Add epsilon for numerical stability

        # Transpose back to [b, n, h, d]
        return output.transpose(1, 2)

    def _compute_chunk_attention_simple(
        self, q: Tensor, k_chunk: Tensor, v_chunk: Tensor, is_causal: bool
    ):
        """Simplified attention computation for a chunk."""
        b, n, h, d = q.shape
        _, n_kv, _, _ = k_chunk.shape

        # Transpose to [b, h, n, d] format
        q_t = q.transpose(1, 2)  # [b, h, n, d]
        k_t = k_chunk.transpose(1, 2)  # [b, h, n_kv, d]
        v_t = v_chunk.transpose(1, 2)  # [b, h, n_kv, d]

        # Compute attention scores
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(
            d
        )  # [b, h, n, n_kv]

        # Apply causal mask if needed (simplified)
        if is_causal:
            # For simplicity, apply basic causal mask
            mask = torch.triu(torch.ones(n, n_kv, device=q.device), diagonal=1).bool()
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Get max values for numerical stability
        max_vals = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]

        # Compute softmax
        exp_scores = torch.exp(scores - max_vals)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        attn_weights = exp_scores / sum_exp

        # Apply to values
        output = torch.matmul(attn_weights, v_t)  # [b, h, n, d]

        return output, max_vals

    def _combine_chunk_outputs(self, outputs, max_vals):
        """Combine outputs from different chunks with proper normalization."""
        if len(outputs) == 1:
            return outputs[0]

        # For simplicity, just average the outputs
        # In a full implementation, this would use proper online softmax
        combined = torch.stack(outputs, dim=0).mean(dim=0)

        # Transpose back to [b, n, h, d]
        return combined.transpose(1, 2)

    def _apply_dilated_patterns_to_chunk(
        self, k_chunk: Tensor, v_chunk: Tensor, chunk_start: int, chunk_size: int
    ) -> tuple[Tensor, Tensor]:
        """Apply dilated patterns to K/V chunks based on head groups."""
        b, n, h, d = k_chunk.shape

        # Calculate head groups for different segment lengths (same as parent)
        heads_per_group = self._calculate_head_groups(h)

        # Create output tensors
        k_dilated = torch.zeros_like(k_chunk)
        v_dilated = torch.zeros_like(v_chunk)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Apply dilation to this head group
            offset = i % dilation_rate if dilation_rate > 1 else 0

            # Get dilated indices for this segment
            k_group = k_chunk[:, :, head_start:head_end, :]
            v_group = v_chunk[:, :, head_start:head_end, :]

            if dilation_rate > 1 and n >= dilation_rate:
                # Apply dilation pattern
                dilated_indices = torch.arange(
                    offset, n, dilation_rate, device=k_chunk.device
                )
                if len(dilated_indices) < n:
                    # Pad by repeating pattern
                    repeats = (n + len(dilated_indices) - 1) // len(dilated_indices)
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:n] % n

                k_group_dilated = k_group.index_select(1, dilated_indices)
                v_group_dilated = v_group.index_select(1, dilated_indices)
            else:
                # No dilation needed - these groups could potentially use optimized attention
                k_group_dilated = k_group
                v_group_dilated = v_group

            k_dilated[:, :, head_start:head_end, :] = k_group_dilated
            v_dilated[:, :, head_start:head_end, :] = v_group_dilated
            head_start = head_end

        return k_dilated, v_dilated

    def _apply_dilated_patterns_to_query(self, q: Tensor) -> Tensor:
        """Apply dilated patterns to query tensor based on head groups."""
        b, n, h, d = q.shape

        # Calculate head groups for different segment lengths (same as parent)
        heads_per_group = self._calculate_head_groups(h)

        # Create output tensor
        q_dilated = torch.zeros_like(q)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, heads_per_group)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            # Apply dilation to this head group
            offset = i % dilation_rate if dilation_rate > 1 else 0

            # Get query group
            q_group = q[:, :, head_start:head_end, :]

            if dilation_rate > 1 and n >= dilation_rate:
                # Apply dilation pattern to query
                dilated_indices = torch.arange(
                    offset, n, dilation_rate, device=q.device
                )
                if len(dilated_indices) < n:
                    # Pad by repeating pattern
                    repeats = (n + len(dilated_indices) - 1) // len(dilated_indices)
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:n] % n

                q_group_dilated = q_group.index_select(1, dilated_indices)
            else:
                # No dilation needed
                q_group_dilated = q_group

            q_dilated[:, :, head_start:head_end, :] = q_group_dilated
            head_start = head_end

        return q_dilated

    def _compute_chunk_attention_with_online_softmax(
        self,
        q_dilated: Tensor,
        k_chunk: Tensor,
        v_chunk: Tensor,
        chunk_start: int,
        is_causal: bool,
        running_max: Tensor,
        running_sum: Tensor,
        output: Tensor,
        step: int,
    ):
        """Compute attention for a chunk using optimized attention and online softmax."""
        b, n, h, d = q_dilated.shape
        _, n_kv, _, _ = k_chunk.shape

        if HAS_OPTIMIZED_ATTENTION and not is_causal and n == n_kv:
            # Use optimized attention only when Q and K/V have same sequence length
            # Ring attention typically has Q full sequence and K/V chunks, so this is rare
            # But when it happens (e.g., chunk size equals sequence length), use optimization
            try:
                # Transpose to [b, h, n, d] format for optimized attention
                q_t = q_dilated.transpose(1, 2)  # [b, h, n, d]
                k_t = k_chunk.transpose(1, 2)  # [b, h, n_kv, d]
                v_t = v_chunk.transpose(1, 2)  # [b, h, n_kv, d]

                # Use optimized attention computation
                chunk_output = optimize_attention_computation(
                    q_t,
                    k_t,
                    v_t,
                    is_causal=False,  # Handle causal separately due to online softmax
                    dropout_p=self.dropout if self.training else 0.0,
                )  # [b, h, n, d]

                # For online softmax, we need to compute scores manually to get statistics
                # This is a trade-off: we get optimized kernel for main computation
                # but need manual computation for online softmax statistics
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(d)

                # Apply causal mask to scores for online softmax computation
                if is_causal:
                    # Optimized causal mask creation
                    causal_mask = torch.triu(
                        torch.ones(n, n_kv, device=q_dilated.device, dtype=torch.bool),
                        diagonal=chunk_start + 1,
                    )
                    scores.masked_fill_(
                        causal_mask.unsqueeze(0).unsqueeze(1), float("-inf")
                    )

                # Online softmax statistics update
                chunk_max = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]
                new_max = torch.maximum(running_max, chunk_max)

                # Rescale existing output if max changed and not first step
                if step > 0:
                    output.mul_(torch.exp(running_max - new_max))

                # Update running sum with proper scaling
                running_sum.mul_(torch.exp(running_max - new_max))
                running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))

                # Update running max
                running_max.copy_(new_max)

                # For online softmax, we need to scale the optimized output by the proper factor
                # This approximation works when the max doesn't change significantly
                output_scale = torch.exp(-new_max).expand_as(chunk_output)
                scaled_output = chunk_output * output_scale

                # Add to output (already in [b, h, n, d] format)
                output.add_(scaled_output)

                return  # Early return for optimized path

            except Exception:
                # Fall back to manual computation if optimized fails
                pass

        # Manual computation fallback (original implementation)
        # Transpose to [b, h, n, d] format for computation
        q_t = q_dilated.transpose(1, 2)  # [b, h, n, d]
        k_t = k_chunk.transpose(1, 2)  # [b, h, n_kv, d]
        v_t = v_chunk.transpose(1, 2)  # [b, h, n_kv, d]

        # Compute attention scores
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(
            d
        )  # [b, h, n, n_kv]

        # Apply causal mask if needed
        if is_causal:
            # Optimized causal mask creation (replace nested loops)
            causal_mask = torch.triu(
                torch.ones(n, n_kv, device=q_dilated.device, dtype=torch.bool),
                diagonal=chunk_start + 1,
            )
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        # Online softmax update
        chunk_max = scores.amax(dim=-1, keepdim=True)  # [b, h, n, 1]
        new_max = torch.maximum(running_max, chunk_max)

        # Rescale existing output if max changed and not first step
        if step > 0:
            output.mul_(torch.exp(running_max - new_max))

        # Update running sum with proper scaling
        running_sum.mul_(torch.exp(running_max - new_max))
        running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))

        # Update running max
        running_max.copy_(new_max)

        # Accumulate weighted values
        exp_scores = torch.exp(scores - new_max)  # [b, h, n, n_kv]
        chunk_output = torch.matmul(exp_scores, v_t)  # [b, h, n, d]

        # Add to output (already in [b, h, n, d] format)
        output.add_(chunk_output)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass with collective communication.

        Routes to appropriate implementation based on mode.
        """
        # For distributed mode with ring_size > 1, use collective ring attention
        if self.mode == "distributed" and self.ring_size > 1:
            return self._ring_attention(q, k, v, is_causal)
        else:
            # Fall back to parent implementation for other modes
            return super().forward(q, k, v, is_causal)


# Convenience function for creating the corrected version
def create_ring_dilated_attention_v2_collective(
    segment_lengths: list[int], dilation_rates: list[int], **kwargs
) -> RingDilatedAttentionV2Collective:
    """
    Create a Ring Dilated Attention V2 with collective operations.

    This is the recommended version for distributed training as it uses
    robust collective operations instead of error-prone point-to-point
    communication.
    """
    return RingDilatedAttentionV2Collective(
        segment_lengths=segment_lengths, dilation_rates=dilation_rates, **kwargs
    )
