"""
Ring Dilated Attention V2 with Flash Attention support.

This implementation enhances RingDilatedAttentionV2Collective with:
1. GPU architecture-aware Flash Attention integration
2. Automatic fallback for unsupported GPUs
3. Memory-efficient chunked processing
4. Optimized for both Pascal and modern GPUs
"""

import math
import warnings
from typing import Optional

import torch
from torch import Tensor

# Import base class
from .ring_dilated_attention_v2_collective import RingDilatedAttentionV2Collective

# Import utilities
try:
    from .utils.flash_attention_utils import (
        flash_attention_forward,
        chunked_flash_attention,
        get_flash_attention_support,
    )

    HAS_FLASH_UTILS = True
except ImportError:
    HAS_FLASH_UTILS = False
    warnings.warn("Flash Attention utilities not available")

try:
    from .utils.gpu_utils import get_optimal_dtype

    _ = get_optimal_dtype  # Mark as used

    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False


class RingDilatedAttentionV2Flash(RingDilatedAttentionV2Collective):
    """
    Ring Dilated Attention with Flash Attention optimization.

    This class extends RingDilatedAttentionV2Collective with:
    - Flash Attention 3/2 support with automatic fallback
    - GPU architecture-aware optimization
    - Memory-efficient attention computation
    - Reduced memory usage for long sequences
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
        memory_pool_threshold_mb: float = 16.0,
        use_flash_attention: bool = True,
        flash_chunk_size: int = 2048,
    ):
        """
        Initialize Ring Dilated Attention with Flash Attention support.

        Additional args:
            use_flash_attention: Whether to use Flash Attention when available
            flash_chunk_size: Chunk size for memory-efficient processing
        """
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
            memory_pool_threshold_mb=memory_pool_threshold_mb,
        )

        self.use_flash_attention = use_flash_attention and HAS_FLASH_UTILS
        self.flash_chunk_size = flash_chunk_size

        # Check Flash Attention support for this device
        if self.use_flash_attention:
            self.flash_support = get_flash_attention_support(self.device)
            self.flash_backend = self.flash_support["recommended_backend"]

            # Log attention backend configuration
            if self.flash_backend == "xformers":
                print(
                    f"Using xformers for GPU {torch.cuda.get_device_name(self.device)} "
                    f"(compute {self.flash_support['compute_capability']})"
                )
            elif self.flash_backend == "sdpa":
                print(
                    f"Using PyTorch SDPA for GPU {torch.cuda.get_device_name(self.device)} "
                    f"(compute {self.flash_support['compute_capability']})"
                )
            elif self.flash_backend != "standard":
                print(f"Using {self.flash_backend} for attention computation")
            else:
                warnings.warn(
                    f"No optimized attention backend available for {torch.cuda.get_device_name(self.device)} "
                    f"(compute {self.flash_support['compute_capability']}). Using standard attention.",
                    RuntimeWarning,
                )

    def _compute_attention_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        chunk_offset: int = 0,
    ) -> Tensor:
        """
        Compute attention for a chunk using optimized backend.

        Args:
            q, k, v: Query, key, value tensors [batch, seq_len, num_heads, head_dim]
            is_causal: Whether to use causal masking
            chunk_offset: Offset for causal masking in chunked processing

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        try:
            # Adjust causal mask for chunk offset
            adjusted_causal = is_causal and chunk_offset == 0

            # Use optimized attention backend
            output = flash_attention_forward(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=adjusted_causal,
                backend=self.flash_backend,
            )
            return output

        except Exception as e:
            warnings.warn(
                f"Optimized attention backend failed, falling back to standard: {e}",
                stacklevel=2,
            )
            # Fallback to standard computation
            return self._compute_attention_standard(q, k, v, is_causal, chunk_offset)

    def _compute_attention_standard(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
        chunk_offset: int = 0,
    ) -> Tensor:
        """Standard attention computation (fallback)."""
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask if needed
        if is_causal:
            if chunk_offset > 0:
                # For chunks, adjust the causal mask
                causal_mask = torch.ones(
                    seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool
                )
                for i in range(seq_len_q):
                    for j in range(seq_len_kv):
                        if i + chunk_offset < j:
                            causal_mask[i, j] = False
                scores.masked_fill_(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )
            else:
                # Standard causal mask
                causal_mask = torch.triu(
                    torch.ones(
                        seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                scores.masked_fill_(
                    causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                )

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)

        # Apply attention
        output = torch.matmul(attn_weights, v)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)

        return output

    def _compute_chunk_attention_with_online_softmax_flash(
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
        """
        Compute chunk attention using Flash Attention with online softmax.

        This is a more memory-efficient version that uses Flash Attention
        when possible and falls back to chunked processing for OOM prevention.
        """
        b, n, h, d = q_dilated.shape
        _, n_kv, _, _ = k_chunk.shape

        # For small chunks, use Flash Attention directly
        if n_kv <= self.flash_chunk_size and self.use_flash_attention:
            try:
                # Flash Attention handles softmax internally, so we need a different approach
                # We'll compute raw scores for online softmax compatibility
                q_t = q_dilated.transpose(1, 2)
                k_t = k_chunk.transpose(1, 2)

                # Get attention scores (not normalized)
                scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(d)

                # Apply causal mask
                if is_causal:
                    causal_mask = torch.triu(
                        torch.ones(n, n_kv, device=q_dilated.device, dtype=torch.bool),
                        diagonal=chunk_start + 1,
                    )
                    scores.masked_fill_(
                        causal_mask.unsqueeze(0).unsqueeze(1), float("-inf")
                    )

                # Online softmax update
                chunk_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, chunk_max)

                # Rescale existing output
                if step > 0:
                    output.mul_(torch.exp(running_max - new_max))

                # Update running sum
                running_sum.mul_(torch.exp(running_max - new_max))
                running_sum.add_(torch.exp(scores - new_max).sum(dim=-1, keepdim=True))

                # Update running max
                running_max.copy_(new_max)

                # Use Flash Attention for the actual computation
                # Reshape for Flash Attention [batch, seq_len, num_heads, head_dim]
                q_flash = q_dilated  # Already in correct shape
                k_flash = k_chunk
                v_flash = v_chunk

                # Compute normalized attention for this chunk
                chunk_output_flash = self._compute_attention_chunk(
                    q_flash, k_flash, v_flash, is_causal, chunk_start
                )

                # We need to adjust the Flash Attention output for online softmax
                # Flash Attention returns normalized output, but we need unnormalized for online softmax
                # So we'll scale it by the local normalization factor
                exp_scores = torch.exp(scores - new_max)
                local_sum = exp_scores.sum(dim=-1, keepdim=True)

                # Transpose Flash output to match our format
                chunk_output_flash_t = chunk_output_flash.transpose(1, 2)

                # Scale by local normalization
                chunk_output_scaled = chunk_output_flash_t * local_sum

                # Add to output
                output.add_(chunk_output_scaled)

                return

            except Exception:
                # Fall back to standard computation
                pass

        # Standard computation (original implementation)
        self._compute_chunk_attention_with_online_softmax(
            q_dilated,
            k_chunk,
            v_chunk,
            chunk_start,
            is_causal,
            running_max,
            running_sum,
            output,
            step,
        )

    def _apply_dilated_attention_pattern(
        self, query: Tensor, key: Tensor, value: Tensor, is_causal: bool
    ) -> Tensor:
        """
        Apply dilated attention pattern with optimized attention backend.

        For single GPU mode without ring processing, we can use optimized backends
        more efficiently without the online softmax constraints.
        """
        if not is_causal:
            # For non-causal attention, we can use optimized backend directly
            try:
                return self._compute_attention_chunk(query, key, value, is_causal)
            except Exception:
                # Fall back to parent implementation
                pass

        # Use parent implementation for causal or when optimized backend fails
        return super()._apply_dilated_attention_pattern(query, key, value, is_causal)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass with Flash Attention optimization.

        Automatically handles:
        1. GPU architecture detection
        2. Flash Attention 3/2/1 selection
        3. Fallback for unsupported GPUs
        4. Memory-efficient processing
        """
        # For very long sequences, use chunked processing to prevent OOM
        seq_len = q.shape[1]

        if seq_len > 16384 and self.use_flash_attention:
            # Use chunked Flash Attention for very long sequences
            try:
                return chunked_flash_attention(
                    q,
                    k,
                    v,
                    chunk_size=self.flash_chunk_size,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            except Exception as e:
                warnings.warn(f"Chunked Flash Attention failed: {e}", stacklevel=2)

        # Use standard forward pass (with Flash Attention in chunks if enabled)
        return super().forward(q, k, v, is_causal)
