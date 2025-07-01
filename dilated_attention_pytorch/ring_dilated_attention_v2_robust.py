"""
Ring Dilated Attention V2 Robust - Working ring communication with proper synchronization.
"""

import math
import os
from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RingDilatedAttentionV2Robust(nn.Module):
    """
    Robust Ring Dilated Attention with working ring communication.

    Key features:
    1. Proper ring passing with synchronized communication
    2. O(n/ring_size) memory for K/V
    3. No deadlocks or race conditions
    4. Fallback to single GPU when needed
    """

    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = True,
        min_seq_length_for_ring: int = 8192,
    ):
        super().__init__()

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout
        self.min_seq_length_for_ring = min_seq_length_for_ring

        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.ring_size = ring_size or self.world_size

        # Device setup
        if device is None:
            if dist.is_initialized() and torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = device

        # Dtype setup - smart selection based on GPU
        if dtype is not None:
            self.dtype = dtype
        else:
            if self.device.type == "cuda":
                compute_capability = torch.cuda.get_device_capability(self.device)
                # Pascal GPUs (6.x) don't support FP16 well
                self.dtype = (
                    torch.float32 if compute_capability[0] < 7 else torch.float16
                )
            else:
                self.dtype = torch.float32

        # Flash attention setup
        self.use_flash_attention = use_flash_attention
        self.has_flash = False
        self.flash_attention_forward = None

        if use_flash_attention:
            try:
                from .utils.flash_attention_utils import (
                    flash_attention_forward,
                    get_flash_attention_support,
                )

                self.flash_attention_forward = flash_attention_forward
                self.flash_support = get_flash_attention_support(self.device)
                # Check for any flash support
                self.has_flash = (
                    self.flash_support.get("flash3_available", False)
                    or self.flash_support.get("flash2_available", False)
                    or self.flash_support.get("flash_attn_available", False)
                )
            except (ImportError, KeyError, AttributeError):
                self.has_flash = False
                self.flash_attention_forward = None

        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def _calculate_head_groups(self, num_heads: int) -> List[int]:
        """Distribute heads across segment lengths."""
        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        head_groups = [base_heads] * num_segments
        for i in range(extra_heads):
            head_groups[-(i + 1)] += 1

        return head_groups

    def _apply_dilated_patterns(
        self, k: Tensor, v: Tensor, head_groups: List[int]
    ) -> tuple[Tensor, Tensor]:
        """Apply dilated patterns to K/V based on head groups."""
        b, n, h, d = k.shape

        k_dilated = torch.zeros_like(k)
        v_dilated = torch.zeros_like(v)

        head_start = 0
        for i, (segment_len, dilation_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, head_groups)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size

            if dilation_rate > 1 and n >= dilation_rate:
                # Apply dilation
                offset = i % dilation_rate
                dilated_indices = torch.arange(
                    offset, n, dilation_rate, device=k.device
                )

                # Handle wraparound
                if len(dilated_indices) < n:
                    repeats = (n + len(dilated_indices) - 1) // len(dilated_indices)
                    extended = dilated_indices.repeat(repeats)
                    dilated_indices = extended[:n] % n

                # Fix: Make slices contiguous before index_select to avoid CUDA errors
                k_heads = k[:, :, head_start:head_end].contiguous()
                v_heads = v[:, :, head_start:head_end].contiguous()
                k_dilated[:, :, head_start:head_end] = k_heads.index_select(
                    1, dilated_indices
                )
                v_dilated[:, :, head_start:head_end] = v_heads.index_select(
                    1, dilated_indices
                )
            else:
                # No dilation
                k_dilated[:, :, head_start:head_end] = k[:, :, head_start:head_end]
                v_dilated[:, :, head_start:head_end] = v[:, :, head_start:head_end]

            head_start = head_end

        return k_dilated, v_dilated

    def _single_gpu_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Standard attention for single GPU."""
        b, n, h, d = q.shape
        head_groups = self._calculate_head_groups(h)

        # Apply dilated patterns
        k_dilated, v_dilated = self._apply_dilated_patterns(k, v, head_groups)

        # Try Flash Attention first
        if self.has_flash and self.flash_attention_forward is not None:
            try:
                output = self.flash_attention_forward(
                    q,
                    k_dilated,
                    v_dilated,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=is_causal,
                    backend=self.flash_support.get("recommended_backend", "auto"),
                )
                return output
            except Exception:
                pass  # Fall through to standard

        # Standard PyTorch implementation
        q = q.transpose(1, 2)  # [b, h, n, d]
        k_dilated = k_dilated.transpose(1, 2)
        v_dilated = v_dilated.transpose(1, 2)

        scores = torch.matmul(q, k_dilated.transpose(-2, -1)) / math.sqrt(d)

        if is_causal:
            mask = torch.triu(
                torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout_layer and self.training:
            attn = self.dropout_layer(attn)

        output = torch.matmul(attn, v_dilated)
        return output.transpose(1, 2)  # [b, n, h, d]

    def _ring_forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        """Ring attention with proper synchronization."""
        b, n, h, d = q.shape
        chunk_size = (n + self.ring_size - 1) // self.ring_size
        head_groups = self._calculate_head_groups(h)

        # Get local K/V chunk
        local_start = self.rank * chunk_size
        local_end = min((self.rank + 1) * chunk_size, n)
        local_size = local_end - local_start

        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Apply dilation to local chunk
        k_local_dilated, v_local_dilated = self._apply_dilated_patterns(
            k_local, v_local, head_groups
        )

        # Pad to chunk_size if needed
        if local_size < chunk_size:
            pad_size = chunk_size - local_size
            k_local_dilated = F.pad(k_local_dilated, (0, 0, 0, 0, 0, pad_size))
            v_local_dilated = F.pad(v_local_dilated, (0, 0, 0, 0, 0, pad_size))

        # Initialize output with online softmax
        output = torch.zeros(b, h, n, d, device=q.device, dtype=q.dtype)
        max_scores = torch.full(
            (b, h, n, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        sum_exp = torch.zeros(b, h, n, 1, device=q.device, dtype=q.dtype)

        # Pre-transpose Q
        q_t = q.transpose(1, 2)  # [b, h, n, d]

        # Current K/V chunks
        k_current = k_local_dilated.clone()
        v_current = v_local_dilated.clone()

        # Process chunks in ring
        for step in range(self.ring_size):
            # Which chunk are we processing?
            source_rank = (self.rank - step) % self.ring_size
            chunk_start = source_rank * chunk_size

            # Trim padding for last chunk
            if source_rank == self.ring_size - 1 and n % chunk_size != 0:
                actual_size = n - (self.ring_size - 1) * chunk_size
                k_chunk = k_current[:, :actual_size]
                v_chunk = v_current[:, :actual_size]
            else:
                k_chunk = k_current
                v_chunk = v_current

            # Compute attention scores
            k_t = k_chunk.transpose(1, 2)  # [b, h, chunk_size, d]
            v_t = v_chunk.transpose(1, 2)

            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(d)

            # Apply causal mask if needed
            if is_causal:
                _ = chunk_start + k_chunk.shape[1]
                for i in range(n):
                    for j in range(k_chunk.shape[1]):
                        actual_col = chunk_start + j
                        if i < actual_col:
                            scores[:, :, i, j] = float("-inf")

            # Online softmax update
            new_max_scores = torch.maximum(
                max_scores, scores.amax(dim=-1, keepdim=True)
            )

            # Rescale previous results
            scale_factor = torch.exp(max_scores - new_max_scores)
            output = output * scale_factor
            sum_exp = sum_exp * scale_factor

            # Add new contribution
            exp_scores = torch.exp(scores - new_max_scores)
            output = output + torch.matmul(exp_scores, v_t)
            sum_exp = sum_exp + exp_scores.sum(dim=-1, keepdim=True)

            # Update max scores
            max_scores = new_max_scores

            # Ring exchange (except last step)
            if step < self.ring_size - 1:
                # Use P2P operations with proper synchronization
                next_rank = (self.rank + 1) % self.ring_size
                prev_rank = (self.rank - 1) % self.ring_size

                k_new = torch.empty_like(k_current)
                v_new = torch.empty_like(v_current)

                # Use async operations for better overlap
                send_k = dist.isend(k_current, next_rank)
                send_v = dist.isend(v_current, next_rank)
                recv_k = dist.irecv(k_new, prev_rank)
                recv_v = dist.irecv(v_new, prev_rank)

                # Wait for all operations to complete
                send_k.wait()
                send_v.wait()
                recv_k.wait()
                recv_v.wait()

                k_current = k_new
                v_current = v_new

        # Final normalization
        output = output / (sum_exp + 1e-8)

        return output.transpose(1, 2)  # [b, n, h, d]

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        """Forward pass with automatic routing."""
        seq_length = q.shape[1]

        # Use single GPU if:
        # 1. Not distributed
        # 2. Sequence too short
        # 3. Ring size is 1
        if (
            self.world_size == 1
            or self.ring_size == 1
            or seq_length < self.min_seq_length_for_ring
        ):
            return self._single_gpu_forward(q, k, v, is_causal)
        else:
            return self._ring_forward(q, k, v, is_causal)
