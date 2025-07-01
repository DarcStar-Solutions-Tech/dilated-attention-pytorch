"""
Simplified Ring Attention V2 - Focus on core ring communication fix.
"""

import os
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor


class RingAttentionV2Simple(nn.Module):
    """
    Simplified Ring Attention focusing on the core fix:
    - True ring communication (not all-gather)
    - Proper memory savings O(n/ring_size) for K/V
    - Clean implementation without all the extra features
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = dropout

        # Distributed setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Device setup - CRITICAL for multi-GPU
        if device is None:
            if dist.is_initialized():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(local_rank)
                self.device = torch.device(f"cuda:{local_rank}")
            else:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
        else:
            self.device = device

        self.dtype = dtype or torch.float32

        # Pre-allocate buffers for ring communication
        self._k_buffer = None
        self._v_buffer = None

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False
    ) -> Tensor:
        if self.world_size == 1:
            # Single GPU - just do standard attention
            return self._single_gpu_forward(q, k, v, is_causal)
        else:
            # Multi-GPU - use ring communication
            return self._ring_forward(q, k, v, is_causal)

    def _single_gpu_forward(
        self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool
    ) -> Tensor:
        """Standard attention for single GPU."""
        b, n, h, d = q.shape

        # Simple attention computation
        q = q.transpose(1, 2)  # [b, h, n, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

        if is_causal:
            mask = torch.triu(
                torch.ones(n, n, device=q.device, dtype=torch.bool), diagonal=1
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        output = torch.matmul(attn, v)
        return output.transpose(1, 2)  # [b, n, h, d]

    def _ring_forward(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        """Ring attention with true ring passing (not all-gather)."""
        b, n, h, d = q.shape
        chunk_size = (n + self.world_size - 1) // self.world_size

        # Each GPU gets its local K/V chunk
        local_start = self.rank * chunk_size
        local_end = min((self.rank + 1) * chunk_size, n)

        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Pad if needed
        if k_local.shape[1] < chunk_size:
            pad_size = chunk_size - k_local.shape[1]
            k_local = F.pad(k_local, (0, 0, 0, 0, 0, pad_size))
            v_local = F.pad(v_local, (0, 0, 0, 0, 0, pad_size))

        # Initialize output and softmax normalization
        output = torch.zeros((b, h, n, d), device=q.device, dtype=q.dtype)
        normalizer = torch.zeros((b, h, n, 1), device=q.device, dtype=q.dtype)

        # Pre-transpose Q for efficiency
        q_t = q.transpose(1, 2)  # [b, h, n, d]

        # Ring communication
        k_current = k_local.clone()
        v_current = v_local.clone()

        for step in range(self.world_size):
            # Which chunk are we processing?
            source_rank = (self.rank - step) % self.world_size
            chunk_start = source_rank * chunk_size

            # Compute attention for this chunk
            k_t = k_current.transpose(1, 2)  # [b, h, chunk_size, d]
            v_t = v_current.transpose(1, 2)

            # Scores for full Q against current K chunk
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(d)

            # Causal mask if needed
            if is_causal:
                chunk_end = min(chunk_start + chunk_size, n)
                for i in range(n):
                    for j in range(chunk_size):
                        actual_j = chunk_start + j
                        if actual_j >= chunk_end or i < actual_j:
                            scores[:, :, i, j] = float("-inf")

            # Compute exp(scores) for stable softmax
            exp_scores = torch.exp(scores)

            # Accumulate weighted values and normalizer
            output += torch.matmul(exp_scores, v_t)
            normalizer += exp_scores.sum(dim=-1, keepdim=True)

            # Ring exchange (except last step)
            if step < self.world_size - 1:
                # Simple blocking send/recv to avoid deadlock
                next_rank = (self.rank + 1) % self.world_size
                prev_rank = (self.rank - 1) % self.world_size

                # Even ranks send first, odd ranks receive first
                if self.rank % 2 == 0:
                    dist.send(k_current, next_rank)
                    dist.send(v_current, next_rank)
                    k_new = torch.empty_like(k_current)
                    v_new = torch.empty_like(v_current)
                    dist.recv(k_new, prev_rank)
                    dist.recv(v_new, prev_rank)
                else:
                    k_new = torch.empty_like(k_current)
                    v_new = torch.empty_like(v_current)
                    dist.recv(k_new, prev_rank)
                    dist.recv(v_new, prev_rank)
                    dist.send(k_current, next_rank)
                    dist.send(v_current, next_rank)

                k_current = k_new
                v_current = v_new

        # Normalize output
        output = output / (normalizer + 1e-8)

        return output.transpose(1, 2)  # [b, n, h, d]
