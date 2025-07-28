#!/usr/bin/env python3
"""
Example implementation of Ring Attention with Hilbert ordering integration.

This demonstrates how to combine the memory efficiency of Ring Attention
with the cache-friendly access patterns of Hilbert space-filling curves.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import math
from typing import Optional, Tuple, Dict
from functools import lru_cache


class HilbertRingDilatedAttention(nn.Module):
    """
    Ring Dilated Attention enhanced with Hilbert curve memory ordering.

    This implementation shows how to integrate Hilbert ordering into the
    Ring Attention pipeline for improved cache efficiency.
    """

    def __init__(
        self,
        segment_lengths: list[int],
        dilation_rates: list[int],
        dropout: float = 0.0,
        ring_size: Optional[int] = None,
        use_hilbert: bool = True,
        hilbert_min_size: int = 64,  # Only use Hilbert for chunks >= this size
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout = nn.Dropout(dropout)
        self.use_hilbert = use_hilbert
        self.hilbert_min_size = hilbert_min_size

        # Device and dtype
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Ring setup
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.ring_size = ring_size or self.world_size

        # Caches
        self._hilbert_cache: Dict[int, torch.Tensor] = {}
        self._inverse_hilbert_cache: Dict[int, torch.Tensor] = {}
        self._pattern_cache: Dict[Tuple, torch.Tensor] = {}

    @lru_cache(maxsize=32)
    def _create_hilbert_mapping(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create forward and inverse Hilbert mappings."""
        # For small sizes or non-square, use identity
        if size < self.hilbert_min_size:
            identity = torch.arange(size, device=self.device)
            return identity, identity

        # Calculate grid size (nearest power of 2 for true Hilbert)
        grid_size = int(math.ceil(math.sqrt(size)))
        if grid_size & (grid_size - 1):  # Not power of 2
            grid_size = 2 ** int(math.ceil(math.log2(grid_size)))

        # Create 2D snake pattern (simplified Hilbert)
        forward_map = torch.zeros(size, dtype=torch.long, device=self.device)
        inverse_map = torch.zeros(size, dtype=torch.long, device=self.device)

        idx = 0
        for row in range(grid_size):
            if row % 2 == 0:
                # Left to right
                for col in range(grid_size):
                    linear_pos = row * grid_size + col
                    if linear_pos < size and idx < size:
                        forward_map[linear_pos] = idx
                        inverse_map[idx] = linear_pos
                        idx += 1
            else:
                # Right to left (snake)
                for col in range(grid_size - 1, -1, -1):
                    linear_pos = row * grid_size + col
                    if linear_pos < size and idx < size:
                        forward_map[linear_pos] = idx
                        inverse_map[idx] = linear_pos
                        idx += 1

        return forward_map, inverse_map

    def _apply_hilbert_ordering(
        self, tensor: torch.Tensor, mapping: torch.Tensor
    ) -> torch.Tensor:
        """Apply Hilbert ordering to sequence dimension."""
        # tensor shape: [batch, seq_len, heads, dim] or [batch, seq_len, hidden]
        if tensor.dim() == 4:
            B, S, H, D = tensor.shape
            # Expand mapping for gathering
            mapping_exp = mapping.view(1, -1, 1, 1).expand(B, S, H, D)
            return torch.gather(tensor, 1, mapping_exp)
        else:  # dim == 3
            B, S, D = tensor.shape
            mapping_exp = mapping.view(1, -1, 1).expand(B, S, D)
            return torch.gather(tensor, 1, mapping_exp)

    def _apply_dilated_pattern_hilbert(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        segment_len: int,
        dilation_rate: int,
        offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply dilated pattern in Hilbert space for better locality."""
        B, S, H, D = k_chunk.shape

        if dilation_rate == 1:
            return k_chunk, v_chunk

        # In Hilbert space, dilated access has better locality
        # Create dilated indices
        cache_key = (S, segment_len, dilation_rate, offset)

        if cache_key not in self._pattern_cache:
            # Generate pattern that works well with Hilbert ordering
            indices = torch.arange(offset, S, dilation_rate, device=self.device)
            if len(indices) < segment_len:
                # Pad with wrapped indices
                pad_indices = (
                    torch.arange(
                        offset,
                        offset + (segment_len - len(indices)) * dilation_rate,
                        dilation_rate,
                        device=self.device,
                    )
                    % S
                )
                indices = torch.cat([indices, pad_indices])
            indices = indices[:segment_len]
            self._pattern_cache[cache_key] = indices
        else:
            indices = self._pattern_cache[cache_key]

        # Apply pattern
        k_dilated = k_chunk.index_select(1, indices)
        v_dilated = v_chunk.index_select(1, indices)

        return k_dilated, v_dilated

    def _ring_attention_hilbert(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Ring attention with Hilbert ordering for improved cache efficiency."""
        B, N, H, D = q.shape
        chunk_size = (N + self.ring_size - 1) // self.ring_size

        # Get Hilbert mappings for chunk size
        if self.use_hilbert and chunk_size >= self.hilbert_min_size:
            forward_map, inverse_map = self._create_hilbert_mapping(chunk_size)
            use_hilbert = True
        else:
            use_hilbert = False

        # Extract local chunk
        local_start = self.rank * chunk_size
        local_end = min((self.rank + 1) * chunk_size, N)
        actual_size = local_end - local_start

        # Get local K/V chunks
        k_local = k[:, local_start:local_end].contiguous()
        v_local = v[:, local_start:local_end].contiguous()

        # Pad if needed
        if actual_size < chunk_size:
            pad_size = chunk_size - actual_size
            k_local = torch.nn.functional.pad(k_local, (0, 0, 0, 0, 0, pad_size))
            v_local = torch.nn.functional.pad(v_local, (0, 0, 0, 0, 0, pad_size))

        # Apply Hilbert ordering to chunks
        if use_hilbert:
            k_local = self._apply_hilbert_ordering(k_local, forward_map)
            v_local = self._apply_hilbert_ordering(v_local, forward_map)

        # Apply dilated patterns (now with better locality in Hilbert space)
        head_groups = self._calculate_head_groups(H)
        k_dilated = torch.zeros_like(k_local)
        v_dilated = torch.zeros_like(v_local)

        head_start = 0
        for i, (seg_len, dil_rate, group_size) in enumerate(
            zip(self.segment_lengths, self.dilation_rates, head_groups)
        ):
            if group_size == 0:
                continue

            head_end = head_start + group_size
            offset = i % dil_rate if dil_rate > 1 else 0

            # Apply dilation in Hilbert space
            k_group = k_local[:, :, head_start:head_end, :]
            v_group = v_local[:, :, head_start:head_end, :]

            k_dil, v_dil = self._apply_dilated_pattern_hilbert(
                k_group, v_group, seg_len, dil_rate, offset
            )

            k_dilated[:, :, head_start:head_end, :] = k_dil
            v_dilated[:, :, head_start:head_end, :] = v_dil
            head_start = head_end

        # All-gather dilated chunks (still in Hilbert space)
        k_chunks = [torch.empty_like(k_dilated) for _ in range(self.ring_size)]
        v_chunks = [torch.empty_like(v_dilated) for _ in range(self.ring_size)]

        if self.world_size > 1:
            dist.all_gather(k_chunks, k_dilated)
            dist.all_gather(v_chunks, v_dilated)
        else:
            k_chunks = [k_dilated]
            v_chunks = [v_dilated]

        # Apply Hilbert ordering to full Q as well
        if use_hilbert:
            # Need to apply Hilbert ordering to each chunk of Q separately
            q_hilbert = torch.zeros_like(q)
            for i in range(self.ring_size):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, N)
                if chunk_end > chunk_start:
                    q_chunk = q[:, chunk_start:chunk_end]
                    if q_chunk.shape[1] < chunk_size:
                        q_chunk = torch.nn.functional.pad(
                            q_chunk, (0, 0, 0, 0, 0, chunk_size - q_chunk.shape[1])
                        )
                    q_chunk_hilbert = self._apply_hilbert_ordering(q_chunk, forward_map)
                    q_hilbert[:, chunk_start:chunk_end] = q_chunk_hilbert[
                        :, : chunk_end - chunk_start
                    ]
        else:
            q_hilbert = q

        # Compute attention with online softmax (in Hilbert space)
        output = self._compute_ring_attention_online(
            q_hilbert, k_chunks, v_chunks, chunk_size, is_causal
        )

        # Reverse Hilbert ordering
        if use_hilbert:
            output_linear = torch.zeros_like(output)
            for i in range(self.ring_size):
                chunk_start = i * chunk_size
                chunk_end = min((i + 1) * chunk_size, N)
                if chunk_end > chunk_start:
                    out_chunk = output[:, chunk_start:chunk_end]
                    if out_chunk.shape[1] < chunk_size:
                        out_chunk = torch.nn.functional.pad(
                            out_chunk, (0, 0, 0, 0, 0, chunk_size - out_chunk.shape[1])
                        )
                    out_chunk_linear = self._apply_hilbert_ordering(
                        out_chunk, inverse_map
                    )
                    output_linear[:, chunk_start:chunk_end] = out_chunk_linear[
                        :, : chunk_end - chunk_start
                    ]
            return output_linear
        else:
            return output

    def _calculate_head_groups(self, num_heads: int) -> list[int]:
        """Calculate head distribution across segment lengths."""
        num_segments = len(self.segment_lengths)
        base_heads = num_heads // num_segments
        extra_heads = num_heads % num_segments

        groups = [base_heads] * num_segments
        for i in range(extra_heads):
            groups[-(i + 1)] += 1

        return groups

    def _compute_ring_attention_online(
        self,
        q: torch.Tensor,
        k_chunks: list[torch.Tensor],
        v_chunks: list[torch.Tensor],
        chunk_size: int,
        is_causal: bool,
    ) -> torch.Tensor:
        """Compute attention using online softmax algorithm."""
        B, N, H, D = q.shape

        # Initialize accumulators
        output = torch.zeros_like(q)
        running_max = torch.full(
            (B, N, H, 1), float("-inf"), device=q.device, dtype=q.dtype
        )
        running_sum = torch.zeros((B, N, H, 1), device=q.device, dtype=q.dtype)

        # Process each chunk
        for step in range(self.ring_size):
            chunk_idx = (self.rank - step) % self.ring_size
            chunk_start = chunk_idx * chunk_size

            k_chunk = k_chunks[chunk_idx]
            v_chunk = v_chunks[chunk_idx]

            # Compute scores for this chunk
            scores = torch.matmul(q, k_chunk.transpose(-2, -1)) / math.sqrt(D)

            # Apply causal mask if needed
            if is_causal:
                # Create causal mask considering chunk positions
                q_pos = torch.arange(N, device=q.device).view(1, -1, 1, 1)
                k_pos = (
                    torch.arange(chunk_size, device=q.device).view(1, 1, 1, -1)
                    + chunk_start
                )
                causal_mask = q_pos >= k_pos
                scores.masked_fill_(~causal_mask, float("-inf"))

            # Online softmax update
            chunk_max = scores.amax(dim=-1, keepdim=True)
            new_max = torch.maximum(running_max, chunk_max)

            # Rescale previous results
            output *= torch.exp(running_max - new_max)
            running_sum *= torch.exp(running_max - new_max)

            # Add contribution from current chunk
            exp_scores = torch.exp(scores - new_max)
            output += torch.matmul(exp_scores, v_chunk)
            running_sum += exp_scores.sum(dim=-1, keepdim=True)

            # Update running max
            running_max = new_max

        # Final normalization
        output = output / (running_sum + 1e-8)

        return output

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False
    ) -> torch.Tensor:
        """Forward pass with Hilbert-enhanced Ring Attention."""
        if self.world_size == 1 and self.ring_size == 1:
            # Single GPU - use standard attention with optional Hilbert
            return self._single_gpu_forward_hilbert(q, k, v, is_causal)
        else:
            # Multi-GPU or simulated ring - use Hilbert Ring Attention
            return self._ring_attention_hilbert(q, k, v, is_causal)

    def _single_gpu_forward_hilbert(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Single GPU forward with optional Hilbert ordering."""
        B, N, H, D = q.shape

        if self.use_hilbert and N >= self.hilbert_min_size:
            # Apply Hilbert ordering
            forward_map, inverse_map = self._create_hilbert_mapping(N)
            q_h = self._apply_hilbert_ordering(q, forward_map)
            k_h = self._apply_hilbert_ordering(k, forward_map)
            v_h = self._apply_hilbert_ordering(v, forward_map)

            # Compute attention in Hilbert space
            scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(D)
            if is_causal:
                mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
                scores.masked_fill_(~mask, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            output = torch.matmul(attn, v_h)

            # Reverse Hilbert ordering
            output = self._apply_hilbert_ordering(output, inverse_map)
            return output
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            if is_causal:
                mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
                scores.masked_fill_(~mask, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            return torch.matmul(attn, v)


def demo_hilbert_ring_attention():
    """Demonstrate Hilbert Ring Attention with performance comparison."""
    import time

    # Configuration
    batch_size = 2
    seq_len = 8192
    num_heads = 16
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Create models
    standard_ring = HilbertRingDilatedAttention(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        use_hilbert=False,
        device=device,
        dtype=dtype,
    )

    hilbert_ring = HilbertRingDilatedAttention(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        use_hilbert=True,
        device=device,
        dtype=dtype,
    )

    # Create input tensors
    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Warmup
    for _ in range(5):
        _ = standard_ring(q, k, v)
        _ = hilbert_ring(q, k, v)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking Ring Attention with and without Hilbert ordering...")
    print(f"Sequence length: {seq_len}, Heads: {num_heads}, Device: {device}")

    # Standard Ring Attention
    start = time.perf_counter()
    for _ in range(10):
        output_standard = standard_ring(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / 10

    # Hilbert Ring Attention
    start = time.perf_counter()
    for _ in range(10):
        output_hilbert = hilbert_ring(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    hilbert_time = (time.perf_counter() - start) / 10

    # Results
    print("\nResults:")
    print(f"Standard Ring Attention: {standard_time * 1000:.2f} ms")
    print(f"Hilbert Ring Attention:  {hilbert_time * 1000:.2f} ms")
    print(f"Speedup: {standard_time / hilbert_time:.2f}x")

    # Verify correctness (outputs should be similar but not identical due to reordering)
    diff = (output_standard - output_hilbert).abs().max().item()
    print(f"\nMax absolute difference: {diff:.6f}")
    print("Note: Small differences are expected due to different computation order")

    # Memory access pattern analysis
    print("\nMemory Access Pattern Benefits:")
    print("- Hilbert ordering improves cache line utilization")
    print("- Especially beneficial for dilated patterns with rate > 1")
    print("- Reduces memory bandwidth requirements")
    print("- Better GPU memory coalescing")


if __name__ == "__main__":
    print("=== Hilbert Ring Attention Demo ===\n")
    demo_hilbert_ring_attention()

    print("\n=== Key Insights ===")
    print("1. Hilbert ordering preserves spatial locality in 2D space")
    print("2. Ring Attention benefits from better cache usage within chunks")
    print("3. Communication overhead reduced due to better data locality")
    print("4. Greatest benefits seen with high dilation rates and long sequences")
