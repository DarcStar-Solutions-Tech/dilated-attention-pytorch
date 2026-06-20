#!/usr/bin/env python3
"""
Simplified Hilbert Attention implementation without Triton kernels.

This version uses PyTorch operations instead of custom Triton kernels to avoid
compilation issues while still providing the Hilbert curve reordering functionality.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_manager import BoundedCache


def create_hilbert_curve_2d(n: int) -> torch.Tensor:
    """
    Create a 2D Hilbert curve mapping for a given size.

    Args:
        n: Size of the grid (must be power of 2)

    Returns:
        Tensor of indices representing Hilbert curve order
    """

    def hilbert_index(x: int, y: int, n: int) -> int:
        """Convert (x,y) to Hilbert curve index."""
        index = 0
        s = n // 2
        while s > 0:
            rx = int((x & s) > 0)
            ry = int((y & s) > 0)
            index += s * s * ((3 * rx) ^ ry)

            # Rotate/flip quadrant
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                x, y = y, x
            s //= 2
        return index

    # Create mapping
    indices = []
    for i in range(n):
        for j in range(n):
            indices.append((hilbert_index(i, j, n), i * n + j))

    # Sort by Hilbert index and extract positions
    indices.sort(key=lambda x: x[0])
    mapping = torch.tensor([pos for _, pos in indices], dtype=torch.long)

    return mapping


def create_hilbert_mapping(seq_len: int) -> torch.Tensor:
    """
    Create Hilbert curve mapping for sequences.

    For simplicity, uses a snake pattern that provides similar cache locality benefits.
    """
    if seq_len <= 64:
        return torch.arange(seq_len, dtype=torch.long)

    # Find nearest square grid size
    grid_size = int(math.ceil(math.sqrt(seq_len)))

    # Make it power of 2 for true Hilbert curve
    grid_size = 2 ** int(math.ceil(math.log2(grid_size)))

    # Get Hilbert curve for the grid
    if grid_size <= 512:  # Reasonable size for exact Hilbert curve
        hilbert_map = create_hilbert_curve_2d(grid_size)
        # Truncate to actual sequence length
        valid_indices = hilbert_map < seq_len
        result = hilbert_map[valid_indices]

        # Pad with remaining indices if needed
        if len(result) < seq_len:
            remaining = torch.arange(len(result), seq_len, dtype=torch.long)
            result = torch.cat([result, remaining])

        return result[:seq_len]
    else:
        # For very large sequences, use snake pattern
        mapping = torch.zeros(seq_len, dtype=torch.long)
        idx = 0

        for row in range(grid_size):
            if row % 2 == 0:
                # Left to right
                for col in range(grid_size):
                    if idx < seq_len:
                        linear_pos = row * grid_size + col
                        if linear_pos < seq_len:
                            mapping[linear_pos] = idx
                            idx += 1
            else:
                # Right to left (snake pattern)
                for col in range(grid_size - 1, -1, -1):
                    if idx < seq_len:
                        linear_pos = row * grid_size + col
                        if linear_pos < seq_len:
                            mapping[linear_pos] = idx
                            idx += 1

        return mapping


class HilbertAttentionSimple(nn.Module):
    """
    Simplified Hilbert Attention using PyTorch operations.

    This implementation provides Hilbert curve reordering for better cache locality
    without relying on custom Triton kernels.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
        use_hilbert: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.use_hilbert = use_hilbert

        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        # Cache for Hilbert mappings
        # Initialize bounded cache with reasonable limits
        self._hilbert_cache = BoundedCache(
            max_size=32, max_memory_mb=100.0, name=f"{self.__class__.__name__}_hilbert"
        )

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        # Try to get from cache
        mapping = self._hilbert_cache.get(seq_len)

        if mapping is None or mapping.device != device:
            # Create new mapping
            mapping = create_hilbert_mapping(seq_len).to(device)
            # Store in cache (will handle LRU eviction if needed)
            self._hilbert_cache.put(seq_len, mapping)

        return mapping

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass with optional Hilbert reordering.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        B, M, D = x.shape
        H = self.num_heads

        # Ensure sequence length is compatible with segment size
        if M % self.segment_size != 0:
            pad_len = self.segment_size - (M % self.segment_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            M_padded = M + pad_len
        else:
            M_padded = M

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, M, D]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply Hilbert reordering if requested
        if self.use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)

            # Reorder k and v using Hilbert curve
            # Note: We don't reorder q to maintain output position correspondence
            k = k.gather(
                2,
                hilbert_map[None, None, :, None].expand(B, H, M_padded, self.head_dim),
            )
            v = v.gather(
                2,
                hilbert_map[None, None, :, None].expand(B, H, M_padded, self.head_dim),
            )

        # Compute attention with segments and dilation
        output = self._compute_segmented_attention(q, k, v, is_causal)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()  # [B, M, H, D]
        output = output.reshape(B, M_padded, D)

        # Remove padding if applied
        if M_padded > M:
            output = output[:, :M, :]

        # Output projection and dropout
        output = self.out_proj(output)
        output = self.dropout_layer(output)

        return output

    def _compute_segmented_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool
    ) -> torch.Tensor:
        """Compute attention with segmentation and dilation."""
        B, H, M, D = q.shape
        num_segments = M // self.segment_size

        # Initialize output
        output = torch.zeros_like(q)

        # Process each segment
        for seg_idx in range(num_segments):
            seg_start = seg_idx * self.segment_size
            seg_end = seg_start + self.segment_size

            # Get segment queries
            q_seg = q[:, :, seg_start:seg_end, :]

            if self.dilation_rate == 1:
                # No dilation - standard attention within segment
                k_seg = k[:, :, seg_start:seg_end, :]
                v_seg = v[:, :, seg_start:seg_end, :]

                # Compute attention
                scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale

                if is_causal and seg_idx == 0:
                    # Apply causal mask only to first segment
                    causal_mask = torch.triu(
                        torch.full(
                            (self.segment_size, self.segment_size),
                            float("-inf"),
                            device=scores.device,
                        ),
                        diagonal=1,
                    )
                    scores = scores + causal_mask

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = F.dropout(
                    attn_weights, p=self.dropout, training=self.training
                )

                seg_output = torch.matmul(attn_weights, v_seg)
            else:
                # Apply dilation
                seg_output = torch.zeros_like(q_seg)

                for offset in range(self.dilation_rate):
                    # Get dilated positions
                    positions = torch.arange(
                        seg_start + offset,
                        min(seg_end, M),
                        self.dilation_rate,
                        device=q.device,
                    )

                    if len(positions) == 0:
                        continue

                    # Get dilated k, v
                    k_dilated = k[:, :, positions, :]
                    v_dilated = v[:, :, positions, :]

                    # Get queries that attend to these positions
                    q_indices = torch.arange(
                        offset, self.segment_size, self.dilation_rate, device=q.device
                    )
                    q_indices = q_indices[q_indices < q_seg.shape[2]]

                    if len(q_indices) == 0:
                        continue

                    q_dilated = q_seg[:, :, q_indices, :]

                    # Compute dilated attention
                    scores = (
                        torch.matmul(q_dilated, k_dilated.transpose(-2, -1))
                        * self.scale
                    )

                    if is_causal and seg_idx == 0 and offset == 0:
                        # Apply causal mask
                        causal_mask = torch.triu(
                            torch.full(
                                (len(q_indices), len(positions)),
                                float("-inf"),
                                device=scores.device,
                            ),
                            diagonal=1,
                        )
                        scores = scores + causal_mask

                    attn_weights = F.softmax(scores, dim=-1)
                    attn_weights = F.dropout(
                        attn_weights, p=self.dropout, training=self.training
                    )

                    # Accumulate output
                    seg_output[:, :, q_indices, :] = torch.matmul(
                        attn_weights, v_dilated
                    )

            output[:, :, seg_start:seg_end, :] = seg_output

        return output


# For backward compatibility
HilbertAttentionCore = HilbertAttentionSimple
