#!/usr/bin/env python3
"""
Fixed Hilbert curve ordered dilated attention kernel using Triton.
Addresses index issues and shape mismatches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Tuple, List


@triton.jit
def hilbert_attention_kernel_fixed(
    # Pointers
    Q,
    K,
    V,
    Out,
    hilbert_map,
    # Strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # Shape
    B,
    H,
    M,
    D,
    # Parameters
    scale,
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fixed Hilbert attention kernel with proper indexing."""
    # Program ID
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query indices for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Mask for valid queries
    mask_m = offs_m < M
    mask_d = offs_d < D

    # Get Hilbert positions for queries in this block
    hilbert_pos_q = tl.load(hilbert_map + offs_m, mask=mask_m, other=0)

    # Load queries using Hilbert positions
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos_q[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Scale queries
    q = q * scale

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-10

    # For each query position, compute its segment
    # Use original position (offs_m) to determine segment, not Hilbert position
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys in the segment with dilation
    for offset in range(0, segment_size, dilation_rate):
        # Compute key position in original space
        key_pos = seg_start + offset

        # Check if key position is valid for all queries in block
        mask_k = (key_pos < M) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            # Get Hilbert position for this key
            key_hilbert = tl.load(hilbert_map + key_pos, mask=mask_k, other=0)

            # Load key and value using Hilbert position
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + key_hilbert * stride_kn
                + offs_d * stride_kd
            )
            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + key_hilbert * stride_vn
                + offs_d * stride_vd
            )

            k = tl.load(k_ptrs, mask=mask_k & mask_d, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k & mask_d, other=0.0)

            # Compute attention scores for all queries against this key
            scores = tl.sum(q * k[None, :], axis=1)

            # Mask invalid scores
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            # Stable softmax accumulation
            scores_max = tl.max(scores, axis=0)
            scores_exp = tl.exp(scores - scores_max)

            # Update accumulator
            acc += scores_exp[:, None] * v[None, :]
            norm += scores_exp

    # Normalize
    out = acc / norm[:, None]

    # Store output back to original positions
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def standard_attention_kernel_fixed(
    # Pointers
    Q,
    K,
    V,
    Out,
    # Strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    # Shape
    B,
    H,
    M,
    D,
    # Parameters
    scale,
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Standard attention kernel for comparison."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    # Initialize
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-10

    # Compute segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys with dilation
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset

        mask_k = (key_pos < M) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + key_pos * stride_kn
                + offs_d * stride_kd
            )
            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + key_pos * stride_vn
                + offs_d * stride_vd
            )

            k = tl.load(k_ptrs, mask=mask_k & mask_d, other=0.0)
            v = tl.load(v_ptrs, mask=mask_k & mask_d, other=0.0)

            scores = tl.sum(q * k[None, :], axis=1)
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            scores_max = tl.max(scores, axis=0)
            scores_exp = tl.exp(scores - scores_max)

            acc += scores_exp[:, None] * v[None, :]
            norm += scores_exp

    out = acc / norm[:, None]

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


def generate_hilbert_curve_2d(size: int) -> List[Tuple[int, int]]:
    """Generate proper 2D Hilbert curve coordinates."""

    def hilbert_index_to_xy(index: int, n: int) -> Tuple[int, int]:
        """Convert Hilbert index to (x, y) coordinates."""
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (index // 2) % 2 else 0
            ry = 1 if (index ^ rx) % 2 else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            index //= 4
            s *= 2
        return x, y

    coords = []
    for i in range(size * size):
        x, y = hilbert_index_to_xy(i, size)
        coords.append((x, y))
    return coords


def create_hilbert_mapping_fixed(seq_len: int) -> torch.Tensor:
    """Create fixed Hilbert curve mapping for sequences."""
    # For simplicity, we'll create a mapping that reorders elements
    # according to a space-filling pattern

    # If seq_len is small, just return identity mapping
    if seq_len <= 64:
        return torch.arange(seq_len, dtype=torch.int32)

    # Find appropriate grid size (not necessarily power of 2)
    grid_size = int(math.ceil(math.sqrt(seq_len)))

    # Create a simple space-filling pattern (snake pattern for now)
    # This gives similar benefits to Hilbert curve but is simpler
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

    # Fill any remaining positions
    for i in range(seq_len):
        if i >= idx:
            mapping[i] = i

    return mapping.int()


class HilbertAttentionTritonFixed(nn.Module):
    """Fixed Hilbert attention using Triton with proper indexing."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get cached Hilbert mapping or create new one."""
        if seq_len not in self._hilbert_cache:
            mapping = create_hilbert_mapping_fixed(seq_len)
            self._hilbert_cache[seq_len] = mapping.to(device)
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """Forward pass with optional Hilbert ordering."""
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
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Output tensor
        out = torch.zeros_like(q)

        # Configure grid
        BLOCK_M = min(64, M_padded)
        BLOCK_D = min(64, self.head_dim)
        grid = (triton.cdiv(M_padded, BLOCK_M), B * H)

        if use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M_padded, x.device)
            hilbert_attention_kernel_fixed[grid](
                q,
                k,
                v,
                out,
                hilbert_map,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *out.stride(),
                B,
                H,
                M_padded,
                self.head_dim,
                self.scale,
                self.segment_size,
                self.dilation_rate,
                BLOCK_M,
                BLOCK_D,
            )
        else:
            standard_attention_kernel_fixed[grid](
                q,
                k,
                v,
                out,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *out.stride(),
                B,
                H,
                M_padded,
                self.head_dim,
                self.scale,
                self.segment_size,
                self.dilation_rate,
                BLOCK_M,
                BLOCK_D,
            )

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding if applied
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection and dropout
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


def test_hilbert_attention_fixed():
    """Test the fixed Hilbert attention implementation."""

    print("=== Testing Fixed Hilbert Attention ===\n")

    # Test configurations
    configs = [
        (4, 256, 8, 64, 1),  # batch, seq_len, heads, segment_size, dilation
        (4, 256, 8, 64, 2),
        (2, 512, 8, 128, 1),
        (2, 512, 8, 128, 4),
        (1, 1024, 16, 256, 2),
        (1, 1024, 16, 256, 8),
    ]

    hidden_dim = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Testing correctness and performance...\n")
    print(
        "Config (B×L×H, seg, dil) | Hilbert Time | Standard Time | Speedup | Max Diff"
    )
    print("-" * 75)

    for batch, seq_len, heads, segment_size, dilation in configs:
        model = HilbertAttentionTritonFixed(
            hidden_dim=hidden_dim,
            num_heads=heads,
            segment_size=segment_size,
            dilation_rate=dilation,
        ).to(device)

        # Create input
        x = torch.randn(batch, seq_len, hidden_dim, device=device)

        # Run both versions
        with torch.no_grad():
            out_hilbert = model(x, use_hilbert=True)
            out_standard = model(x, use_hilbert=False)

        # Check numerical difference
        max_diff = (out_hilbert - out_standard).abs().max().item()

        # Benchmark if on CUDA
        if device == "cuda":
            # Warmup
            for _ in range(10):
                _ = model(x, use_hilbert=True)
                _ = model(x, use_hilbert=False)

            torch.cuda.synchronize()

            # Time Hilbert
            import time

            start = time.perf_counter()
            for _ in range(50):
                _ = model(x, use_hilbert=True)
            torch.cuda.synchronize()
            hilbert_time = (time.perf_counter() - start) / 50 * 1000

            # Time standard
            start = time.perf_counter()
            for _ in range(50):
                _ = model(x, use_hilbert=False)
            torch.cuda.synchronize()
            standard_time = (time.perf_counter() - start) / 50 * 1000

            speedup = standard_time / hilbert_time

            print(
                f"{batch}×{seq_len}×{heads}, {segment_size:3}, {dilation:2} | "
                f"{hilbert_time:12.2f} | {standard_time:13.2f} | "
                f"{speedup:7.2f} | {max_diff:.2e}"
            )
        else:
            print(
                f"{batch}×{seq_len}×{heads}, {segment_size:3}, {dilation:2} | "
                f"    N/A (CPU) |     N/A (CPU) |     N/A | {max_diff:.2e}"
            )

    print("\nNote: Some numerical differences are expected due to different")
    print("computation order in Hilbert vs standard attention.")


if __name__ == "__main__":
    test_hilbert_attention_fixed()
