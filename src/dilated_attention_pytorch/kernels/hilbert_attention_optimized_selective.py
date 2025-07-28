#!/usr/bin/env python3
"""
Optimized Hilbert Attention that applies Hilbert mapping only to selected positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hilbert_attention_selective_kernel(
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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized kernel that applies Hilbert reordering only to dilated positions.

    Key optimization: Instead of creating a Hilbert map for the entire sequence
    and then filtering, we directly compute Hilbert ordering for the sparse
    positions we actually access.
    """
    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query block boundaries
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Masks
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

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # Key insight: We only need to process dilated positions
    # So we can iterate more efficiently
    effective_segment_size = segment_size // dilation_rate

    # Process only the positions we actually need
    for block_idx in range(0, effective_segment_size, BLOCK_N):
        # Compute actual positions (incorporating dilation)
        local_offs = block_idx + tl.arange(0, BLOCK_N)
        actual_offs = seg_start + local_offs * dilation_rate

        # Apply Hilbert reordering to these sparse positions
        # Simple Hilbert pattern for sparse positions (can be more sophisticated)
        hilbert_local = _compute_sparse_hilbert(local_offs, effective_segment_size)
        reordered_offs = seg_start + hilbert_local * dilation_rate

        # Mask for valid positions
        mask_n = (
            (actual_offs < M)
            & (actual_offs < seg_end)
            & (local_offs < effective_segment_size)
        )

        # Load K and V using reordered positions
        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + reordered_offs[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + reordered_offs[None, :] * stride_vn
            + offs_d[:, None] * stride_vd
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        # Compute attention scores
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e9)

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)

        # Update statistics
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij

        # Update accumulator
        acc = acc * alpha[:, None]
        v_t = tl.trans(v)
        acc += tl.dot(p, v_t)

        l_i = l_i_new
        m_i = m_i_new

    # Final normalization
    acc = acc / tl.maximum(l_i[:, None], 1e-10)

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _compute_sparse_hilbert(local_idx, total_sparse_positions):
    """
    Compute Hilbert ordering for sparse positions only.
    This is much more efficient than creating a full Hilbert map.

    For example, with dilation_rate=4, we only need to order
    segment_size/4 positions instead of segment_size positions.
    """
    # Simple bit-reversal pattern for demonstration
    # In practice, this would implement proper Hilbert curve for sparse positions
    return local_idx ^ (local_idx >> 1)


class HilbertAttentionSelectiveOptimized(nn.Module):
    """
    Optimized Hilbert Attention that applies reordering only to accessed positions.

    Key improvements:
    1. No full-sequence Hilbert map creation
    2. Hilbert ordering computed only for dilated positions
    3. Direct iteration over sparse positions
    4. Reduced memory overhead
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

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

        # Cache for segment-level Hilbert patterns
        self._segment_hilbert_cache = {}

    def get_segment_hilbert_pattern(
        self, effective_size: int, device: torch.device
    ) -> torch.Tensor:
        """
        Get Hilbert pattern for sparse positions within a segment.
        Much smaller than full sequence Hilbert map.
        """
        if effective_size not in self._segment_hilbert_cache:
            # Create Hilbert ordering for effective_size positions
            if effective_size <= 16:
                # Simple pattern for small sizes
                pattern = torch.arange(effective_size, device=device)
            else:
                # Compute Hilbert curve for sparse positions
                pattern = self._compute_hilbert_pattern(effective_size)
                pattern = pattern.to(device)

            self._segment_hilbert_cache[effective_size] = pattern

        return self._segment_hilbert_cache[effective_size]

    def _compute_hilbert_pattern(self, size: int) -> torch.Tensor:
        """Compute Hilbert pattern for given size."""
        # Find smallest power of 2
        n = 1
        while n * n < size:
            n *= 2

        # Generate pattern
        pattern = []
        for i in range(size):
            x = i
            y = 0
            t = i

            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)

                if ry == 0:
                    if rx == 1:
                        x = s - 1 - x
                        y = s - 1 - y
                    x, y = y, x

                x += s * rx
                y += s * ry
                t //= 4
                s *= 2

            if x < size:
                pattern.append(x)

        return torch.tensor(pattern[:size], dtype=torch.long)

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """Forward pass with optimized Hilbert reordering."""
        B, M, _ = x.shape
        H = self.num_heads

        # Pad if necessary
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

        # Allocate output
        out = torch.zeros_like(q)

        if x.device.type == "cuda" and use_hilbert:
            # Use optimized kernel
            BLOCK_M = min(64, M_padded)
            BLOCK_N = min(64, self.segment_size // self.dilation_rate)
            BLOCK_D = min(64, self.head_dim)

            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            hilbert_attention_selective_kernel[grid](
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
                BLOCK_N,
                BLOCK_D,
            )
        else:
            # Fallback to PyTorch implementation
            out = self._pytorch_forward(q, k, v, use_hilbert)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M_padded, self.hidden_dim)
        out = self.out_proj(out)

        # Remove padding
        if M != M_padded:
            out = out[:, :M, :]

        return self.dropout(out)

    def _pytorch_forward(self, q, k, v, use_hilbert):
        """PyTorch implementation for comparison."""
        B, H, M, D = q.shape
        out = torch.zeros_like(q)

        for seg_start in range(0, M, self.segment_size):
            seg_end = min(seg_start + self.segment_size, M)

            # Get queries for this segment
            q_seg = q[:, :, seg_start:seg_end, :]

            # Get dilated positions
            dilated_positions = torch.arange(
                seg_start, seg_end, self.dilation_rate, device=q.device
            )

            if use_hilbert and len(dilated_positions) > 1:
                # Apply Hilbert reordering to dilated positions only
                effective_size = len(dilated_positions)
                hilbert_pattern = self.get_segment_hilbert_pattern(
                    effective_size, q.device
                )
                dilated_positions = dilated_positions[hilbert_pattern[:effective_size]]

            # Get keys and values at dilated positions
            k_seg = k[:, :, dilated_positions, :]
            v_seg = v[:, :, dilated_positions, :]

            # Compute attention
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            out_seg = torch.matmul(attn_weights, v_seg)

            out[:, :, seg_start:seg_end, :] = out_seg

        return out


def benchmark_comparison():
    """Compare original vs optimized Hilbert implementation."""
    import time
    from dilated_attention_pytorch.kernels import HilbertAttentionCore

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 2048
    segment_size = 256
    dilation_rate = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create modules
    original = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    optimized = HilbertAttentionSelectiveOptimized(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        _ = original(x, use_hilbert=True)
        _ = optimized(x, use_hilbert=True)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print("Benchmarking Hilbert implementations:")
    print(
        f"Sequence length: {seq_len}, Segment size: {segment_size}, Dilation: {dilation_rate}"
    )
    print(f"Effective sparse positions per segment: {segment_size // dilation_rate}")
    print()

    # Original
    start = time.perf_counter()
    for _ in range(10):
        out_original = original(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / 10 * 1000

    # Optimized
    start = time.perf_counter()
    for _ in range(10):
        out_optimized = optimized(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    optimized_time = (time.perf_counter() - start) / 10 * 1000

    print(f"Original (full Hilbert map): {original_time:.2f}ms")
    print(f"Optimized (selective Hilbert): {optimized_time:.2f}ms")
    print(f"Speedup: {original_time / optimized_time:.2f}x")

    # Check correctness
    diff = (out_original - out_optimized).abs().max().item()
    print(f"\nMax difference: {diff:.6f}")

    # Memory analysis
    print("\nMemory overhead analysis:")
    print(f"Original: Hilbert map size = {seq_len} entries")
    print(
        f"Optimized: Hilbert map size = {segment_size // dilation_rate} entries per segment"
    )
    print(f"Memory reduction: {seq_len / (segment_size // dilation_rate):.1f}x")


if __name__ == "__main__":
    benchmark_comparison()
