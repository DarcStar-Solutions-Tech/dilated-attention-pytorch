#!/usr/bin/env python3
"""
Optimized Hilbert Attention that applies reordering only to sparse dilated positions.
This implementation creates Hilbert patterns for the subset of positions actually accessed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict, Tuple

from .cache_manager import BoundedCache


@triton.jit
def hilbert_sparse_attention_kernel(
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
    use_hilbert: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized kernel that applies Hilbert reordering only to dilated positions.

    Key optimization: We compute Hilbert ordering on-the-fly for sparse positions
    instead of using a precomputed full-sequence map.
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

    # Number of sparse positions in segment
    sparse_positions_per_segment = segment_size // dilation_rate

    # Process sparse positions directly
    for sparse_block_idx in range(0, sparse_positions_per_segment, BLOCK_N):
        # Sparse indices within segment
        sparse_offs = sparse_block_idx + tl.arange(0, BLOCK_N)

        # Apply Hilbert reordering to sparse indices if enabled
        if use_hilbert:
            # Simple bit-reversal pattern for Hilbert approximation
            # This gives good cache locality without complex loops
            hilbert_sparse_offs = sparse_offs
            # Bit reversal pattern
            hilbert_sparse_offs = ((hilbert_sparse_offs & 0x55555555) << 1) | (
                (hilbert_sparse_offs & 0xAAAAAAAA) >> 1
            )
            hilbert_sparse_offs = ((hilbert_sparse_offs & 0x33333333) << 2) | (
                (hilbert_sparse_offs & 0xCCCCCCCC) >> 2
            )
            hilbert_sparse_offs = ((hilbert_sparse_offs & 0x0F0F0F0F) << 4) | (
                (hilbert_sparse_offs & 0xF0F0F0F0) >> 4
            )

            # Modulo to keep within range
            hilbert_sparse_offs = hilbert_sparse_offs % sparse_positions_per_segment

            # Convert sparse Hilbert indices to actual positions
            actual_positions = seg_start + hilbert_sparse_offs * dilation_rate
        else:
            # Direct sparse positions without reordering
            actual_positions = seg_start + sparse_offs * dilation_rate

        # Mask for valid positions
        mask_n = (
            (sparse_offs < sparse_positions_per_segment)
            & (actual_positions < M)
            & (actual_positions < seg_end)
        )

        # Load K and V at sparse positions
        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + actual_positions[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + actual_positions[None, :] * stride_vn
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


class HilbertAttentionSparseOptimized(nn.Module):
    """
    Optimized Hilbert Attention that applies reordering only to sparse positions.

    Key improvements over original:
    1. No full-sequence Hilbert map - only sparse positions
    2. Direct iteration over dilated positions
    3. On-the-fly Hilbert computation for better memory efficiency
    4. Simplified kernel logic with less masking overhead
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

        # Cache for sparse Hilbert patterns (much smaller than full sequence)
        # Initialize bounded cache for sparse patterns
        self._sparse_hilbert_cache = BoundedCache(
            max_size=32,
            max_memory_mb=50.0,
            name=f"{self.__class__.__name__}_sparse_hilbert",
        )

    def create_sparse_hilbert_pattern(self, num_sparse_positions: int) -> torch.Tensor:
        """
        Create Hilbert pattern for sparse positions only.
        This is much more memory efficient than full sequence mapping.
        """
        if num_sparse_positions <= 1:
            return torch.tensor([0], dtype=torch.long)

        # Find smallest power of 2 >= num_sparse_positions
        n = 1
        while n * n < num_sparse_positions:
            n *= 2

        # Generate Hilbert curve for n x n grid
        _ = []

        def hilbert_index_to_xy(index: int, n: int) -> Tuple[int, int]:
            x = y = 0
            s = 1
            while s < n:
                rx = 1 & (index // 2)
                ry = 1 & (index ^ rx)
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

        # Generate positions
        positions = []
        for i in range(n * n):
            x, y = hilbert_index_to_xy(i, n)
            linear_pos = y * n + x
            if linear_pos < num_sparse_positions:
                positions.append((i, linear_pos))

        # Sort by Hilbert index and extract positions
        positions.sort(key=lambda p: p[0])
        pattern = torch.tensor([p[1] for p in positions], dtype=torch.long)

        return pattern

    def get_sparse_hilbert_pattern(
        self, num_sparse_positions: int, device: torch.device
    ) -> torch.Tensor:
        """Get cached sparse Hilbert pattern."""
        # Try to get from cache
        pattern = self._sparse_hilbert_cache.get(num_sparse_positions)

        if pattern is None or pattern.device != device:
            # Create new pattern
            pattern = self.create_sparse_hilbert_pattern(num_sparse_positions).to(
                device
            )
            # Store in cache (will handle LRU eviction if needed)
            self._sparse_hilbert_cache.put(num_sparse_positions, pattern)

        return pattern

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """Forward pass with sparse Hilbert optimization."""
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

        # Handle dtype conversion
        compute_dtype = torch.float32 if qkv.dtype == torch.float16 else qkv.dtype
        original_dtype = qkv.dtype
        if qkv.dtype == torch.float16:
            qkv = qkv.to(compute_dtype)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Check if we can use Triton
        use_triton = (
            self.head_dim >= 16
            and M_padded >= 16
            and x.device.type == "cuda"
            and self.segment_size // self.dilation_rate >= 16
        )

        if use_triton:
            # Allocate output
            out = torch.zeros_like(q)

            # Configure grid
            BLOCK_M = min(64, M_padded)
            BLOCK_N = min(64, self.segment_size // self.dilation_rate)
            BLOCK_D = min(64, self.head_dim)

            # Ensure minimum block sizes
            BLOCK_M = max(16, BLOCK_M)
            BLOCK_N = max(16, BLOCK_N)
            BLOCK_D = max(16, BLOCK_D)

            grid = (triton.cdiv(M_padded, BLOCK_M) * B * H,)

            # Launch optimized kernel
            hilbert_sparse_attention_kernel[grid](
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
                use_hilbert,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
            )
        else:
            # PyTorch fallback
            out = self._pytorch_sparse_forward(q, k, v, use_hilbert)

        # Convert back to original dtype
        if original_dtype == torch.float16:
            out = out.to(original_dtype)

        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M_padded, self.hidden_dim)
        out = self.out_proj(out)

        # Remove padding
        if M != M_padded:
            out = out[:, :M, :]

        return self.dropout(out)

    def _pytorch_sparse_forward(self, q, k, v, use_hilbert):
        """PyTorch implementation with sparse Hilbert optimization."""
        B, H, M, D = q.shape
        out = torch.zeros_like(q)

        # Process each segment
        for seg_idx in range(0, M, self.segment_size):
            seg_end = min(seg_idx + self.segment_size, M)
            seg_len = seg_end - seg_idx

            # Get queries for this segment
            q_seg = q[:, :, seg_idx:seg_end, :]

            # Calculate sparse positions
            num_sparse = seg_len // self.dilation_rate
            if num_sparse == 0:
                continue

            # Get sparse indices
            sparse_indices = torch.arange(num_sparse, device=q.device)

            if use_hilbert and num_sparse > 1:
                # Apply Hilbert reordering to sparse indices
                hilbert_pattern = self.get_sparse_hilbert_pattern(num_sparse, q.device)
                sparse_indices = hilbert_pattern[:num_sparse]

            # Convert to actual positions
            actual_positions = seg_idx + sparse_indices * self.dilation_rate

            # Ensure positions are within bounds
            mask = actual_positions < seg_end
            actual_positions = actual_positions[mask]

            if len(actual_positions) == 0:
                continue

            # Get K and V at sparse positions
            k_seg = k[:, :, actual_positions, :]
            v_seg = v[:, :, actual_positions, :]

            # Compute attention
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            out_seg = torch.matmul(attn_weights, v_seg)
            out[:, :, seg_idx:seg_end, :] = out_seg

        return out

    def get_memory_savings(self, seq_len: int) -> Dict[str, int]:
        """Calculate memory savings compared to original implementation."""
        # Original: full sequence Hilbert map
        original_map_size = seq_len

        # Optimized: sparse maps per segment
        num_segments = (seq_len + self.segment_size - 1) // self.segment_size
        sparse_positions_per_segment = self.segment_size // self.dilation_rate
        optimized_map_size = sparse_positions_per_segment  # Only need one cached

        return {
            "original_map_entries": original_map_size,
            "optimized_map_entries": optimized_map_size,
            "memory_reduction_factor": original_map_size / optimized_map_size,
            "positions_accessed": num_segments * sparse_positions_per_segment,
            "sparsity": 1 - (1 / self.dilation_rate),
        }


def test_sparse_optimized():
    """Test the sparse optimized implementation."""
    print("Testing Sparse Optimized Hilbert Attention")
    print("=" * 60)

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 1024
    segment_size = 256
    dilation_rate = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create module
    module = HilbertAttentionSparseOptimized(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    # Memory analysis
    savings = module.get_memory_savings(seq_len)
    print("\nMemory Analysis:")
    print(f"  Original Hilbert map size: {savings['original_map_entries']} entries")
    print(f"  Optimized map size: {savings['optimized_map_entries']} entries")
    print(f"  Memory reduction: {savings['memory_reduction_factor']:.1f}x")
    print(f"  Sparsity: {savings['sparsity']:.1%}")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        # Without Hilbert
        out_no_hilbert = module(x, use_hilbert=False)
        print(f"  Without Hilbert: {out_no_hilbert.shape}")

        # With Hilbert
        out_with_hilbert = module(x, use_hilbert=True)
        print(f"  With Hilbert: {out_with_hilbert.shape}")

        # Compare
        diff = (out_with_hilbert - out_no_hilbert).abs().max().item()
        print(f"  Max difference: {diff:.6f}")

    # Benchmark
    import time

    print("\nBenchmarking (10 iterations)...")

    # Warmup
    for _ in range(3):
        _ = module(x, use_hilbert=False)
        _ = module(x, use_hilbert=True)

    if device == "cuda":
        torch.cuda.synchronize()

    # Without Hilbert
    start = time.perf_counter()
    for _ in range(10):
        _ = module(x, use_hilbert=False)
    if device == "cuda":
        torch.cuda.synchronize()
    time_no_hilbert = (time.perf_counter() - start) / 10 * 1000

    # With Hilbert
    start = time.perf_counter()
    for _ in range(10):
        _ = module(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    time_with_hilbert = (time.perf_counter() - start) / 10 * 1000

    print(f"  Without Hilbert: {time_no_hilbert:.2f}ms")
    print(f"  With Hilbert: {time_with_hilbert:.2f}ms")
    print(
        f"  Overhead: {(time_with_hilbert - time_no_hilbert) / time_no_hilbert * 100:.1f}%"
    )

    print("\nSparse optimized implementation complete!")


if __name__ == "__main__":
    test_sparse_optimized()
