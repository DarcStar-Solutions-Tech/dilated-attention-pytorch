#!/usr/bin/env python3
"""
Simplified Hilbert attention with custom backward pass demonstration.

This version uses a hybrid approach: Triton for forward, PyTorch for optimized backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Tuple
import time


@triton.jit
def hilbert_attention_fwd_simple(
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
    N,
    D,
    # Parameters
    scale,
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Simple forward kernel for Hilbert attention."""
    # Program IDs
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N
    mask_d = offs_d < D

    # Get Hilbert positions for queries
    hilbert_pos_q = tl.load(hilbert_map + offs_m, mask=mask_m, other=0)

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos_q[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    # Initialize accumulators
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32) + 1e-10

    # Segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys with dilation
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset
        mask_k = (key_pos < N) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            # Get Hilbert position for key
            key_hilbert = tl.load(hilbert_map + key_pos, mask=mask_k, other=0)

            # Load key and value
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

            # Compute scores
            scores = tl.sum(q * k[None, :], axis=1)
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            # Stable softmax
            scores_max = tl.max(scores, axis=0)
            scores_exp = tl.exp(scores - scores_max)

            # Update accumulator
            acc += scores_exp[:, None] * v[None, :]
            norm += scores_exp

    # Normalize
    out = acc / norm[:, None]

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


class HilbertAttentionFunc(torch.autograd.Function):
    """Custom autograd function with optimized backward pass."""

    @staticmethod
    def forward(
        ctx, q, k, v, scale, hilbert_map, inv_hilbert_map, segment_size, dilation_rate
    ):
        B, H, N, D = q.shape

        # Allocate output
        out = torch.empty_like(q)

        # Configure kernel
        BLOCK_M = 64
        BLOCK_D = min(64, D)

        grid = (triton.cdiv(N, BLOCK_M), B * H)

        # Launch forward kernel
        hilbert_attention_fwd_simple[grid](
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
            N,
            D,
            scale,
            segment_size,
            dilation_rate,
            BLOCK_M,
            BLOCK_D,
            num_warps=4,
        )

        # Save for backward (we'll save the reordered tensors)
        # This is the key optimization: reorder once and save
        q_reordered = q.gather(2, hilbert_map[None, None, :, None].expand(B, H, N, D))
        k_reordered = k.gather(2, hilbert_map[None, None, :, None].expand(B, H, N, D))
        v_reordered = v.gather(2, hilbert_map[None, None, :, None].expand(B, H, N, D))

        ctx.save_for_backward(
            q_reordered, k_reordered, v_reordered, out, hilbert_map, inv_hilbert_map
        )
        ctx.scale = scale
        ctx.segment_size = segment_size
        ctx.dilation_rate = dilation_rate

        return out

    @staticmethod
    def backward(ctx, dout):
        q_reordered, k_reordered, v_reordered, out, hilbert_map, inv_hilbert_map = (
            ctx.saved_tensors
        )
        B, H, N, D = q_reordered.shape
        scale = ctx.scale
        segment_size = ctx.segment_size
        dilation_rate = ctx.dilation_rate

        # Key insight: Use PyTorch's optimized operations for the backward pass
        # but with pre-reordered tensors to avoid repeated reordering

        # Reshape for batch matrix multiply
        q_r = q_reordered.reshape(B * H, N, D) * scale
        k_r = k_reordered.reshape(B * H, N, D)
        v_r = v_reordered.reshape(B * H, N, D)
        dout_flat = dout.reshape(B * H, N, D)
        _ = out.reshape(B * H, N, D)

        # Initialize gradients
        dq_reordered = torch.zeros_like(q_r)
        dk_reordered = torch.zeros_like(k_r)
        dv_reordered = torch.zeros_like(v_r)

        # Process each segment
        for i in range(0, N, segment_size):
            seg_end = min(i + segment_size, N)

            # Create attention mask for dilation
            if dilation_rate > 1:
                mask = torch.zeros(seg_end - i, dtype=torch.bool, device=q_r.device)
                mask[::dilation_rate] = True
                attn_mask = mask.unsqueeze(0).expand(seg_end - i, -1)
            else:
                attn_mask = None

            # Extract segment
            q_seg = q_r[:, i:seg_end]
            k_seg = k_r[:, i:seg_end]
            v_seg = v_r[:, i:seg_end]
            dout_seg = dout_flat[:, i:seg_end]

            # Recompute attention weights for this segment
            scores = torch.bmm(q_seg, k_seg.transpose(-2, -1))

            if attn_mask is not None:
                scores.masked_fill_(~attn_mask.unsqueeze(0), float("-inf"))

            # Stable softmax
            attn_weights = F.softmax(scores, dim=-1)

            # Gradient w.r.t values: dV = A^T @ dO
            dv_reordered[:, i:seg_end] += torch.bmm(
                attn_weights.transpose(-2, -1), dout_seg
            )

            # Gradient w.r.t attention weights
            dattn = torch.bmm(dout_seg, v_seg.transpose(-2, -1))

            # Softmax backward
            dattn_weights = attn_weights * (
                dattn - (dattn * attn_weights).sum(dim=-1, keepdim=True)
            )

            # Gradient w.r.t queries and keys
            dq_reordered[:, i:seg_end] += torch.bmm(dattn_weights, k_seg) * scale
            dk_reordered[:, i:seg_end] += torch.bmm(
                dattn_weights.transpose(-2, -1), q_seg
            )

        # Reshape back
        dq_reordered = dq_reordered.reshape(B, H, N, D)
        dk_reordered = dk_reordered.reshape(B, H, N, D)
        dv_reordered = dv_reordered.reshape(B, H, N, D)

        # Reverse the Hilbert reordering using inverse map
        dq = dq_reordered.gather(
            2, inv_hilbert_map[None, None, :, None].expand(B, H, N, D)
        )
        dk = dk_reordered.gather(
            2, inv_hilbert_map[None, None, :, None].expand(B, H, N, D)
        )
        dv = dv_reordered.gather(
            2, inv_hilbert_map[None, None, :, None].expand(B, H, N, D)
        )

        return dq, dk, dv, None, None, None, None, None


def create_hilbert_mapping_simple(seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create Hilbert mapping and its inverse."""
    # Simple snake pattern
    mapping = torch.arange(seq_len, dtype=torch.long)
    inverse = torch.zeros(seq_len, dtype=torch.long)

    grid_size = int(math.ceil(math.sqrt(seq_len)))
    idx = 0

    for row in range(grid_size):
        if row % 2 == 0:
            for col in range(grid_size):
                if idx < seq_len:
                    linear_pos = row * grid_size + col
                    if linear_pos < seq_len:
                        mapping[idx] = linear_pos
                        inverse[linear_pos] = idx
                        idx += 1
        else:
            for col in range(grid_size - 1, -1, -1):
                if idx < seq_len:
                    linear_pos = row * grid_size + col
                    if linear_pos < seq_len:
                        mapping[idx] = linear_pos
                        inverse[linear_pos] = idx
                        idx += 1

    return mapping.long(), inverse.long()


class HilbertAttentionTritonSimple(nn.Module):
    """Simplified Hilbert attention with optimized backward."""

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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cache for mappings
        self._mapping_cache = {}

    def get_mappings(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached Hilbert mappings."""
        if seq_len not in self._mapping_cache:
            mapping, inverse = create_hilbert_mapping_simple(seq_len)
            self._mapping_cache[seq_len] = (mapping.to(device), inverse.to(device))
        return self._mapping_cache[seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with custom backward."""
        B, M, D = x.shape
        H = self.num_heads

        # Pad if needed
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

        # Get mappings
        hilbert_map, inv_hilbert_map = self.get_mappings(M_padded, x.device)

        # Apply custom function
        out = HilbertAttentionFunc.apply(
            q,
            k,
            v,
            self.scale,
            hilbert_map,
            inv_hilbert_map,
            self.segment_size,
            self.dilation_rate,
        )

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, M_padded, D)

        # Remove padding
        if M_padded > M:
            out = out[:, :M, :]

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


def benchmark_backward_performance():
    """Benchmark the backward pass performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test configuration
    batch_size = 2
    seq_len = 2048
    hidden_dim = 768
    num_heads = 12

    print("Benchmarking Hilbert Attention Backward Pass")
    print("=" * 60)

    # Create models
    model_optimized = HilbertAttentionTritonSimple(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=128,
        dilation_rate=1,
    ).to(device)

    # For comparison, use the original
    from .hilbert_dilated_attention_triton_fixed import HilbertAttentionTritonFixed

    model_original = HilbertAttentionTritonFixed(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=128,
        dilation_rate=1,
    ).to(device)

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

    if device.type == "cuda":
        print("\nWarming up...")
        # Warmup
        for _ in range(5):
            x.grad = None
            out = model_optimized(x)
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()

        print("\nOptimized Implementation:")
        # Time forward
        start = time.perf_counter()
        for _ in range(20):
            out = model_optimized(x)
            torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / 20 * 1000

        # Time backward
        start = time.perf_counter()
        for _ in range(20):
            x.grad = None
            out = model_optimized(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / 20 * 1000
        bwd_time = total_time - fwd_time

        print(f"  Forward: {fwd_time:.2f}ms")
        print(f"  Backward: {bwd_time:.2f}ms")
        print(f"  Ratio: {bwd_time / fwd_time:.2f}x")

        print("\nOriginal Implementation:")
        # Time original forward
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                out = model_original(x)
            torch.cuda.synchronize()
        orig_fwd_time = (time.perf_counter() - start) / 10 * 1000

        # Time forward + backward
        start = time.perf_counter()
        for _ in range(5):  # Fewer iterations as it's slower
            x.grad = None
            out = model_original(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        orig_total_time = (time.perf_counter() - start) / 5 * 1000
        orig_bwd_time = orig_total_time - orig_fwd_time

        print(f"  Forward: {orig_fwd_time:.2f}ms")
        print(f"  Backward: {orig_bwd_time:.2f}ms")
        print(f"  Ratio: {orig_bwd_time / orig_fwd_time:.2f}x")

        print(f"\n✅ Backward speedup: {orig_bwd_time / bwd_time:.2f}x faster!")

    else:
        print("(Skipping benchmarks on CPU)")


if __name__ == "__main__":
    benchmark_backward_performance()
