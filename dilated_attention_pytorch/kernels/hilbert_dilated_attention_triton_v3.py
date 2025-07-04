#!/usr/bin/env python3
"""
Simplified Hilbert curve ordered dilated attention kernel using Triton.
Fixed to work with Triton's constraints.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def hilbert_attention_kernel_simple(
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
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    # Shape
    B,
    H,
    M,
    N,
    D,
    # Parameters
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Simplified Hilbert attention kernel."""
    # Program ID
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Mask
    mask_m = offs_m < M

    # Get Hilbert positions for this block
    hilbert_pos = tl.load(hilbert_map + offs_m, mask=mask_m, other=0)

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos[:, None] * stride_qm
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute segment boundaries
    seg_idx = hilbert_pos // segment_size
    seg_start = seg_idx * segment_size

    # Process keys in the segment with dilation
    max_keys = segment_size // dilation_rate
    for i in range(max_keys):
        key_idx = seg_start + i * dilation_rate

        # Bounds check
        valid = (key_idx < N) & (key_idx < seg_start + segment_size)

        # Get Hilbert position for key
        key_hilbert = tl.load(hilbert_map + key_idx, mask=valid, other=0)

        # Load key and value
        k_ptrs = (
            K + pid_b * stride_kb + pid_h * stride_kh + key_hilbert * stride_kn + offs_d
        )
        v_ptrs = (
            V + pid_b * stride_vb + pid_h * stride_vh + key_hilbert * stride_vn + offs_d
        )

        k = tl.load(k_ptrs, mask=valid, other=0.0)
        v = tl.load(v_ptrs, mask=valid, other=0.0)

        # Compute attention score
        score = tl.sum(q * k[None, :], 1)
        score = tl.where(valid, score, -1e9)

        # Accumulate (simplified - no stable softmax for clarity)
        weight = tl.exp(score)
        acc += weight[:, None] * v[None, :]
        norm += weight

    # Normalize
    acc = acc / (norm[:, None] + 1e-6)

    # Store output
    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + hilbert_pos[:, None] * stride_om
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def standard_attention_kernel_simple(
    # Pointers
    Q,
    K,
    V,
    Out,
    # Strides
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    # Shape
    B,
    H,
    M,
    N,
    D,
    # Parameters
    segment_size: tl.constexpr,
    dilation_rate: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Standard attention kernel for comparison."""
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M

    # Load queries
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Initialize
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    norm = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Compute segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size

    # Process keys with dilation
    max_keys = segment_size // dilation_rate
    for i in range(max_keys):
        key_idx = seg_start + i * dilation_rate

        valid = (key_idx < N) & (key_idx < seg_start + segment_size)

        k_ptrs = (
            K + pid_b * stride_kb + pid_h * stride_kh + key_idx * stride_kn + offs_d
        )
        v_ptrs = (
            V + pid_b * stride_vb + pid_h * stride_vh + key_idx * stride_vn + offs_d
        )

        k = tl.load(k_ptrs, mask=valid, other=0.0)
        v = tl.load(v_ptrs, mask=valid, other=0.0)

        score = tl.sum(q * k[None, :], 1)
        score = tl.where(valid, score, -1e9)

        weight = tl.exp(score)
        acc += weight[:, None] * v[None, :]
        norm += weight

    acc = acc / (norm[:, None] + 1e-6)

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None])


def generate_hilbert_mapping_simple(seq_len: int) -> torch.Tensor:
    """Generate simple Hilbert curve mapping."""
    # Simplified Hilbert mapping for testing
    # In practice, would use full Hilbert curve
    grid_size = int(math.ceil(math.sqrt(seq_len)))

    _ = torch.arange(seq_len)

    # Simple shuffle to simulate Hilbert ordering
    # Real implementation would compute actual Hilbert curve
    indices = []
    for i in range(0, seq_len, grid_size):
        row = list(range(i, min(i + grid_size, seq_len)))
        if (i // grid_size) % 2 == 1:
            row.reverse()
        indices.extend(row)

    return torch.tensor(indices[:seq_len], dtype=torch.int32)


class HilbertAttentionTritonSimple(nn.Module):
    """Simplified Hilbert attention using Triton."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Cache
        self._hilbert_cache = {}

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len not in self._hilbert_cache:
            self._hilbert_cache[seq_len] = generate_hilbert_mapping_simple(seq_len).to(
                device
            )
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        B, M, D = x.shape
        H = self.num_heads

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale queries
        q = q * (self.head_dim**-0.5)

        # Output tensor
        out = torch.zeros_like(q)

        # Grid
        BLOCK_M = 32
        BLOCK_D = min(self.head_dim, 64)
        grid = (triton.cdiv(M, BLOCK_M), B, H)

        if use_hilbert:
            hilbert_map = self.get_hilbert_mapping(M, x.device)
            hilbert_attention_kernel_simple[grid](
                q,
                k,
                v,
                out,
                hilbert_map,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                B,
                H,
                M,
                M,
                self.head_dim,
                self.segment_size,
                self.dilation_rate,
                BLOCK_M,
                BLOCK_D,
            )
        else:
            standard_attention_kernel_simple[grid](
                q,
                k,
                v,
                out,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                B,
                H,
                M,
                M,
                self.head_dim,
                self.segment_size,
                self.dilation_rate,
                BLOCK_M,
                BLOCK_D,
            )

        # Reshape output
        out = out.transpose(1, 2).reshape(B, M, D)
        out = self.out_proj(out)

        return out


def benchmark_triton_implementation():
    """Benchmark the Triton implementation."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("=== Benchmarking Hilbert Attention with Triton ===\n")

    # Test configurations
    configs = [
        (512, 128, 1, 4),  # seq_len, segment_size, dilation_rate, batch_size
        (512, 128, 2, 4),
        (512, 128, 4, 4),
        (1024, 256, 1, 2),
        (1024, 256, 2, 2),
        (1024, 256, 4, 2),
        (2048, 512, 2, 1),
        (2048, 512, 4, 1),
    ]

    hidden_dim = 256
    num_heads = 8

    print(
        "Config: batch_size x seq_len | Segment | Dilation | Hilbert (ms) | Standard (ms) | Speedup"
    )
    print("-" * 90)

    for seq_len, segment_size, dilation_rate, batch_size in configs:
        model = HilbertAttentionTritonSimple(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        ).cuda()

        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(x, use_hilbert=True)
                _ = model(x, use_hilbert=False)

        torch.cuda.synchronize()

        # Benchmark
        iterations = 50

        # Hilbert timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x, use_hilbert=True)
        end_event.record()
        torch.cuda.synchronize()
        hilbert_time = start_event.elapsed_time(end_event) / iterations

        # Standard timing
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(x, use_hilbert=False)
        end_event.record()
        torch.cuda.synchronize()
        standard_time = start_event.elapsed_time(end_event) / iterations

        speedup = standard_time / hilbert_time

        print(
            f"{batch_size}x{seq_len:4d}             | {segment_size:7} | {dilation_rate:8} | "
            f"{hilbert_time:12.2f} | {standard_time:13.2f} | {speedup:7.2f}x"
        )

    print("\nNote: This is a simplified implementation for demonstration.")
    print("A full implementation would use proper Hilbert curve generation and")
    print("more optimized kernels with stable softmax.")


if __name__ == "__main__":
    benchmark_triton_implementation()
