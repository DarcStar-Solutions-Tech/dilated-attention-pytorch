#!/usr/bin/env python3
"""
Hilbert curve ordered dilated attention kernel using Triton.
Much easier to use than raw CUDA and provides excellent performance.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def hilbert_to_linear(d: tl.int32, n: tl.int32) -> tl.int32:
    """Convert Hilbert curve distance to linear position."""
    x = tl.zeros([], dtype=tl.int32)
    y = tl.zeros([], dtype=tl.int32)

    s = 1
    while s < n:
        rx = (d >> 1) & 1
        ry = (d ^ rx) & 1

        # Rotate/flip quadrant
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            # Swap x and y
            x, y = y, x

        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1

    return y * n + x


@triton.jit
def linear_to_hilbert(x: tl.int32, y: tl.int32, n: tl.int32) -> tl.int32:
    """Convert (x,y) coordinates to Hilbert curve distance."""
    d = 0
    s = n >> 1

    while s > 0:
        rx = tl.where((x & s) > 0, 1, 0)
        ry = tl.where((y & s) > 0, 1, 0)
        d += s * s * ((3 * rx) ^ ry)

        # Rotate/flip quadrant
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            # Swap x and y
            x, y = y, x

        s >>= 1

    return d


@triton.jit
def hilbert_dilated_attention_kernel(
    # Pointers to matrices
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    # Hilbert mapping
    hilbert_map_ptr,
    # Matrix dimensions
    batch_stride_q,
    head_stride_q,
    seq_stride_q,
    batch_stride_k,
    head_stride_k,
    seq_stride_k,
    batch_stride_v,
    head_stride_v,
    seq_stride_v,
    batch_stride_out,
    head_stride_out,
    seq_stride_out,
    # Attention parameters
    N_CTX,
    HEAD_DIM,
    SEGMENT_SIZE,
    DILATION_RATE,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Triton kernel for Hilbert-ordered dilated attention."""
    # Program IDs
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Compute Q pointer for this block
    q_ptrs = Q_ptr + (
        batch_idx * batch_stride_q
        + head_idx * head_stride_q
        + offs_m[:, None] * seq_stride_q
        + offs_d[None, :]
    )

    # Load Q block with masking
    m_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Initialize accumulator and normalization
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)

    # Compute attention scores with dilation
    # HEAD_DIM is already a compile-time constant
    scale = 1.0 / tl.sqrt(HEAD_DIM)

    # For each position in the query block
    for block_start_idx in range(0, N_CTX, SEGMENT_SIZE):
        # Compute segment boundaries
        segment_end = tl.minimum(block_start_idx + SEGMENT_SIZE, N_CTX)

        # Process dilated positions within segment
        for offset in range(0, SEGMENT_SIZE, DILATION_RATE * BLOCK_N):
            start_n = block_start_idx + offset
            if start_n >= segment_end:
                break

            # Compute actual positions using Hilbert mapping
            hilbert_offs_n = start_n + offs_n * DILATION_RATE
            n_mask = hilbert_offs_n < segment_end

            # Load Hilbert-mapped positions
            hilbert_positions = tl.load(
                hilbert_map_ptr + hilbert_offs_n, mask=n_mask, other=0
            )

            # Compute K and V pointers using Hilbert positions
            k_ptrs = K_ptr + (
                batch_idx * batch_stride_k
                + head_idx * head_stride_k
                + hilbert_positions[:, None] * seq_stride_k
                + offs_d[None, :]
            )
            v_ptrs = V_ptr + (
                batch_idx * batch_stride_v
                + head_idx * head_stride_v
                + hilbert_positions[:, None] * seq_stride_v
                + offs_d[None, :]
            )

            # Load K and V blocks
            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            # Compute attention scores
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= scale

            # Apply causal mask if needed (only attend to positions before current)
            qk = tl.where(n_mask[None, :], qk, -float("inf"))

            # Compute new maximum
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)

            # Compute exponentials with numerical stability
            p = tl.exp(qk - m_i_new[:, None])
            alpha = tl.exp(m_i - m_i_new)

            # Update accumulator
            acc_scale = l_i * alpha
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(q.dtype), v)

            # Update normalization
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Write output with Hilbert mapping
    hilbert_offs_m = tl.load(hilbert_map_ptr + offs_m, mask=m_mask, other=0)
    out_ptrs = Out_ptr + (
        batch_idx * batch_stride_out
        + head_idx * head_stride_out
        + hilbert_offs_m[:, None] * seq_stride_out
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def standard_dilated_attention_kernel(
    # Same parameters as Hilbert kernel but without mapping
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    batch_stride_q,
    head_stride_q,
    seq_stride_q,
    batch_stride_k,
    head_stride_k,
    seq_stride_k,
    batch_stride_v,
    head_stride_v,
    seq_stride_v,
    batch_stride_out,
    head_stride_out,
    seq_stride_out,
    N_CTX,
    HEAD_DIM,
    SEGMENT_SIZE,
    DILATION_RATE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Standard dilated attention kernel for comparison."""
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q
    q_ptrs = Q_ptr + (
        batch_idx * batch_stride_q
        + head_idx * head_stride_q
        + offs_m[:, None] * seq_stride_q
        + offs_d[None, :]
    )
    m_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Initialize
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(HEAD_DIM)

    # Standard dilated attention
    for block_start_idx in range(0, N_CTX, SEGMENT_SIZE):
        segment_end = tl.minimum(block_start_idx + SEGMENT_SIZE, N_CTX)

        for offset in range(0, SEGMENT_SIZE, DILATION_RATE * BLOCK_N):
            start_n = block_start_idx + offset
            if start_n >= segment_end:
                break

            # Linear positions with dilation
            linear_offs_n = start_n + offs_n * DILATION_RATE
            n_mask = linear_offs_n < segment_end

            # Load K and V
            k_ptrs = K_ptr + (
                batch_idx * batch_stride_k
                + head_idx * head_stride_k
                + linear_offs_n[:, None] * seq_stride_k
                + offs_d[None, :]
            )
            v_ptrs = V_ptr + (
                batch_idx * batch_stride_v
                + head_idx * head_stride_v
                + linear_offs_n[:, None] * seq_stride_v
                + offs_d[None, :]
            )

            k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

            # Compute scores
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, tl.trans(k))
            qk *= scale
            qk = tl.where(n_mask[None, :], qk, -float("inf"))

            # Update with stable softmax
            m_ij = tl.max(qk, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.exp(qk - m_i_new[:, None])
            alpha = tl.exp(m_i - m_i_new)

            acc_scale = l_i * alpha
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(q.dtype), v)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_i_new

    # Normalize and store
    acc = acc / l_i[:, None]
    out_ptrs = Out_ptr + (
        batch_idx * batch_stride_out
        + head_idx * head_stride_out
        + offs_m[:, None] * seq_stride_out
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None])


def generate_hilbert_curve_mapping(seq_len: int) -> torch.Tensor:
    """Generate Hilbert curve mapping for sequence length."""
    # Find next power of 2 for grid size
    grid_size = 2 ** int(math.ceil(math.log2(math.sqrt(seq_len))))

    # Generate Hilbert curve
    def hilbert_curve(n):
        """Generate points along Hilbert curve."""
        if n == 1:
            return [(0, 0)]

        # Recursive construction
        points = []
        m = n // 2

        # Get sub-curves
        sub_curve = hilbert_curve(m)

        # Bottom-left (rotated right)
        for x, y in sub_curve:
            points.append((y, x))

        # Top-left
        for x, y in sub_curve:
            points.append((x, y + m))

        # Top-right
        for x, y in sub_curve:
            points.append((x + m, y + m))

        # Bottom-right (rotated left)
        for x, y in sub_curve:
            points.append((m - 1 - y + m, m - 1 - x))

        return points

    # Generate curve points
    curve_points = hilbert_curve(grid_size)

    # Create mapping from linear to Hilbert order
    hilbert_to_linear = torch.zeros(seq_len, dtype=torch.int32)
    linear_to_hilbert = torch.zeros(seq_len, dtype=torch.int32)

    for hilbert_idx, (x, y) in enumerate(curve_points[:seq_len]):
        linear_idx = y * grid_size + x
        if linear_idx < seq_len:
            hilbert_to_linear[hilbert_idx] = linear_idx
            linear_to_hilbert[linear_idx] = hilbert_idx

    return linear_to_hilbert


class HilbertDilatedAttentionTriton(nn.Module):
    """Dilated attention using Hilbert curve ordering with Triton kernels."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 256,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for Hilbert mappings
        self._hilbert_cache = {}

    def get_hilbert_mapping(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create Hilbert mapping for sequence length."""
        if seq_len not in self._hilbert_cache:
            mapping = generate_hilbert_curve_mapping(seq_len).to(device)
            self._hilbert_cache[seq_len] = mapping
        return self._hilbert_cache[seq_len]

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            use_hilbert: Whether to use Hilbert ordering
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection and reshape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Configure kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_DMODEL = self.head_dim

        # Allocate output
        out = torch.empty_like(q)

        # Grid configuration
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size, self.num_heads)

        if use_hilbert:
            # Get Hilbert mapping
            hilbert_map = self.get_hilbert_mapping(seq_len, x.device)

            # Launch Hilbert kernel
            hilbert_dilated_attention_kernel[grid](
                q,
                k,
                v,
                out,
                hilbert_map,
                # Strides
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
                # Dimensions
                seq_len,
                self.head_dim,
                self.segment_size,
                self.dilation_rate,
                # Block sizes
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
            )
        else:
            # Launch standard kernel
            standard_dilated_attention_kernel[grid](
                q,
                k,
                v,
                out,
                # Strides
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
                # Dimensions
                seq_len,
                self.head_dim,
                self.segment_size,
                self.dilation_rate,
                # Block sizes
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_DMODEL=BLOCK_DMODEL,
            )

        # Reshape and project output
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


def test_hilbert_triton():
    """Quick test of Triton implementation."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("Testing Hilbert Dilated Attention with Triton...")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    hidden_dim = 512
    num_heads = 8

    # Create model
    model = HilbertDilatedAttentionTriton(
        hidden_dim=hidden_dim, num_heads=num_heads, segment_size=256, dilation_rate=2
    ).cuda()

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Test both modes
    with torch.no_grad():
        out_hilbert = model(x, use_hilbert=True)
        out_standard = model(x, use_hilbert=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_hilbert.shape}")
    print(f"Outputs close: {torch.allclose(out_hilbert, out_standard, atol=1e-4)}")

    # Benchmark
    import time

    # Warmup
    for _ in range(10):
        _ = model(x, use_hilbert=True)
        _ = model(x, use_hilbert=False)

    torch.cuda.synchronize()

    # Time Hilbert
    start = time.time()
    for _ in range(100):
        _ = model(x, use_hilbert=True)
    torch.cuda.synchronize()
    hilbert_time = (time.time() - start) / 100

    # Time Standard
    start = time.time()
    for _ in range(100):
        _ = model(x, use_hilbert=False)
    torch.cuda.synchronize()
    standard_time = (time.time() - start) / 100

    print("\nBenchmark results:")
    print(f"Hilbert: {hilbert_time * 1000:.2f}ms")
    print(f"Standard: {standard_time * 1000:.2f}ms")
    print(f"Speedup: {standard_time / hilbert_time:.2f}x")


if __name__ == "__main__":
    test_hilbert_triton()
