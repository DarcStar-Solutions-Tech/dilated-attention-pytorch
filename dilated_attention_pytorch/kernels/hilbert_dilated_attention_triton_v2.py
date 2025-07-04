#!/usr/bin/env python3
"""
Fixed Hilbert curve ordered dilated attention kernel using Triton.
Addresses the compilation and runtime issues.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


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
    SEGMENT_SIZE,
    DILATION_RATE,
    # Scale factor (precomputed)
    scale,
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
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Initialize accumulator and normalization
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)

    # Get Hilbert positions for queries
    query_hilbert = tl.load(hilbert_map_ptr + offs_m, mask=m_mask, other=0)

    # Process segments
    num_segments = tl.cdiv(N_CTX, SEGMENT_SIZE)

    for seg_idx in range(num_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        seg_end = tl.minimum(seg_start + SEGMENT_SIZE, N_CTX)

        # Process dilated positions within segment
        for dil_idx in range(0, SEGMENT_SIZE, DILATION_RATE):
            key_pos = seg_start + dil_idx
            if key_pos >= seg_end:
                continue

            # Load key position and its Hilbert mapping
            key_hilbert = tl.load(
                hilbert_map_ptr + key_pos, mask=key_pos < N_CTX, other=0
            )

            # Compute K and V pointers
            k_ptr = K_ptr + (
                batch_idx * batch_stride_k
                + head_idx * head_stride_k
                + key_hilbert * seq_stride_k
                + offs_d
            )
            v_ptr = V_ptr + (
                batch_idx * batch_stride_v
                + head_idx * head_stride_v
                + key_hilbert * seq_stride_v
                + offs_d
            )

            # Load K and V
            k = tl.load(k_ptr, mask=key_pos < N_CTX, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=key_pos < N_CTX, other=0.0).to(tl.float32)

            # Compute attention scores for all queries against this key
            qk = tl.sum(q * k[None, :], axis=1) * scale

            # Update with stable softmax
            m_i_new = tl.maximum(m_i, qk)
            p = tl.exp(qk - m_i_new)
            alpha = tl.exp(m_i - m_i_new)

            # Update accumulator
            acc_scale = l_i * alpha
            acc = acc * acc_scale[:, None] + p[:, None] * v[None, :]

            # Update normalization
            l_i = l_i * alpha + p
            m_i = m_i_new

    # Normalize output
    acc = acc / l_i[:, None]

    # Write output using Hilbert mapping
    out_ptrs = Out_ptr + (
        batch_idx * batch_stride_out
        + head_idx * head_stride_out
        + query_hilbert[:, None] * seq_stride_out
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=m_mask[:, None])


@triton.jit
def standard_dilated_attention_kernel(
    # Pointers to matrices
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
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
    SEGMENT_SIZE,
    DILATION_RATE,
    scale,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """Standard dilated attention kernel for comparison."""
    start_m = tl.program_id(0)
    batch_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Load Q
    q_ptrs = Q_ptr + (
        batch_idx * batch_stride_q
        + head_idx * head_stride_q
        + offs_m[:, None] * seq_stride_q
        + offs_d[None, :]
    )
    m_mask = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Initialize
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)

    # Process segments
    num_segments = tl.cdiv(N_CTX, SEGMENT_SIZE)

    for seg_idx in range(num_segments):
        seg_start = seg_idx * SEGMENT_SIZE
        seg_end = tl.minimum(seg_start + SEGMENT_SIZE, N_CTX)

        # Check if queries are in this segment
        in_segment = (offs_m >= seg_start) & (offs_m < seg_end)

        # Process dilated positions
        for dil_idx in range(0, SEGMENT_SIZE, DILATION_RATE):
            key_pos = seg_start + dil_idx
            if key_pos >= seg_end:
                continue

            # Load K and V
            k_ptr = K_ptr + (
                batch_idx * batch_stride_k
                + head_idx * head_stride_k
                + key_pos * seq_stride_k
                + offs_d
            )
            v_ptr = V_ptr + (
                batch_idx * batch_stride_v
                + head_idx * head_stride_v
                + key_pos * seq_stride_v
                + offs_d
            )

            k = tl.load(k_ptr, mask=key_pos < N_CTX, other=0.0).to(tl.float32)
            v = tl.load(v_ptr, mask=key_pos < N_CTX, other=0.0).to(tl.float32)

            # Compute scores only for queries in the same segment
            qk = tl.sum(q * k[None, :], axis=1) * scale
            qk = tl.where(in_segment, qk, -float("inf"))

            # Update with stable softmax
            m_i_new = tl.maximum(m_i, qk)
            p = tl.exp(qk - m_i_new)
            alpha = tl.exp(m_i - m_i_new)

            acc_scale = l_i * alpha
            acc = acc * acc_scale[:, None] + p[:, None] * v[None, :]

            l_i = l_i * alpha + p
            m_i = m_i_new

    # Normalize and store
    acc = acc / (l_i[:, None] + 1e-6)
    out_ptrs = Out_ptr + (
        batch_idx * batch_stride_out
        + head_idx * head_stride_out
        + offs_m[:, None] * seq_stride_out
        + offs_d[None, :]
    )
    tl.store(out_ptrs, acc.to(out_ptrs.dtype.element_ty), mask=m_mask[:, None])


def generate_hilbert_curve_mapping(seq_len: int) -> torch.Tensor:
    """Generate Hilbert curve mapping for sequence length."""
    # Find next power of 2 for grid size
    grid_size = 2 ** int(math.ceil(math.log2(math.sqrt(seq_len))))

    def hilbert_curve(n):
        """Generate points along Hilbert curve."""
        if n == 1:
            return [(0, 0)]

        points = []
        m = n // 2
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

    # Create mapping
    linear_to_hilbert = torch.zeros(seq_len, dtype=torch.int32)

    for hilbert_idx, (x, y) in enumerate(curve_points[:seq_len]):
        linear_idx = y * grid_size + x
        if linear_idx < seq_len:
            linear_to_hilbert[linear_idx] = hilbert_idx

    return linear_to_hilbert


class HilbertDilatedAttentionTriton(nn.Module):
    """Fixed Hilbert dilated attention using Triton kernels."""

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
        self.scale = 1.0 / math.sqrt(self.head_dim)

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
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_DMODEL = self.head_dim

        # Ensure BLOCK_DMODEL is a valid block size
        if BLOCK_DMODEL > 256:
            BLOCK_DMODEL = 256
        elif BLOCK_DMODEL > 128:
            BLOCK_DMODEL = 128
        elif BLOCK_DMODEL > 64:
            BLOCK_DMODEL = 64
        elif BLOCK_DMODEL > 32:
            BLOCK_DMODEL = 32
        else:
            BLOCK_DMODEL = 32

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
                self.segment_size,
                self.dilation_rate,
                self.scale,  # Pass scale as a float
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
                self.segment_size,
                self.dilation_rate,
                self.scale,
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
    """Test the fixed Triton implementation."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("Testing Fixed Hilbert Dilated Attention with Triton...")

    # Test parameters
    batch_size = 2
    seq_len = 1024
    hidden_dim = 256
    num_heads = 8

    # Create model
    model = HilbertDilatedAttentionTriton(
        hidden_dim=hidden_dim, num_heads=num_heads, segment_size=256, dilation_rate=2
    ).cuda()

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

    # Test both modes
    print("\nTesting forward pass...")
    with torch.no_grad():
        try:
            out_hilbert = model(x, use_hilbert=True)
            print("✓ Hilbert forward pass successful")
        except Exception as e:
            print(f"✗ Hilbert forward pass failed: {e}")
            return

        try:
            out_standard = model(x, use_hilbert=False)
            print("✓ Standard forward pass successful")
        except Exception as e:
            print(f"✗ Standard forward pass failed: {e}")
            return

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {out_hilbert.shape}")
    print(f"Outputs close: {torch.allclose(out_hilbert, out_standard, atol=1e-3)}")

    # Benchmark
    import time

    print("\nBenchmarking...")

    # Warmup
    for _ in range(10):
        _ = model(x, use_hilbert=True)
        _ = model(x, use_hilbert=False)

    torch.cuda.synchronize()

    # Time Hilbert
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        _ = model(x, use_hilbert=True)
    torch.cuda.synchronize()
    hilbert_time = (time.time() - start) / iterations

    # Time Standard
    start = time.time()
    for _ in range(iterations):
        _ = model(x, use_hilbert=False)
    torch.cuda.synchronize()
    standard_time = (time.time() - start) / iterations

    print("\nBenchmark results:")
    print(f"Hilbert: {hilbert_time * 1000:.2f}ms")
    print(f"Standard: {standard_time * 1000:.2f}ms")
    print(f"Speedup: {standard_time / hilbert_time:.2f}x")

    # Test different configurations
    print("\nTesting different configurations...")
    configs = [
        (512, 128, 2),
        (1024, 256, 4),
        (2048, 512, 8),
    ]

    for seq_len, seg_size, dilation in configs:
        model = HilbertDilatedAttentionTriton(
            hidden_dim=256, num_heads=8, segment_size=seg_size, dilation_rate=dilation
        ).cuda()

        x = torch.randn(1, seq_len, 256).cuda()

        try:
            with torch.no_grad():
                _ = model(x, use_hilbert=True)
            print(
                f"✓ Config: seq_len={seq_len}, segment={seg_size}, dilation={dilation}"
            )
        except Exception as e:
            print(f"✗ Config failed: {e}")


if __name__ == "__main__":
    test_hilbert_triton()
