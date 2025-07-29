"""
Optimized Triton kernels for sparse attention patterns.
"""

import triton
import triton.language as tl


@triton.jit
def compute_active_positions(
    seg_start: tl.int32,
    seg_size: tl.int32,
    dilation_rate: tl.int32,
    block_idx: tl.int32,
    BLOCK_N: tl.constexpr,
) -> tl.tensor:
    """Compute active positions for a sparse attention block."""
    # Calculate which active positions this block handles
    # Each block handles BLOCK_N / dilation_rate active positions
    active_per_block = BLOCK_N // dilation_rate
    start_active_idx = block_idx * active_per_block

    # Generate indices for active positions
    active_indices = tl.arange(0, active_per_block)
    actual_positions = seg_start + (start_active_idx + active_indices) * dilation_rate

    return actual_positions


@triton.jit
def hilbert_attention_kernel_sparse(
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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Optimized kernel for sparse attention that only processes active positions."""
    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Compute query block boundaries
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

    # Initialize output
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Determine segment for queries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    _ = tl.minimum(seg_start + segment_size, M)

    # For sparse attention, we only need to process active positions
    # Calculate how many active positions are in this segment
    active_per_segment = segment_size // dilation_rate

    # Process only the active positions in blocks
    active_blocks = tl.cdiv(active_per_segment, BLOCK_N // dilation_rate)

    for block_idx in range(active_blocks):
        # Calculate actual positions for this block of active elements
        active_start = block_idx * (BLOCK_N // dilation_rate)

        # Generate positions - only the ones we actually need
        local_indices = tl.arange(0, BLOCK_N // dilation_rate)
        active_positions = active_start + local_indices

        # Convert to actual sequence positions
        actual_n = seg_start + active_positions * dilation_rate

        # Mask for valid positions
        mask_n = (active_positions < active_per_segment) & (actual_n < M)

        # Expand mask for full BLOCK_N size (with zeros for non-active)
        offs_n = actual_n
        expanded_mask = mask_n

        # Load Hilbert indices only for active positions
        h_idx = tl.load(hilbert_map + offs_n, mask=expanded_mask, other=0)

        # Load keys and values
        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + h_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + h_idx[None, :] * stride_vn
            + offs_d[:, None] * stride_vd
        )

        # Note: we're loading fewer elements (BLOCK_N // dilation_rate)
        k = tl.load(k_ptrs, mask=expanded_mask[None, :] & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=expanded_mask[None, :] & mask_d[:, None], other=0.0)

        # Compute attention scores - smaller computation
        s = tl.dot(q, k)
        s = tl.where(expanded_mask[None, :], s, -1e9)

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

        # Update for next iteration
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
def hilbert_attention_kernel_sparse_v2(
    # Pointers
    Q,
    K,
    V,
    Out,
    hilbert_map,
    # Strides - same as before
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
    """Alternative approach: Process with stride."""
    # Get program IDs
    pid = tl.program_id(0)
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_blocks_m
    pid_bh = pid // num_blocks_m
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Load queries (same as before)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D

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
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9

    # Get segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = tl.minimum(seg_start + segment_size, M)

    # Key optimization: iterate with step size = dilation_rate * BLOCK_N
    # This naturally skips over positions we don't need
    step_size = dilation_rate * BLOCK_N if dilation_rate > 1 else BLOCK_N

    # Start at first active position in segment
    for base_n in range(0, segment_size, step_size):
        start_n = seg_start + base_n

        # For sparse patterns, generate only active positions
        if dilation_rate > 1:
            # Generate sparse indices directly
            sparse_offsets = tl.arange(0, BLOCK_N) * dilation_rate
            offs_n = start_n + sparse_offsets
        else:
            # Dense pattern
            offs_n = start_n + tl.arange(0, BLOCK_N)

        # Check bounds
        mask_n = (offs_n >= seg_start) & (offs_n < seg_end) & (offs_n < M)

        # Load and process (same as original)
        h_idx = tl.load(hilbert_map + offs_n, mask=mask_n, other=0)

        k_ptrs = (
            K
            + pid_b * stride_kb
            + pid_h * stride_kh
            + h_idx[None, :] * stride_kn
            + offs_d[:, None] * stride_kd
        )
        v_ptrs = (
            V
            + pid_b * stride_vb
            + pid_h * stride_vh
            + h_idx[None, :] * stride_vn
            + offs_d[:, None] * stride_vd
        )

        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        # Attention computation
        s = tl.dot(q, k)
        s = tl.where(mask_n[None, :], s, -1e9)

        # Softmax and accumulate
        m_ij = tl.max(s, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        p = tl.exp(s - m_i_new[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + l_ij

        acc = acc * alpha[:, None]
        v_t = tl.trans(v)
        acc += tl.dot(p, v_t)

        l_i = l_i_new
        m_i = m_i_new

    # Normalize and store
    acc = acc / tl.maximum(l_i[:, None], 1e-10)

    out_ptrs = (
        Out
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])
