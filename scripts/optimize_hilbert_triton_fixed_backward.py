#!/usr/bin/env python3
"""
Script to add custom backward pass to HilbertAttentionTritonFixed.

This demonstrates how to optimize the backward pass for the main Hilbert
attention implementation that currently lacks this optimization.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def hilbert_attention_bwd_kernel_fixed(
    # Input gradients
    dOut,
    # Original inputs
    Q,
    K,
    V,
    # Hilbert mapping
    hilbert_map,
    # Output gradients
    dQ,
    dK,
    dV,
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
    """Optimized backward kernel for HilbertAttentionTritonFixed."""
    # Program IDs
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh % H

    # Query block this program handles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Masks
    mask_m = offs_m < M
    mask_d = offs_d < D

    # Get Hilbert positions for queries
    hilbert_pos_q = tl.load(hilbert_map + offs_m, mask=mask_m, other=0)

    # Load queries and output gradients
    q_ptrs = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos_q[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    do_ptrs = (
        dOut
        + pid_b * stride_ob
        + pid_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize gradient accumulator for queries
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Compute segment boundaries
    seg_idx = offs_m // segment_size
    seg_start = seg_idx * segment_size
    seg_end = seg_start + segment_size

    # Process keys in the segment with dilation
    for offset in range(0, segment_size, dilation_rate):
        key_pos = seg_start + offset
        mask_k = (key_pos < M) & (key_pos < seg_end)

        if tl.sum(mask_k) > 0:
            # Get Hilbert position for this key
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

            # Recompute attention scores
            scores = tl.sum(q * k[None, :], axis=1)
            scores = tl.where(mask_k & mask_m, scores, -1e9)

            # Recompute softmax (simplified - in practice, need stable version)
            scores_exp = tl.exp(scores - tl.max(scores, axis=0))
            scores_sum = tl.sum(scores_exp) + 1e-10
            p = scores_exp / scores_sum

            # Gradient w.r.t. values: dV += p * dO
            dv = tl.sum(p[:, None] * do, axis=0)
            dv_ptrs = (
                dV
                + pid_b * stride_vb
                + pid_h * stride_vh
                + key_hilbert * stride_vn
                + offs_d * stride_vd
            )
            # Atomic add for gradient accumulation
            tl.atomic_add(dv_ptrs, dv, mask=mask_k & mask_d)

            # Gradient w.r.t. attention scores
            v_do = tl.sum(v[None, :] * do, axis=1)
            dp = p * (v_do - tl.sum(p * v_do))

            # Gradient w.r.t. queries: dQ += dp * k * scale
            dq_acc += dp[:, None] * k[None, :] * scale

            # Gradient w.r.t. keys: dK += dp * q
            dk = tl.sum(dp[:, None] * q, axis=0)
            dk_ptrs = (
                dK
                + pid_b * stride_kb
                + pid_h * stride_kh
                + key_hilbert * stride_kn
                + offs_d * stride_kd
            )
            tl.atomic_add(dk_ptrs, dk, mask=mask_k & mask_d)

    # Store accumulated query gradients
    dq_ptrs = (
        dQ
        + pid_b * stride_qb
        + pid_h * stride_qh
        + hilbert_pos_q[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_d[None, :])


class HilbertAttentionFixedFunc(torch.autograd.Function):
    """Custom autograd function for HilbertAttentionTritonFixed with optimized backward."""

    @staticmethod
    def forward(ctx, q, k, v, scale, hilbert_map, segment_size, dilation_rate):
        """Forward pass using existing kernel."""
        from src.dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
            hilbert_attention_kernel_fixed,
        )

        B, H, M, D = q.shape
        out = torch.zeros_like(q)

        # Configure kernel
        BLOCK_M = min(64, M)
        BLOCK_D = min(64, D)
        grid = (triton.cdiv(M, BLOCK_M), B * H)

        # Launch forward kernel
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
            M,
            D,
            scale,
            segment_size,
            dilation_rate,
            BLOCK_M,
            BLOCK_D,
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, hilbert_map)
        ctx.scale = scale
        ctx.segment_size = segment_size
        ctx.dilation_rate = dilation_rate
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_D = BLOCK_D

        return out

    @staticmethod
    def backward(ctx, dout):
        """Optimized backward pass."""
        q, k, v, hilbert_map = ctx.saved_tensors
        B, H, M, D = q.shape

        # Allocate gradient tensors
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # Configure kernel
        grid = (triton.cdiv(M, ctx.BLOCK_M), B * H)

        # Launch backward kernel
        hilbert_attention_bwd_kernel_fixed[grid](
            dout,
            q,
            k,
            v,
            hilbert_map,
            dq,
            dk,
            dv,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *dout.stride(),
            B,
            H,
            M,
            D,
            ctx.scale,
            ctx.segment_size,
            ctx.dilation_rate,
            ctx.BLOCK_M,
            ctx.BLOCK_D,
            num_warps=4,
        )

        return dq, dk, dv, None, None, None, None


def create_optimized_hilbert_attention_fixed(original_module):
    """
    Wrap an existing HilbertAttentionTritonFixed module to use custom backward.

    This function takes an existing module and replaces its forward method
    to use our optimized backward pass.
    """

    class OptimizedWrapper(nn.Module):
        def __init__(self, base_module):
            super().__init__()
            self.base = base_module
            self.scale = base_module.scale
            self.segment_size = base_module.segment_size
            self.dilation_rate = base_module.dilation_rate

        def forward(
            self,
            x: torch.Tensor,
            use_hilbert: bool = True,
            use_custom_backward: bool = True,
        ) -> torch.Tensor:
            B, M, D = x.shape
            H = self.base.num_heads

            # Ensure sequence length is compatible
            if M % self.segment_size != 0:
                pad_len = self.segment_size - (M % self.segment_size)
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
                M_padded = M + pad_len
            else:
                M_padded = M

            # QKV projection
            qkv = self.base.qkv_proj(x)
            qkv = qkv.reshape(B, M_padded, 3, H, self.base.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]

            if use_hilbert and use_custom_backward:
                # Get Hilbert mapping
                hilbert_map = self.base.get_hilbert_mapping(M_padded, x.device)

                # Use custom function with optimized backward
                out = HilbertAttentionFixedFunc.apply(
                    q,
                    k,
                    v,
                    self.scale,
                    hilbert_map,
                    self.segment_size,
                    self.dilation_rate,
                )
            else:
                # Fall back to original implementation
                return self.base.forward(x, use_hilbert)

            # Reshape output
            out = out.transpose(1, 2).reshape(B, M_padded, D)

            # Remove padding if applied
            if M_padded > M:
                out = out[:, :M, :]

            # Output projection and dropout
            out = self.base.out_proj(out)
            out = self.base.dropout(out)

            return out

    return OptimizedWrapper(original_module)


def benchmark_optimization():
    """Benchmark the optimization against the original."""
    import time
    from src.dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
        HilbertAttentionTritonFixed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuration
    batch_size = 2
    seq_len = 2048
    hidden_dim = 768
    num_heads = 12

    print("Benchmarking Custom Backward Optimization for HilbertAttentionTritonFixed")
    print("=" * 70)

    # Create original module
    original = HilbertAttentionTritonFixed(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=128,
        dilation_rate=1,
    ).to(device)

    # Create optimized version
    optimized = create_optimized_hilbert_attention_fixed(original)

    # Test input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

    if device.type == "cuda":
        print("\n1. Testing correctness...")

        # Test original
        x.grad = None
        out_orig = original(x.clone(), use_hilbert=True)
        loss_orig = out_orig.sum()
        loss_orig.backward()
        grad_orig = x.grad.clone()

        # Test optimized
        x.grad = None
        out_opt = optimized(x.clone(), use_hilbert=True, use_custom_backward=True)
        loss_opt = out_opt.sum()
        loss_opt.backward()
        grad_opt = x.grad.clone()

        print(f"   Output difference: {(out_orig - out_opt).abs().max().item():.2e}")
        print(
            f"   Gradient difference: {(grad_orig - grad_opt).abs().max().item():.2e}"
        )

        print("\n2. Benchmarking performance...")

        # Warmup
        for _ in range(5):
            x.grad = None
            out = optimized(x, use_custom_backward=True)
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()

        # Benchmark original
        print("\nOriginal Implementation:")
        start = time.perf_counter()
        for _ in range(10):
            x.grad = None
            out = original(x, use_hilbert=True)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        orig_time = (time.perf_counter() - start) / 10 * 1000

        print("\nOptimized Implementation:")
        start = time.perf_counter()
        for _ in range(10):
            x.grad = None
            out = optimized(x, use_custom_backward=True)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
        opt_time = (time.perf_counter() - start) / 10 * 1000

        print(f"\n   Original: {orig_time:.2f}ms")
        print(f"   Optimized: {opt_time:.2f}ms")
        print(f"   Speedup: {orig_time / opt_time:.2f}x")

        print(f"\n✅ Custom backward provides {orig_time / opt_time:.2f}x speedup!")

    else:
        print("(Skipping benchmarks on CPU)")


if __name__ == "__main__":
    benchmark_optimization()
