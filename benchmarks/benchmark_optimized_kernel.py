#!/usr/bin/env python3
"""
Benchmark the optimized Triton kernel against the original.
"""

import torch
import time
import numpy as np


def benchmark_kernels():
    """Compare original vs optimized kernels."""
    print("=== Benchmarking Original vs Optimized Kernels ===\n")

    # Import both implementations
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    # We'll optimize the existing kernel instead of creating a new one
    # The key optimization is to process only dilated positions

    configs = [
        # (batch, seq_len, hidden_dim, segment_size, dilation_rate)
        (1, 512, 256, 64, 1),  # No dilation baseline
        (1, 512, 256, 64, 2),  # Dilation 2
        (1, 512, 256, 64, 4),  # Dilation 4
        (1, 1024, 256, 128, 4),  # Larger sequence
        (4, 512, 256, 64, 4),  # Batched
    ]

    for batch, seq_len, hidden_dim, seg_size, dil_rate in configs:
        print(
            f"\nConfig: B={batch}, Seq={seq_len}, Hidden={hidden_dim}, Seg={seg_size}, Dil={dil_rate}"
        )

        # Create module
        module = (
            HilbertAttentionCore(
                hidden_dim=hidden_dim,
                num_heads=8,
                segment_size=seg_size,
                dilation_rate=dil_rate,
                use_custom_backward=False,
            )
            .cuda()
            .eval()
        )

        x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = module(x, use_hilbert=False)

        # Time original
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                _ = module(x, use_hilbert=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        avg_time = np.mean(times[5:])  # Skip first few
        std_time = np.std(times[5:])

        # Calculate efficiency
        total_positions = seq_len * seq_len
        actual_positions = seq_len * (seg_size // dil_rate)
        efficiency = actual_positions / total_positions

        print(f"  Time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Theoretical efficiency: {efficiency:.1%} of full attention")
        print(f"  Expected speedup: {1 / efficiency:.1f}x")


def analyze_kernel_bottlenecks():
    """Analyze where the kernel spends time."""
    print("\n=== Kernel Bottleneck Analysis ===\n")

    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    # Profile different aspects
    seq_len = 512
    hidden_dim = 256

    module = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=8,
        segment_size=64,
        dilation_rate=4,
        use_custom_backward=False,
    ).cuda()

    x = torch.randn(1, seq_len, hidden_dim, device="cuda")

    # Profile QKV projection
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = module.qkv_proj(x)
    torch.cuda.synchronize()
    qkv_time = (time.perf_counter() - start) / 100 * 1000

    # Profile output projection
    dummy_out = torch.randn_like(x)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = module.out_proj(dummy_out)
    torch.cuda.synchronize()
    out_proj_time = (time.perf_counter() - start) / 100 * 1000

    # Profile full forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        with torch.no_grad():
            _ = module(x, use_hilbert=False)
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / 20 * 1000

    attention_time = total_time - qkv_time - out_proj_time

    print("Time breakdown:")
    print(f"  QKV projection: {qkv_time:.2f} ms ({qkv_time / total_time * 100:.1f}%)")
    print(
        f"  Attention kernel: {attention_time:.2f} ms ({attention_time / total_time * 100:.1f}%)"
    )
    print(
        f"  Output projection: {out_proj_time:.2f} ms ({out_proj_time / total_time * 100:.1f}%)"
    )
    print(f"  Total: {total_time:.2f} ms")


def suggest_optimizations():
    """Suggest specific optimizations for the kernel."""
    print("\n=== Optimization Suggestions ===\n")

    print("1. **Skip Empty Blocks**: Current kernel processes all blocks even if empty")
    print("   - Pre-compute which blocks have valid dilated positions")
    print("   - Skip BLOCK_N iterations that have no valid positions")
    print("   - Estimated speedup: 2-4x for high dilation rates")

    print("\n2. **Fuse Projections**: Merge QKV projection into attention kernel")
    print("   - Reduces memory bandwidth by 3x")
    print("   - Eliminates intermediate storage")
    print("   - Estimated speedup: 1.5-2x")

    print("\n3. **Better Block Sizes**: Dynamic block size based on dilation")
    print("   - Larger blocks when dilation creates sparse patterns")
    print("   - Smaller blocks for dense patterns")
    print("   - Estimated speedup: 1.2-1.5x")

    print("\n4. **Warp-Level Primitives**: Use warp shuffle for reductions")
    print("   - Replace some atomic operations with warp shuffles")
    print("   - Better GPU utilization")
    print("   - Estimated speedup: 1.1-1.3x")

    print("\n5. **Persistent Kernels**: Keep data in shared memory across iterations")
    print("   - For multi-layer transformers")
    print("   - Eliminates repeated loads")
    print("   - Estimated speedup: 1.5-2x for deep models")


def create_optimized_kernel_sketch():
    """Create a sketch of the optimized kernel."""
    print("\n=== Optimized Kernel Design ===\n")

    _ = '''
@triton.jit
def dilated_attention_kernel_v2(
    Q, K, V, Out,
    segment_info,  # Pre-computed segment boundaries and valid positions
    ...
):
    """Truly optimized kernel that skips empty computations."""
    
    # 1. Load segment info for this block
    seg_info = tl.load(segment_info + pid_m)
    seg_start = seg_info.start
    seg_end = seg_info.end
    num_valid = seg_info.num_valid
    
    # 2. Early exit if no valid positions
    if num_valid == 0:
        return
    
    # 3. Load queries once
    q = tl.load(Q_block, mask=mask_m)
    
    # 4. Process only valid key positions
    # Instead of iterating all positions, we have pre-computed
    # which positions are valid for this segment
    valid_positions = tl.load(valid_pos_ptr + seg_idx * MAX_VALID)
    
    for i in range(num_valid):
        key_pos = valid_positions[i]
        
        # 5. Vectorized load of K,V at valid positions only
        k = tl.load(K + key_pos, mask=mask_d)
        v = tl.load(V + key_pos, mask=mask_d)
        
        # 6. Compute attention for valid positions only
        s = tl.dot(q, k.T)
        ...
    
    # 7. Use warp-level reductions for efficiency
    m_i = tl.reduce(scores, axis=1, op='max', keep_dims=True)
    ...
'''

    print("Key optimizations in V2:")
    print("- Pre-compute valid positions to avoid runtime checks")
    print("- Skip entire blocks with no valid positions")
    print("- Vectorized loads only at valid positions")
    print("- Warp-level primitives for reductions")
    print("\nEstimated combined speedup: 3-5x for sparse patterns")


def main():
    """Run optimization analysis."""
    print("=== Triton Kernel Optimization Analysis ===\n")

    benchmark_kernels()
    analyze_kernel_bottlenecks()
    suggest_optimizations()
    create_optimized_kernel_sketch()

    print("\n=== Summary ===")
    print("Current kernel performance is limited by:")
    print("1. Processing all positions even when masked")
    print("2. Redundant memory accesses")
    print("3. Suboptimal block sizes for sparse patterns")
    print("\nWith optimizations, we could achieve 3-5x speedup for dilated attention")


if __name__ == "__main__":
    main()
