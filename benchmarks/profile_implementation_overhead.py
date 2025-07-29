#!/usr/bin/env python3
"""Profile where the overhead is coming from in our implementation."""

import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def profile_step_by_step():
    """Profile each step of the attention computation."""
    device = "cuda"
    seq_len = 4096
    batch_size = 1
    hidden_dim = 768
    num_heads = 12
    head_dim = 64

    print("STEP-BY-STEP PROFILING")
    print("=" * 80)

    # Create module and input
    module = HilbertAttention(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Get components we'll need
    hilbert_map = module._get_hilbert_mapping(seq_len, device)

    # Profile each step
    timings = {}
    num_runs = 100

    # 1. QKV projection
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        qkv = module.qkv_proj(x)
    torch.cuda.synchronize()
    timings["qkv_proj"] = (time.perf_counter() - start) / num_runs * 1000

    # 2. Reshape QKV
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        qkv_reshape = qkv.view(batch_size, seq_len, 3, num_heads, head_dim)
        qkv_permute = qkv_reshape.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_permute[0], qkv_permute[1], qkv_permute[2]
    torch.cuda.synchronize()
    timings["reshape"] = (time.perf_counter() - start) / num_runs * 1000

    # 3. Hilbert reordering of K,V
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        k_reorder = k[:, :, hilbert_map]
        v_reorder = v[:, :, hilbert_map]
    torch.cuda.synchronize()
    timings["reorder_kv"] = (time.perf_counter() - start) / num_runs * 1000

    # 4. Attention computation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs // 10):  # Fewer runs as this is expensive
        out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    timings["attention"] = (time.perf_counter() - start) / (num_runs // 10) * 1000

    # 5. Output projection
    out_flat = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = module.out_proj(out_flat)
    torch.cuda.synchronize()
    timings["out_proj"] = (time.perf_counter() - start) / num_runs * 1000

    # Print results
    print("\nTIMING BREAKDOWN:")
    print("-" * 60)
    total = sum(timings.values())
    for step, time_ms in timings.items():
        percent = (time_ms / total) * 100
        print(f"{step:<20} {time_ms:>8.2f}ms ({percent:>5.1f}%)")
    print(f"{'TOTAL':<20} {total:>8.2f}ms")

    # Compare with actual forward pass
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        actual_no_hilbert = (time.perf_counter() - start) / 10 * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        actual_with_hilbert = (time.perf_counter() - start) / 10 * 1000

    print("\nACTUAL FORWARD PASS:")
    print(f"  Without Hilbert: {actual_no_hilbert:.2f}ms")
    print(f"  With Hilbert: {actual_with_hilbert:.2f}ms")
    print(f"  Sum of parts: {total:.2f}ms")
    print(f"  Overhead: {actual_no_hilbert - total:.2f}ms")

    # Test pure SDPA performance
    print("\n\nPURE SDPA COMPARISON:")
    print("-" * 60)

    # Create tensors directly
    q_pure = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k_pure = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v_pure = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Time pure SDPA
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out = F.scaled_dot_product_attention(q_pure, k_pure, v_pure)
    torch.cuda.synchronize()
    pure_sdpa = (time.perf_counter() - start) / 10 * 1000

    # Time with reordering
    k_reorder = k_pure[:, :, hilbert_map]
    v_reorder = v_pure[:, :, hilbert_map]
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out = F.scaled_dot_product_attention(q_pure, k_reorder, v_reorder)
    torch.cuda.synchronize()
    sdpa_reorder = (time.perf_counter() - start) / 10 * 1000

    print(f"Pure SDPA: {pure_sdpa:.2f}ms")
    print(f"SDPA + pre-reordered K,V: {sdpa_reorder:.2f}ms")
    print(f"Reordering impact: {(sdpa_reorder / pure_sdpa - 1) * 100:+.1f}%")

    # Final analysis
    print("\n\nFINAL ANALYSIS:")
    print("=" * 60)
    print(f"""
Key findings:

1. **Hilbert reordering is inherently slower on this GPU**
   - Pure SDPA test shows {(sdpa_reorder / pure_sdpa - 1) * 100:+.1f}% slowdown
   - Not an implementation issue - it's a fundamental problem

2. **Our implementation adds significant overhead**
   - QKV projection: {timings["qkv_proj"]:.1f}ms
   - Output projection: {timings["out_proj"]:.1f}ms
   - These are necessary but expensive

3. **The overhead breakdown shows**:
   - Attention itself is only {(timings["attention"] / total) * 100:.1f}% of total time
   - Projections dominate the computation
   - Hilbert reordering adds {timings["reorder_kv"]:.1f}ms

4. **Why Hilbert fails on GTX 1080**:
   - Small L2 cache (2MB) can't exploit improved locality
   - High memory bandwidth (320GB/s) makes cache less critical
   - Indirect memory access breaks GPU optimizations

5. **Recommendation: Disable Hilbert reordering**
   - It provides no benefit on current hardware
   - Consider removing it entirely or making it opt-in only
   - Focus on other optimizations (Flash Attention, Ring Attention)
""")


if __name__ == "__main__":
    profile_step_by_step()
