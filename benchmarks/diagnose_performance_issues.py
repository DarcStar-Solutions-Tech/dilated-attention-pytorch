#!/usr/bin/env python3
"""Diagnose the root causes of poor performance."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def profile_overhead():
    """Profile the overhead of Hilbert operations."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("PERFORMANCE ISSUE DIAGNOSIS")
    print("=" * 80)

    # Test sequence
    seq_len = 4096
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    x = torch.randn(1, seq_len, 768, device=device)

    print(f"\n1. OVERHEAD ANALYSIS (seq_len={seq_len})")
    print("-" * 60)

    # Time Hilbert mapping generation
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        mapping = module._get_hilbert_mapping(seq_len, device)
    torch.cuda.synchronize()
    mapping_time = (time.perf_counter() - start) / 100 * 1000
    print(f"Hilbert mapping generation: {mapping_time:.3f}ms")

    # Time tensor reordering
    test_tensor = torch.randn(1, 12, seq_len, 64, device=device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        _ = test_tensor[:, :, mapping]
    torch.cuda.synchronize()
    reorder_time = (time.perf_counter() - start) / 100 * 1000
    print(f"Tensor reordering (K or V): {reorder_time:.3f}ms")

    # Time full attention computation
    with torch.no_grad():
        # Warmup
        _ = module(x, use_hilbert=False)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=False)
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / 10 * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = module(x, use_hilbert=True)
        torch.cuda.synchronize()
        hilbert_time = (time.perf_counter() - start) / 10 * 1000

    print("\nFull attention computation:")
    print(f"  Without Hilbert: {baseline_time:.2f}ms")
    print(f"  With Hilbert: {hilbert_time:.2f}ms")
    print(
        f"  Overhead: {hilbert_time - baseline_time:.2f}ms ({(hilbert_time / baseline_time - 1) * 100:.1f}%)"
    )

    # Analyze memory access patterns
    print("\n2. MEMORY ACCESS PATTERN ANALYSIS")
    print("-" * 60)

    # Check how scattered the Hilbert access pattern is
    mapping_cpu = mapping.cpu().numpy()

    # Calculate average jump distance
    jumps = []
    for i in range(1, min(1000, len(mapping_cpu))):
        jump = abs(mapping_cpu[i] - mapping_cpu[i - 1])
        jumps.append(jump)

    avg_jump = sum(jumps) / len(jumps) if jumps else 0
    max_jump = max(jumps) if jumps else 0

    print(f"Average memory jump: {avg_jump:.1f} positions")
    print(f"Max memory jump: {max_jump} positions")
    print("Sequential access would be: 1 position")
    print(f"Random access would be: ~{seq_len / 2:.0f} positions")

    # Check cache line efficiency
    cache_line_size = 128  # bytes (32 float32 elements)
    elements_per_line = cache_line_size // 4

    cache_misses = sum(1 for j in jumps if j > elements_per_line)
    cache_miss_rate = cache_misses / len(jumps) * 100 if jumps else 0

    print("\nCache efficiency estimate:")
    print(f"  Cache line size: {elements_per_line} elements")
    print(f"  Estimated cache miss rate: {cache_miss_rate:.1f}%")

    # Analyze the actual problem
    print("\n3. ROOT CAUSE ANALYSIS")
    print("-" * 60)
    print("""
The performance issues stem from:

1. **Indirect Memory Access Overhead**:
   - Each K,V access requires loading the Hilbert index first
   - This doubles memory operations and breaks coalescing
   - GPU can't prefetch effectively with indirect access

2. **Cache Line Inefficiency**:
   - Hilbert curve creates large jumps in memory
   - Each access likely loads a new cache line
   - Poor spatial locality despite theoretical benefits

3. **Overhead Not Amortized**:
   - Mapping generation + 2x reordering adds fixed cost
   - Benefits only appear when cache reuse is significant
   - For attention, we only read each K,V once per query block

4. **Hardware Mismatch**:
   - GTX 1080 has small L2 cache (2MB)
   - Can't hold enough data to benefit from reordering
   - High bandwidth masks cache miss penalties
""")

    # Test if the issue is in our implementation
    print("\n4. IMPLEMENTATION COMPARISON")
    print("-" * 60)

    # Direct SDPA comparison
    q = torch.randn(1, 12, seq_len, 64, device=device)
    k = torch.randn(1, 12, seq_len, 64, device=device)
    v = torch.randn(1, 12, seq_len, 64, device=device)

    with torch.no_grad():
        # Standard SDPA
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - start) / 10 * 1000

        # SDPA with reordered K,V
        k_reorder = k[:, :, mapping]
        v_reorder = v[:, :, mapping]
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(
                q, k_reorder, v_reorder
            )
        torch.cuda.synchronize()
        sdpa_reorder_time = (time.perf_counter() - start) / 10 * 1000

    print(f"PyTorch SDPA baseline: {sdpa_time:.2f}ms")
    print(f"PyTorch SDPA + reordered K,V: {sdpa_reorder_time:.2f}ms")
    print(
        f"Reordering overhead in SDPA: {(sdpa_reorder_time / sdpa_time - 1) * 100:.1f}%"
    )

    print("\n5. RECOMMENDATIONS")
    print("-" * 60)
    print("""
To fix the performance issues:

1. **Remove Hilbert reordering for sequences < 16K**
   - The overhead exceeds any cache benefits
   - Simple sequential access is more GPU-friendly

2. **If keeping Hilbert, redesign the approach**:
   - Apply Hilbert only within attention tiles (like Flash Attention)
   - Fuse reordering into the kernel to avoid extra passes
   - Use hierarchical Hilbert curves matching GPU cache hierarchy

3. **Focus on algorithms that work with GPU architecture**:
   - Flash Attention's tiling approach
   - Block-sparse patterns with regular structure
   - Ring attention for very long sequences

4. **Current implementation should**:
   - Disable Hilbert by default
   - Only enable for sequences > 32K on GPUs with large caches
   - Warn users about performance implications
""")


if __name__ == "__main__":
    profile_overhead()
