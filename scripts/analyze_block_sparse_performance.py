#!/usr/bin/env python3
"""
Analyze Block-Sparse Ring Dilated Attention performance bottlenecks.

This script profiles the implementation to identify specific optimization opportunities.
"""

import torch
import torch.profiler
from pathlib import Path
from datetime import datetime
import time

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def profile_operation(name, func, *args, **kwargs):
    """Profile a single operation."""
    # Warmup
    for _ in range(3):
        func(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Time the operation
    start = time.perf_counter()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return result, (end - start) * 1000  # ms


def analyze_block_sparse(seq_len=4096, num_heads=8, head_dim=64, sparsity_ratio=0.95):
    """Analyze Block-Sparse performance in detail."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print("Analyzing Block-Sparse Ring Dilated Attention")
    print(f"Device: {device}, Sequence Length: {seq_len}")
    print("=" * 80)

    # Create test inputs
    batch_size = 1
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Configuration - adjust segment lengths based on sequence length
    if seq_len <= 2048:
        segment_lengths = [512, 1024, 2048]
    elif seq_len <= 4096:
        segment_lengths = [1024, 2048, 4096]
    else:
        segment_lengths = [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=sparsity_ratio,
        block_size=64,
        local_window_size=256,
    )

    # Create model
    model = BlockSparseRingDilatedAttention(
        ring_size=1,
        device=device,
        dtype=dtype,
        sparse_config=sparse_config,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
    )

    # Profile different parts of the forward pass
    print("\n1. Analyzing Pattern Generation")
    print("-" * 40)

    # Time pattern generation by running a dummy forward pass
    dummy_q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    dummy_k = torch.randn_like(dummy_q)
    dummy_v = torch.randn_like(dummy_q)

    # Clear pattern cache first
    model.pattern_cache.clear()

    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_q, dummy_k, dummy_v, is_causal=False)
    pattern_time = (time.perf_counter() - start) * 1000
    print(f"First forward pass (includes pattern generation): {pattern_time:.2f} ms")

    # Check if pattern is cached
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_q, dummy_k, dummy_v, is_causal=False)
    pattern_time2 = (time.perf_counter() - start) * 1000
    print(f"Second forward pass (cached pattern): {pattern_time2:.2f} ms")

    cache_benefit = pattern_time - pattern_time2
    print(f"Cache benefit: {cache_benefit:.2f} ms saved")

    # Check pattern cache content
    print(f"Pattern cache size: {len(model.pattern_cache)} entries")

    print("\n2. Profiling Forward Pass Components")
    print("-" * 40)

    # Use PyTorch profiler for detailed analysis
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.profiler.record_function("block_sparse_forward"):
            _ = model(query, key, value, is_causal=False)

    # Print profiler results
    print("\nTop 10 most time-consuming operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Save detailed trace
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    trace_path = Path("analysis") / f"block_sparse_trace_{timestamp}.json"
    trace_path.parent.mkdir(exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    print(f"\nDetailed trace saved to: {trace_path}")

    print("\n3. Memory Analysis")
    print("-" * 40)

    if device.type == "cuda":
        # Clear cache and measure memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure memory during forward pass
        start_mem = torch.cuda.memory_allocated() / 1024**2
        with torch.no_grad():
            _ = model(query, key, value, is_causal=False)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        print(f"Memory before forward: {start_mem:.2f} MB")
        print(f"Peak memory during forward: {peak_mem:.2f} MB")
        print(f"Memory overhead: {peak_mem - start_mem:.2f} MB")

    print("\n4. Comparison with Baseline")
    print("-" * 40)

    # Compare with dense attention
    baseline = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
    )

    # Time both implementations
    _, sparse_time = profile_operation(
        "BlockSparse", model, query, key, value, is_causal=False
    )
    _, dense_time = profile_operation(
        "Baseline", baseline, query, key, value, is_causal=False
    )

    print(f"Block-Sparse time: {sparse_time:.2f} ms")
    print(f"Baseline time: {dense_time:.2f} ms")
    print(f"Slowdown factor: {sparse_time / dense_time:.2f}x")

    print("\n5. Optimization Opportunities")
    print("-" * 40)

    # Check for specific bottlenecks

    # 1. Pattern recomputation
    if pattern_time > 1.0:
        print("⚠️  Pattern generation is slow - implement caching")

    # 2. Memory allocation
    if peak_mem - start_mem > 100:
        print("⚠️  High memory overhead - check for unnecessary allocations")

    # 3. Sparse operations
    if sparse_time / dense_time > 2:
        print("⚠️  Sparse operations are inefficient - consider using torch.sparse")

    print("\nRecommendations:")
    print("1. Cache sparse patterns between forward passes")
    print("2. Use torch.sparse tensors for attention computation")
    print("3. Fuse pattern application with attention computation")
    print("4. Pre-allocate buffers for intermediate tensors")

    return {
        "pattern_time": pattern_time,
        "sparse_time": sparse_time,
        "dense_time": dense_time,
        "slowdown": sparse_time / dense_time,
        "memory_overhead": peak_mem - start_mem if device.type == "cuda" else 0,
    }


def main():
    """Run comprehensive analysis."""
    results = {}

    # Test different configurations
    for seq_len in [2048, 4096, 8192]:
        for sparsity in [0.9, 0.95, 0.98]:
            print(f"\n{'=' * 80}")
            print(f"Configuration: seq_len={seq_len}, sparsity={sparsity}")
            print(f"{'=' * 80}")

            key = f"seq{seq_len}_sparse{sparsity}"
            results[key] = analyze_block_sparse(
                seq_len=seq_len,
                sparsity_ratio=sparsity,
            )

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_path = Path("analysis") / f"block_sparse_analysis_{timestamp}.md"

    with open(output_path, "w") as f:
        f.write("# Block-Sparse Performance Analysis\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Results Summary\n\n")
        f.write(
            "| Configuration | Pattern Gen (ms) | Sparse Time (ms) | Dense Time (ms) | Slowdown |\n"
        )
        f.write(
            "|--------------|------------------|------------------|-----------------|----------|\n"
        )

        for key, res in results.items():
            f.write(
                f"| {key} | {res['pattern_time']:.2f} | "
                f"{res['sparse_time']:.2f} | {res['dense_time']:.2f} | "
                f"{res['slowdown']:.2f}x |\n"
            )

    print(f"\n\nAnalysis complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
