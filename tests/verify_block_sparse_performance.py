#!/usr/bin/env python3
"""
Verify performance improvements of Block Sparse Ring Dilated Attention.
"""

import time

import numpy as np
import torch


def measure_performance(
    attention_module, q, k, v, num_warmup=3, num_runs=10
) -> tuple[float, float]:
    """Measure forward pass performance."""
    device = q.device

    # Warmup
    for _ in range(num_warmup):
        _ = attention_module(q, k, v, is_causal=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Time runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = attention_module(q, k, v, is_causal=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)
    return mean_time, std_time


def compare_implementations():
    """Compare Block Sparse vs Regular Ring Attention."""
    print("=" * 60)
    print("Block Sparse vs Regular Ring Attention Performance")
    print("=" * 60)

    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention

    # Test configurations
    configs = [
        {"seq_len": 1024, "batch": 2, "heads": 8, "dim": 64},
        {"seq_len": 2048, "batch": 2, "heads": 8, "dim": 64},
        {"seq_len": 4096, "batch": 1, "heads": 8, "dim": 64},
    ]

    sparsity_ratios = [0.1, 0.25, 0.5]  # 90%, 75%, 50% sparse

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    for config in configs:
        print(f"\nSequence Length: {config['seq_len']}")
        print("-" * 50)

        # Create input tensors
        q = torch.randn(
            config["batch"],
            config["seq_len"],
            config["heads"],
            config["dim"],
            device=device,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Regular Ring Attention
        regular_attention = RingDilatedAttention(
            segment_lengths=[config["seq_len"] // 2],
            dilation_rates=[1],
            dropout=0.0,
            device=device,
        )

        regular_time, regular_std = measure_performance(regular_attention, q, k, v)
        print(f"Regular Ring Attention: {regular_time * 1000:.2f} ± {regular_std * 1000:.2f} ms")

        # Block Sparse variants
        for sparsity in sparsity_ratios:
            sparse_config = SparsePatternConfig(
                pattern_type="dilated_sparse", sparsity_ratio=sparsity, block_size=32
            )

            sparse_attention = BlockSparseRingDilatedAttention(
                segment_lengths=[config["seq_len"] // 2],
                dilation_rates=[1],
                sparse_config=sparse_config,
                dropout=0.0,
                device=device,
            )

            sparse_time, sparse_std = measure_performance(sparse_attention, q, k, v)
            speedup = regular_time / sparse_time
            theoretical_speedup = 1.0 / sparsity
            efficiency = (speedup / theoretical_speedup) * 100

            print(
                f"Block Sparse (sparsity={sparsity:.0%}): "
                f"{sparse_time * 1000:.2f} ± {sparse_std * 1000:.2f} ms "
                f"| Speedup: {speedup:.2f}x "
                f"| Theoretical: {theoretical_speedup:.1f}x "
                f"| Efficiency: {efficiency:.0f}%"
            )


def test_pattern_efficiency():
    """Test efficiency of different sparse patterns."""
    print("\n" + "=" * 60)
    print("Sparse Pattern Efficiency Comparison")
    print("=" * 60)

    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )

    patterns = ["local_window", "dilated_sparse", "global_local"]
    seq_len = 2048
    batch_size = 2
    num_heads = 8
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    print(f"\nSequence Length: {seq_len}, Sparsity: 75%")
    print("-" * 50)

    for pattern in patterns:
        sparse_config = SparsePatternConfig(
            pattern_type=pattern,
            sparsity_ratio=0.25,
            block_size=32,  # 75% sparse
        )

        attention = BlockSparseRingDilatedAttention(
            segment_lengths=[seq_len // 2],
            dilation_rates=[1],
            sparse_config=sparse_config,
            dropout=0.0,
            device=device,
        )

        mean_time, std_time = measure_performance(attention, q, k, v)
        print(f"{pattern:.<30} {mean_time * 1000:.2f} ± {std_time * 1000:.2f} ms")


def test_adaptive_performance():
    """Test adaptive sparsity performance."""
    print("\n" + "=" * 60)
    print("Adaptive Sparsity Performance")
    print("=" * 60)

    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )

    seq_len = 1024
    batch_size = 2
    num_heads = 8
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Fixed sparsity
    fixed_config = SparsePatternConfig(
        pattern_type="dilated_sparse", sparsity_ratio=0.25, block_size=32
    )

    fixed_attention = BlockSparseRingDilatedAttention(
        segment_lengths=[seq_len // 2],
        dilation_rates=[1],
        sparse_config=fixed_config,
        use_adaptive_sparsity=False,
        dropout=0.0,
        device=device,
    )

    # Adaptive sparsity
    adaptive_config = SparsePatternConfig(
        pattern_type="learned", sparsity_ratio=0.25, block_size=32
    )

    adaptive_attention = BlockSparseRingDilatedAttention(
        segment_lengths=[seq_len // 2],
        dilation_rates=[1],
        sparse_config=adaptive_config,
        use_adaptive_sparsity=True,
        dropout=0.0,
        device=device,
    )

    print(f"\nSequence Length: {seq_len}")
    print("-" * 50)

    fixed_time, fixed_std = measure_performance(fixed_attention, q, k, v)
    print(f"Fixed Sparsity (75% sparse):    {fixed_time * 1000:.2f} ± {fixed_std * 1000:.2f} ms")

    adaptive_time, adaptive_std = measure_performance(adaptive_attention, q, k, v)
    print(
        f"Adaptive Sparsity:              {adaptive_time * 1000:.2f} ± {adaptive_std * 1000:.2f} ms"
    )

    overhead = ((adaptive_time - fixed_time) / fixed_time) * 100
    print(f"Adaptive overhead:              {overhead:+.1f}%")


def main():
    """Run all performance tests."""
    try:
        compare_implementations()
        test_pattern_efficiency()
        test_adaptive_performance()

        print("\n" + "=" * 60)
        print("✓ Performance verification complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Performance verification failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
