#!/usr/bin/env python3
"""
Analyze performance patterns in the final Hilbert kernels to understand
when each implementation performs best.
"""

import torch
import time
import sys
import matplotlib.pyplot as plt

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimized,
    UnifiedHilbertAttentionOptimizedEnhanced,
)


def benchmark_with_config(impl_class, config, x, warmup=3, runs=10):
    """Benchmark an implementation with specific config."""
    try:
        module = impl_class(**config).cuda()
        module.eval()

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = module(x)
            torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(runs):
            with torch.no_grad():
                _ = module(x)
            torch.cuda.synchronize()

        end = time.perf_counter()
        return (end - start) / runs * 1000  # ms

    except Exception as e:
        print(f"Error: {e}")
        return None


def analyze_performance_patterns():
    """Analyze when each implementation performs best."""

    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    implementations = {
        "Unified": UnifiedHilbertAttention,
        "Optimized": UnifiedHilbertAttentionOptimized,
        "Enhanced": UnifiedHilbertAttentionOptimizedEnhanced,
    }

    print("=== Performance Pattern Analysis ===")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Test 1: Small sequences (where PyTorch fallback might be used)
    print("\n1. Small Sequence Performance (PyTorch fallback):")
    for seq_len in [128, 256, 512]:
        print(f"\n  Sequence length {seq_len}:")
        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        for name, impl in implementations.items():
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": segment_size,
                "dilation_rate": 1,
                "hilbert_threshold": 1024,  # This affects when Triton is used
            }
            if name == "Enhanced":
                config["enable_8k_optimization"] = True

            time_ms = benchmark_with_config(impl, config, x)
            if time_ms:
                print(f"    {name}: {time_ms:.2f}ms")

    # Test 2: Hilbert threshold impact
    print("\n2. Hilbert Threshold Impact (seq_len=2048):")
    x = torch.randn(batch_size, 2048, hidden_dim).cuda()

    for threshold in [512, 1024, 2048, 4096]:
        print(f"\n  Threshold={threshold}:")

        for name, impl in implementations.items():
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": segment_size,
                "dilation_rate": 1,
                "hilbert_threshold": threshold,
            }
            if name == "Enhanced":
                config["enable_8k_optimization"] = True

            time_ms = benchmark_with_config(impl, config, x)
            if time_ms:
                print(f"    {name}: {time_ms:.2f}ms")

    # Test 3: Segment size impact
    print("\n3. Segment Size Impact (seq_len=4096):")
    x = torch.randn(batch_size, 4096, hidden_dim).cuda()

    for seg_size in [64, 128, 256, 512]:
        print(f"\n  Segment size={seg_size}:")

        for name, impl in implementations.items():
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": seg_size,
                "dilation_rate": 1,
                "hilbert_threshold": 1024,
            }
            if name == "Enhanced":
                config["enable_8k_optimization"] = True

            time_ms = benchmark_with_config(impl, config, x)
            if time_ms:
                print(f"    {name}: {time_ms:.2f}ms")

    # Test 4: Different head dimensions
    print("\n4. Head Dimension Impact (seq_len=2048):")

    for num_heads in [4, 8, 16]:
        head_dim = hidden_dim // num_heads
        print(f"\n  Heads={num_heads}, Head dim={head_dim}:")
        x = torch.randn(batch_size, 2048, hidden_dim).cuda()

        for name, impl in implementations.items():
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "segment_size": segment_size,
                "dilation_rate": 1,
                "hilbert_threshold": 1024,
            }
            if name == "Enhanced":
                config["enable_8k_optimization"] = True

            time_ms = benchmark_with_config(impl, config, x)
            if time_ms:
                print(f"    {name}: {time_ms:.2f}ms")

    # Test 5: Sparse patterns with different configurations
    print("\n5. Sparse Pattern Performance:")

    for seq_len in [2048, 4096, 8192]:
        for dilation in [2, 4]:
            print(f"\n  Seq={seq_len}, Dilation={dilation}:")
            x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

            for name, impl in implementations.items():
                config = {
                    "hidden_dim": hidden_dim,
                    "num_heads": 8,
                    "segment_size": segment_size,
                    "dilation_rate": dilation,
                    "hilbert_threshold": 1024,
                }
                if name == "Enhanced":
                    config["enable_8k_optimization"] = True

                time_ms = benchmark_with_config(impl, config, x)
                if time_ms:
                    print(f"    {name}: {time_ms:.2f}ms")

    # Test 6: Batch size scaling
    print("\n6. Batch Size Scaling (seq_len=2048):")

    for batch in [1, 2, 4, 8]:
        print(f"\n  Batch size={batch}:")
        x = torch.randn(batch, 2048, hidden_dim).cuda()

        for name, impl in implementations.items():
            config = {
                "hidden_dim": hidden_dim,
                "num_heads": 8,
                "segment_size": segment_size,
                "dilation_rate": 1,
                "hilbert_threshold": 1024,
            }
            if name == "Enhanced":
                config["enable_8k_optimization"] = True

            time_ms = benchmark_with_config(impl, config, x)
            if time_ms:
                print(f"    {name}: {time_ms:.2f}ms")

    # Test 7: Check actual kernel usage
    print("\n7. Kernel Usage Analysis:")

    # Force PyTorch path
    print("\n  Forcing PyTorch path (small sequence):")
    x = torch.randn(1, 256, hidden_dim).cuda()
    for name, impl in implementations.items():
        config = {
            "hidden_dim": hidden_dim,
            "num_heads": 8,
            "segment_size": segment_size,
            "dilation_rate": 1,
            "hilbert_threshold": 1024,
        }
        if name == "Enhanced":
            config["enable_8k_optimization"] = True

        module = impl(**config).cuda()
        module.eval()

        # Check which path is taken
        with torch.no_grad():
            # Small sequence should use PyTorch
            out = module(x)
            print(f"    {name}: Output shape {out.shape}")

    # Force Triton path
    print("\n  Forcing Triton path (large sequence):")
    x = torch.randn(1, 4096, hidden_dim).cuda()
    for name, impl in implementations.items():
        config = {
            "hidden_dim": hidden_dim,
            "num_heads": 8,
            "segment_size": segment_size,
            "dilation_rate": 1,
            "hilbert_threshold": 512,  # Lower threshold
        }
        if name == "Enhanced":
            config["enable_8k_optimization"] = True

        module = impl(**config).cuda()
        module.eval()

        with torch.no_grad():
            out = module(x)
            print(f"    {name}: Output shape {out.shape}")


def plot_performance_breakdown():
    """Create detailed performance breakdown plots."""

    # Run focused benchmarks
    results = {
        "seq_lens": [512, 1024, 2048, 4096, 8192],
        "unified": [],
        "optimized": [],
        "enhanced": [],
    }

    batch_size = 2
    hidden_dim = 512
    num_heads = 8
    segment_size = 128

    for seq_len in results["seq_lens"]:
        x = torch.randn(batch_size, seq_len, hidden_dim).cuda()

        # Unified
        config = {
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "segment_size": segment_size,
            "dilation_rate": 1,
            "hilbert_threshold": 1024,
        }
        time_ms = benchmark_with_config(UnifiedHilbertAttention, config, x)
        results["unified"].append(time_ms)

        # Optimized
        time_ms = benchmark_with_config(UnifiedHilbertAttentionOptimized, config, x)
        results["optimized"].append(time_ms)

        # Enhanced
        config["enable_8k_optimization"] = True
        config["enable_multi_row"] = True
        time_ms = benchmark_with_config(
            UnifiedHilbertAttentionOptimizedEnhanced, config, x
        )
        results["enhanced"].append(time_ms)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        results["seq_lens"],
        results["unified"],
        "o-",
        label="Unified",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        results["seq_lens"],
        results["optimized"],
        "s-",
        label="Optimized",
        linewidth=2,
        markersize=8,
    )
    plt.plot(
        results["seq_lens"],
        results["enhanced"],
        "^-",
        label="Enhanced",
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Sequence Length")
    plt.ylabel("Time (ms)")
    plt.title("Performance Comparison by Sequence Length")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("kernel_performance_patterns.png", dpi=150)
    print("\n✓ Performance pattern plot saved to kernel_performance_patterns.png")


if __name__ == "__main__":
    analyze_performance_patterns()
    plot_performance_breakdown()
