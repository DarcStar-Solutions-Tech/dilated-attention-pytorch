#!/usr/bin/env python3
"""
Comprehensive benchmark report for dilated attention implementations.
"""

import time
import torch
import gc
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def benchmark_implementation(name, model_fn, inputs, num_runs=5, warmup=2):
    """Benchmark a model with timing and memory tracking."""
    q, k, v = inputs
    device = q.device

    try:
        model = model_fn()
        model = model.to(device).eval()
    except Exception as e:
        return {"error": str(e)}

    # Warmup
    try:
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(q, k, v)
            if device.type == "cuda":
                torch.cuda.synchronize()
    except Exception as e:
        return {"error": f"Forward failed: {str(e)}"}

    # Clear cache and measure
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

    # Time runs
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            output = model(q, k, v)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = np.std(times)

    # Memory usage
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated()
        mem_used = (peak_mem - start_mem) / 1024**2  # MB
    else:
        mem_used = 0

    return {
        "time_ms": avg_time * 1000,
        "time_std_ms": std_time * 1000,
        "memory_mb": mem_used,
        "output_shape": list(output.shape),
    }


def main():
    """Run comprehensive benchmarks."""
    print("=== Comprehensive Dilated Attention Benchmark Report ===")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {timestamp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )
    else:
        print("Device: CPU")

    # Test configurations
    test_configs = [
        {
            "name": "Small",
            "seq_len": 2048,
            "batch_size": 4,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [1024, 2048],
            "dilation_rates": [1, 2],
        },
        {
            "name": "Medium",
            "seq_len": 4096,
            "batch_size": 2,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [2048, 4096],
            "dilation_rates": [1, 2],
        },
        {
            "name": "Large",
            "seq_len": 8192,
            "batch_size": 1,
            "num_heads": 8,
            "head_dim": 64,
            "segment_lengths": [4096, 8192],
            "dilation_rates": [1, 2],
        },
    ]

    if device.type == "cuda":
        # Add extra large config for GPU
        test_configs.append(
            {
                "name": "Extra Large",
                "seq_len": 16384,
                "batch_size": 1,
                "num_heads": 8,
                "head_dim": 64,
                "segment_lengths": [8192, 16384],
                "dilation_rates": [1, 2],
            }
        )

    results = defaultdict(dict)

    print("\nRunning benchmarks...")
    print("-" * 80)

    for config in test_configs:
        print(f"\n### {config['name']} Configuration")
        print(f"Sequence Length: {config['seq_len']:,}")
        print(f"Batch Size: {config['batch_size']}")

        # Create inputs
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        q = torch.randn(
            config["batch_size"],
            config["seq_len"],
            config["num_heads"],
            config["head_dim"],
            device=device,
            dtype=dtype,
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Define implementations
        implementations = []

        # 1. PyTorch baseline
        def pytorch_attention():
            class PyTorchAttention(torch.nn.Module):
                def forward(self, q, k, v):
                    b, s, h, d = q.shape
                    q_reshaped = q.transpose(1, 2).reshape(b * h, s, d)
                    k_reshaped = k.transpose(1, 2).reshape(b * h, s, d)
                    v_reshaped = v.transpose(1, 2).reshape(b * h, s, d)

                    scores = torch.bmm(q_reshaped, k_reshaped.transpose(-2, -1)) / (
                        d**0.5
                    )
                    attn = torch.softmax(scores, dim=-1)
                    out = torch.bmm(attn, v_reshaped)

                    return out.reshape(b, h, s, d).transpose(1, 2)

            return PyTorchAttention()

        implementations.append(("PyTorch (Baseline)", pytorch_attention))

        # 2. Block-Sparse variants
        from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
            BlockSparseRingDilatedAttention,
            SparsePatternConfig,
        )

        sparsity_configs = [
            ("Block-Sparse (90%)", 0.1),
            ("Block-Sparse (95%)", 0.05),
            ("Block-Sparse (98%)", 0.02),
        ]

        for name, sparsity in sparsity_configs:

            def make_sparse(sparsity_ratio=sparsity):
                sparse_config = SparsePatternConfig(
                    pattern_type="dilated_sparse",
                    sparsity_ratio=sparsity_ratio,
                    block_size=64,
                )

                return BlockSparseRingDilatedAttention(
                    segment_lengths=config["segment_lengths"],
                    dilation_rates=config["dilation_rates"],
                    sparse_config=sparse_config,
                )

            implementations.append((name, make_sparse))

        # Benchmark each implementation
        for impl_name, impl_fn in implementations:
            result = benchmark_implementation(impl_name, impl_fn, (q, k, v))
            results[config["name"]][impl_name] = result

            if "error" in result:
                print(f"\n{impl_name}: ❌ {result['error']}")
            else:
                print(f"\n{impl_name}:")
                print(
                    f"  Time: {result['time_ms']:.1f} ± {result['time_std_ms']:.1f}ms"
                )
                print(f"  Memory: {result['memory_mb']:.1f}MB")

                # Calculate speedup
                if "PyTorch (Baseline)" in results[config["name"]]:
                    baseline = results[config["name"]]["PyTorch (Baseline)"]
                    if "time_ms" in baseline and "time_ms" in result:
                        speedup = baseline["time_ms"] / result["time_ms"]
                        print(f"  Speedup: {speedup:.2f}x")

    # Save results
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(
            {"timestamp": timestamp, "device": str(device), "results": results},
            f,
            indent=2,
        )

    print(f"\n\nResults saved to: {output_file}")

    # Generate plots if we have results
    if any(results.values()):
        plot_results(results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"{'Config':<12} {'Implementation':<25} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}"
    )
    print("-" * 80)

    for config_name, impl_results in results.items():
        baseline_time = impl_results.get("PyTorch (Baseline)", {}).get("time_ms", 0)

        for impl_name, result in impl_results.items():
            if "error" not in result:
                time_ms = result["time_ms"]
                mem_mb = result["memory_mb"]
                speedup = baseline_time / time_ms if baseline_time > 0 else 0

                print(
                    f"{config_name:<12} {impl_name:<25} {time_ms:<12.1f} {mem_mb:<12.1f} {speedup:<10.2f}x"
                )


def plot_results(results):
    """Generate visualization plots."""
    try:
        # Extract data for plotting
        configs = list(results.keys())
        implementations = list(next(iter(results.values())).keys())

        # Create speedup plot
        plt.figure(figsize=(10, 6))

        x = np.arange(len(configs))
        width = 0.2

        for i, impl in enumerate(implementations):
            if impl == "PyTorch (Baseline)":
                continue

            speedups = []
            for config in configs:
                baseline = (
                    results[config].get("PyTorch (Baseline)", {}).get("time_ms", 1)
                )
                impl_time = results[config].get(impl, {}).get("time_ms", baseline)
                speedups.append(baseline / impl_time)

            plt.bar(x + i * width, speedups, width, label=impl)

        plt.xlabel("Configuration")
        plt.ylabel("Speedup vs PyTorch Baseline")
        plt.title("Block-Sparse Attention Speedup")
        plt.xticks(x + width, configs)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig("benchmark_speedup_plot.png", dpi=150, bbox_inches="tight")
        print("\nSpeedup plot saved to: benchmark_speedup_plot.png")

    except Exception as e:
        print(f"\nFailed to generate plots: {e}")


if __name__ == "__main__":
    main()
