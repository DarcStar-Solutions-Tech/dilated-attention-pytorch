#!/usr/bin/env python3
"""
Comprehensive benchmark of all Block-Sparse implementations.

Compares:
1. Original BlockSparseRingDilatedAttention
2. BlockSparseOptimized (with batching and caching)
3. BlockSparseTorchSparse (using PyTorch sparse tensors)
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.block_sparse_optimized import BlockSparseOptimized
from dilated_attention_pytorch.block_sparse_torch_sparse import BlockSparseTorchSparse
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_implementation(
    name,
    model,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    device,
    dtype,
    runs=10,
    warmup=3,
    test_backward=False,
):
    """Benchmark a single implementation."""
    print(f"\n{name}:")

    # Create inputs
    query = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=test_backward,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    try:
        for _ in range(warmup):
            if test_backward:
                output = model(query, key, value, is_causal=False)
                loss = output.mean()
                loss.backward()
            else:
                with torch.no_grad():
                    _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Measure forward pass
        forward_times = []
        backward_times = []

        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Forward pass
            start = time.perf_counter()
            if test_backward:
                output = model(query, key, value, is_causal=False)
            else:
                with torch.no_grad():
                    output = model(query, key, value, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            forward_time = time.perf_counter() - start
            forward_times.append(forward_time)

            # Backward pass if requested
            if test_backward:
                start = time.perf_counter()
                loss = output.mean()
                loss.backward()

                if device.type == "cuda":
                    torch.cuda.synchronize()

                backward_time = time.perf_counter() - start
                backward_times.append(backward_time)

        # Get memory usage
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        # Calculate statistics
        forward_mean = np.mean(forward_times) * 1000  # ms
        forward_std = np.std(forward_times) * 1000

        print(f"  Forward: {forward_mean:.2f} ms (±{forward_std:.2f})")

        if test_backward:
            backward_mean = np.mean(backward_times) * 1000
            backward_std = np.std(backward_times) * 1000
            total_mean = (np.mean(forward_times) + np.mean(backward_times)) * 1000
            print(f"  Backward: {backward_mean:.2f} ms (±{backward_std:.2f})")
            print(f"  Total: {total_mean:.2f} ms")

        print(f"  Peak memory: {peak_memory:.2f} MB")

        # Get optimization stats if available
        if hasattr(model, "get_optimization_stats"):
            stats = model.get_optimization_stats()
            if "pattern_cache" in stats:
                cache_stats = stats["pattern_cache"]
                print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
            if "sparse_backend_enabled" in stats:
                print(f"  Sparse backend: {stats['sparse_backend_enabled']}")

        return {
            "name": name,
            "forward_mean_ms": forward_mean,
            "forward_std_ms": forward_std,
            "backward_mean_ms": backward_mean if test_backward else None,
            "backward_std_ms": backward_std if test_backward else None,
            "total_mean_ms": total_mean if test_backward else forward_mean,
            "peak_memory_mb": peak_memory,
            "success": True,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {
            "name": name,
            "success": False,
            "error": str(e),
        }
    finally:
        # Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def test_different_sparsities(seq_len, batch_size, num_heads, head_dim, device, dtype):
    """Test performance at different sparsity levels."""
    print("\n" + "=" * 80)
    print("SPARSITY COMPARISON")
    print("=" * 80)

    segment_lengths = [1024, 2048, 4096] if seq_len <= 4096 else [2048, 4096, 8192]
    dilation_rates = [1, 2, 4]

    sparsity_results = []

    for sparsity in [0.8, 0.9, 0.95, 0.98]:
        print(f"\nSparsity: {sparsity} ({(1 - sparsity) * 100:.0f}% active blocks)")

        sparse_config = SparsePatternConfig(
            pattern_type="local_window",
            sparsity_ratio=sparsity,
            block_size=64,
            local_window_size=256,
        )

        # Test torch sparse implementation
        model = BlockSparseTorchSparse(
            ring_size=1,
            device=device,
            dtype=dtype,
            sparse_config=sparse_config,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            use_sparse_backend=True,
            sparse_threshold=0.7,  # Use sparse backend for >70% sparsity
        )

        result = benchmark_implementation(
            f"TorchSparse_{sparsity}",
            model,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            device,
            dtype,
            runs=5,
        )

        if result["success"]:
            sparsity_results.append(
                {
                    "sparsity": sparsity,
                    "time_ms": result["forward_mean_ms"],
                    "memory_mb": result["peak_memory_mb"],
                }
            )

        del model

    return sparsity_results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Block-Sparse benchmark")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Sparsity ratio")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument(
        "--test_backward", action="store_true", help="Test backward pass"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Comprehensive Block-Sparse Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Sparsity: {args.sparsity}")
    print(f"Test backward: {args.test_backward}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Adjust segment lengths based on sequence length
    if args.seq_len <= 2048:
        segment_lengths = [512, 1024, 2048]
    elif args.seq_len <= 4096:
        segment_lengths = [1024, 2048, 4096]
    elif args.seq_len <= 8192:
        segment_lengths = [2048, 4096, 8192]
    else:
        segment_lengths = [4096, 8192, 16384]
    dilation_rates = [1, 2, 4]

    print(f"\nSegment lengths: {segment_lengths}")
    print(f"Dilation rates: {dilation_rates}")

    # Sparse configuration
    sparse_config = SparsePatternConfig(
        pattern_type="local_window",
        sparsity_ratio=args.sparsity,
        block_size=64,
        local_window_size=256,
    )

    results = []

    # Test all implementations
    implementations = [
        (
            "Baseline",
            lambda: ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            ),
        ),
        (
            "Original",
            lambda: BlockSparseRingDilatedAttention(
                ring_size=1,
                device=device,
                dtype=dtype,
                sparse_config=sparse_config,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
            ),
        ),
        (
            "Optimized",
            lambda: BlockSparseOptimized(
                ring_size=1,
                device=device,
                dtype=dtype,
                sparse_config=sparse_config,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                enable_batched_ops=True,
                cache_size=100,
            ),
        ),
        (
            "TorchSparse",
            lambda: BlockSparseTorchSparse(
                ring_size=1,
                device=device,
                dtype=dtype,
                sparse_config=sparse_config,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_sparse_backend=True,
                sparse_threshold=0.8,
            ),
        ),
    ]

    print("\n" + "=" * 80)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 80)

    for name, create_model in implementations:
        model = create_model()
        result = benchmark_implementation(
            name,
            model,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
            test_backward=args.test_backward,
        )
        results.append(result)
        del model

    # Test different sparsity levels
    sparsity_results = test_different_sparsities(
        args.seq_len,
        args.batch_size,
        args.num_heads,
        args.head_dim,
        device,
        dtype,
    )

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"block-sparse-comprehensive-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Comprehensive Block-Sparse Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Data type: {dtype}\n")
        f.write(f"- Sequence length: {args.seq_len}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Default sparsity: {args.sparsity}\n")
        f.write(f"- Segment lengths: {segment_lengths}\n")
        f.write(f"- Dilation rates: {dilation_rates}\n\n")

        f.write("## Implementation Comparison\n\n")

        if args.test_backward:
            f.write(
                "| Implementation | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) | Speedup |\n"
            )
            f.write(
                "|----------------|--------------|---------------|------------|-------------|---------||\n"
            )
        else:
            f.write(
                "| Implementation | Time (ms) | Memory (MB) | Speedup vs Original |\n"
            )
            f.write(
                "|----------------|-----------|-------------|--------------------|\n"
            )

        if all(r["success"] for r in results):
            _ = results[0]["total_mean_ms"]
            original_time = next(
                r["total_mean_ms"] for r in results if "Original" in r["name"]
            )

            for r in results:
                speedup = original_time / r["total_mean_ms"]

                if args.test_backward:
                    f.write(
                        f"| {r['name']} | {r['forward_mean_ms']:.2f} | "
                        f"{r['backward_mean_ms']:.2f} | {r['total_mean_ms']:.2f} | "
                        f"{r['peak_memory_mb']:.2f} | {speedup:.2f}x |\n"
                    )
                else:
                    f.write(
                        f"| {r['name']} | {r['total_mean_ms']:.2f} | "
                        f"{r['peak_memory_mb']:.2f} | {speedup:.2f}x |\n"
                    )

        f.write("\n## Sparsity Analysis\n\n")

        if sparsity_results:
            f.write("| Sparsity | Active Blocks | Time (ms) | Memory (MB) |\n")
            f.write("|----------|---------------|-----------|-------------|\n")

            for result in sparsity_results:
                active_pct = (1 - result["sparsity"]) * 100
                f.write(
                    f"| {result['sparsity']:.2f} | {active_pct:.0f}% | "
                    f"{result['time_ms']:.2f} | {result['memory_mb']:.2f} |\n"
                )

        f.write("\n## Key Findings\n\n")

        if all(r["success"] for r in results):
            # Find best performer
            best = min(results, key=lambda x: x["total_mean_ms"])
            worst = max(results, key=lambda x: x["total_mean_ms"])

            f.write(
                f"- Best performance: {best['name']} ({best['total_mean_ms']:.2f} ms)\n"
            )
            f.write(
                f"- Worst performance: {worst['name']} ({worst['total_mean_ms']:.2f} ms)\n"
            )

            # Calculate improvements
            original_result = next(r for r in results if "Original" in r["name"])
            optimized_result = next(r for r in results if "Optimized" in r["name"])
            sparse_result = next(r for r in results if "TorchSparse" in r["name"])

            opt_improvement = (
                (original_result["total_mean_ms"] - optimized_result["total_mean_ms"])
                / original_result["total_mean_ms"]
                * 100
            )
            sparse_improvement = (
                (original_result["total_mean_ms"] - sparse_result["total_mean_ms"])
                / original_result["total_mean_ms"]
                * 100
            )

            f.write(f"- Optimized improvement over Original: {opt_improvement:.1f}%\n")
            f.write(
                f"- TorchSparse improvement over Original: {sparse_improvement:.1f}%\n"
            )

    print(f"\n\nResults saved to: {md_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all(r["success"] for r in results):
        for r in results:
            print(f"{r['name']:20} {r['total_mean_ms']:8.2f} ms")

        print("\nSparsity Impact:")
        for result in sparsity_results:
            print(f"  {result['sparsity']:.2f} sparsity: {result['time_ms']:8.2f} ms")


if __name__ == "__main__":
    main()
