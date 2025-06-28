"""
Benchmark Block-Sparse Ring Attention after Ring Attention fixes.

This script specifically tests BlockSparse implementations to see if they
benefit from the recent Ring Attention normalization fixes.
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_implementation(
    name, model, batch_size, seq_len, num_heads, head_dim, device, dtype, runs=10
):
    """Benchmark a single implementation."""
    print(f"\n{name}:")

    # Create inputs
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    try:
        for _ in range(3):
            with torch.no_grad():
                _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Measure forward pass
        times = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                output = model(query, key, value, is_causal=False)

            if device.type == "cuda":
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        # Get memory usage
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0

        # Calculate statistics
        mean_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000

        print(f"  Mean time: {mean_time:.2f} ms (±{std_time:.2f})")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  Peak memory: {peak_memory:.2f} MB")
        print(f"  Output shape: {output.shape}")

        return {
            "name": name,
            "mean_time_ms": mean_time,
            "std_time_ms": std_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "peak_memory_mb": peak_memory,
            "success": True,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Block-Sparse after Ring Attention fixes"
    )
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Block-Sparse Ring Attention Benchmark (Post-Fix)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Segment configuration
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    print(f"\nSegment lengths: {segment_lengths}")
    print(f"Dilation rates: {dilation_rates}")

    results = []

    # 1. Baseline: ImprovedDilatedAttention
    print("\n" + "=" * 80)
    print("BASELINE IMPLEMENTATIONS")
    print("=" * 80)

    model = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
    )
    results.append(
        benchmark_implementation(
            "ImprovedDilatedAttention",
            model,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del model

    # 2. Ring Attention V2 (fixed version)
    model = RingDilatedAttentionV2(
        ring_size=1,  # Single device
        device=device,
        dtype=dtype,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
    )
    results.append(
        benchmark_implementation(
            "RingDilatedAttentionV2",
            model,
            args.batch_size,
            args.seq_len,
            args.num_heads,
            args.head_dim,
            device,
            dtype,
            args.runs,
        )
    )
    del model

    # 3. Block-Sparse with different sparsity levels
    print("\n" + "=" * 80)
    print("BLOCK-SPARSE IMPLEMENTATIONS")
    print("=" * 80)

    for sparsity_ratio in [0.9, 0.95, 0.98]:
        print(f"\nSparsity ratio: {sparsity_ratio}")

        # Local window pattern
        sparse_config = SparsePatternConfig(
            pattern_type="local_window",
            sparsity_ratio=sparsity_ratio,
            block_size=64,
            local_window_size=256,
        )

        model = BlockSparseRingDilatedAttention(
            ring_size=1,
            device=device,
            dtype=dtype,
            sparse_config=sparse_config,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
        )

        results.append(
            benchmark_implementation(
                f"BlockSparse_LocalWindow_{sparsity_ratio}",
                model,
                args.batch_size,
                args.seq_len,
                args.num_heads,
                args.head_dim,
                device,
                dtype,
                args.runs,
            )
        )
        del model

        # Dilated sparse pattern
        sparse_config = SparsePatternConfig(
            pattern_type="dilated_sparse",
            sparsity_ratio=sparsity_ratio,
            block_size=64,
        )

        model = BlockSparseRingDilatedAttention(
            ring_size=1,
            device=device,
            dtype=dtype,
            sparse_config=sparse_config,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
        )

        results.append(
            benchmark_implementation(
                f"BlockSparse_Dilated_{sparsity_ratio}",
                model,
                args.batch_size,
                args.seq_len,
                args.num_heads,
                args.head_dim,
                device,
                dtype,
                args.runs,
            )
        )
        del model

    # Save results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"block-sparse-post-fix-benchmark-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Block-Sparse Ring Attention Benchmark (Post Ring Attention Fix)\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Sequence length: {args.seq_len}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Segment lengths: {segment_lengths}\n")
        f.write(f"- Dilation rates: {dilation_rates}\n\n")

        f.write("## Results\n\n")

        f.write(
            "| Implementation | Mean Time (ms) | Std Dev | Memory (MB) | Speedup vs Baseline |\n"
        )
        f.write(
            "|---------------|----------------|---------|-------------|--------------------|\n"
        )

        baseline_time = (
            results[0]["mean_time_ms"] if results[0]["success"] else float("inf")
        )

        for r in results:
            if r["success"]:
                speedup = baseline_time / r["mean_time_ms"]
                f.write(
                    f"| {r['name']} | {r['mean_time_ms']:.2f} | "
                    f"±{r['std_time_ms']:.2f} | {r['peak_memory_mb']:.2f} | "
                    f"{speedup:.2f}x |\n"
                )
            else:
                f.write(f"| {r['name']} | FAILED | - | - | - |\n")

        f.write("\n## Analysis\n\n")

        # Compare block-sparse performance
        block_sparse_results = [
            r for r in results if "BlockSparse" in r["name"] and r["success"]
        ]
        if block_sparse_results:
            best_bs = min(block_sparse_results, key=lambda x: x["mean_time_ms"])
            worst_bs = max(block_sparse_results, key=lambda x: x["mean_time_ms"])

            f.write(
                f"- Best Block-Sparse: {best_bs['name']} ({best_bs['mean_time_ms']:.2f} ms)\n"
            )
            f.write(
                f"- Worst Block-Sparse: {worst_bs['name']} ({worst_bs['mean_time_ms']:.2f} ms)\n"
            )

            if baseline_time < float("inf"):
                best_speedup = baseline_time / best_bs["mean_time_ms"]
                f.write(f"- Best speedup vs baseline: {best_speedup:.2f}x\n")

        f.write("\n## Notes\n\n")
        f.write(
            "This benchmark was run after fixing Ring Attention normalization issues.\n"
        )
        f.write(
            "The goal is to verify if Block-Sparse implementations benefit from the fixes.\n"
        )

    print(f"\nResults saved to: {md_path}")

    # Print summary
    print("\nPerformance Summary:")
    print("-" * 50)
    for r in results:
        if r["success"]:
            print(f"{r['name']:40} {r['mean_time_ms']:8.2f} ms")
        else:
            print(f"{r['name']:40} FAILED")


if __name__ == "__main__":
    main()
