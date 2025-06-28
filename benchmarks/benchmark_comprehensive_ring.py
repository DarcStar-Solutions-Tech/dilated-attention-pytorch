"""
Comprehensive benchmark of all Ring Attention implementations.

This script benchmarks:
1. RingDilatedAttentionV2 - Main implementation
2. RingDilatedAttentionProduction - Production-ready implementation
3. ImprovedDilatedAttention - Baseline
4. Different sequence lengths and configurations
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def measure_performance(
    model, batch_size, seq_len, num_heads, head_dim, device, dtype, runs=10
):
    """Measure performance metrics for a model."""
    # Create inputs
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(query, key, value, is_causal=False)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Forward timing
    forward_times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        forward_times.append(time.perf_counter() - start)

    # Memory measurement
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
    else:
        peak_memory = 0

    # Throughput calculation
    avg_time = np.mean(forward_times)
    throughput = (batch_size * seq_len) / avg_time  # tokens/sec

    return {
        "avg_time_ms": avg_time * 1000,
        "std_time_ms": np.std(forward_times) * 1000,
        "peak_memory_gb": peak_memory,
        "throughput_tokens_sec": throughput,
    }


def create_models(seq_len, device, dtype):
    """Create all model variants for benchmarking."""
    # Adaptive segment lengths based on sequence length
    if seq_len <= 2048:
        segment_lengths = [512, 1024]
    elif seq_len <= 8192:
        segment_lengths = [1024, 2048, 4096]
    else:
        segment_lengths = [2048, 4096, 8192]

    dilation_rates = [1, 2, 4][: len(segment_lengths)]

    models = {}

    # 1. RingDilatedAttentionV2 (single device)
    models["RingDilatedV2"] = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=dtype,
    )

    # 2. RingDilatedAttentionV2 (ring size 4)
    models["RingDilatedV2_Ring4"] = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=4,
        device=device,
        dtype=dtype,
    )

    # 3. Production without gradient checkpointing
    config_no_cp = RingAttentionConfig(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
        use_gradient_checkpointing=False,
        use_memory_pool=True,
        mixed_precision=(dtype == torch.float16),
    )
    models["Production_NoCP"] = RingDilatedAttentionProduction(config_no_cp)

    # 4. Production with gradient checkpointing
    config_with_cp = RingAttentionConfig(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
        use_gradient_checkpointing=True,
        use_memory_pool=True,
        mixed_precision=(dtype == torch.float16),
    )
    models["Production_CP"] = RingDilatedAttentionProduction(config_with_cp)

    # 5. Production with ring size 4
    config_ring4 = RingAttentionConfig(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=4,
        use_gradient_checkpointing=True,
        use_memory_pool=True,
        mixed_precision=(dtype == torch.float16),
    )
    models["Production_Ring4"] = RingDilatedAttentionProduction(config_ring4)

    # 6. Baseline ImprovedDilatedAttention
    models["ImprovedDilated"] = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device,
        dtype=dtype,
    )

    return models


def run_comprehensive_benchmark(args):
    """Run comprehensive benchmarks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Running comprehensive Ring Attention benchmark")
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print()

    # Results storage
    all_results = []

    for seq_len in args.seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"Sequence Length: {seq_len:,}")
        print(f"{'=' * 80}")

        # Skip if too large for memory
        estimated_memory = (
            args.batch_size * seq_len * args.num_heads * args.head_dim * 4
        ) / (1024**3)
        if estimated_memory > 8:  # Skip if > 8GB base memory estimate
            print(f"  Skipping - estimated memory too high: {estimated_memory:.1f} GB")
            continue

        # Create models for this sequence length
        models = create_models(seq_len, device, dtype)

        for name, model in models.items():
            print(f"\n{name}:")

            try:
                # Clear memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Run benchmark
                metrics = measure_performance(
                    model,
                    args.batch_size,
                    seq_len,
                    args.num_heads,
                    args.head_dim,
                    device,
                    dtype,
                    runs=args.runs,
                )

                print(
                    f"  Time: {metrics['avg_time_ms']:.2f} ± {metrics['std_time_ms']:.2f} ms"
                )
                print(f"  Memory: {metrics['peak_memory_gb']:.3f} GB")
                print(
                    f"  Throughput: {metrics['throughput_tokens_sec']:,.0f} tokens/sec"
                )

                # Store results
                result = {
                    "model": name,
                    "seq_len": seq_len,
                    "time_ms": metrics["avg_time_ms"],
                    "memory_gb": metrics["peak_memory_gb"],
                    "throughput": metrics["throughput_tokens_sec"],
                }
                all_results.append(result)

            except Exception as e:
                print(f"  Failed: {e}")
                result = {
                    "model": name,
                    "seq_len": seq_len,
                    "time_ms": float("inf"),
                    "memory_gb": float("inf"),
                    "throughput": 0,
                }
                all_results.append(result)

    # Create summary DataFrame
    df = pd.DataFrame(all_results)

    # Print summary table
    print(f"\n{'=' * 100}")
    print("SUMMARY TABLE")
    print(f"{'=' * 100}")

    # Pivot table for better visualization
    for metric in ["time_ms", "memory_gb", "throughput"]:
        print(f"\n{metric.upper()}:")
        pivot = df.pivot(index="seq_len", columns="model", values=metric)
        print(pivot.round(2))

    # Create visualizations
    if args.plot:
        create_plots(df, args)

    # Save results
    if args.save_results:
        save_results(df, all_results)

    return df


def create_plots(df, args):
    """Create performance visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Ring Attention Performance Comparison", fontsize=16)

    # Filter out failed results
    df_valid = df[df["time_ms"] != float("inf")]

    # 1. Time vs Sequence Length
    ax = axes[0, 0]
    for model in df_valid["model"].unique():
        model_data = df_valid[df_valid["model"] == model]
        ax.plot(
            model_data["seq_len"],
            model_data["time_ms"],
            marker="o",
            label=model,
            linewidth=2,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Forward Pass Time vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 2. Memory vs Sequence Length
    ax = axes[0, 1]
    for model in df_valid["model"].unique():
        model_data = df_valid[df_valid["model"] == model]
        ax.plot(
            model_data["seq_len"],
            model_data["memory_gb"],
            marker="o",
            label=model,
            linewidth=2,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Memory (GB)")
    ax.set_title("Memory Usage vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 3. Throughput vs Sequence Length
    ax = axes[1, 0]
    for model in df_valid["model"].unique():
        model_data = df_valid[df_valid["model"] == model]
        ax.plot(
            model_data["seq_len"],
            model_data["throughput"],
            marker="o",
            label=model,
            linewidth=2,
        )
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 4. Relative Performance (bar chart for max seq length)
    ax = axes[1, 1]
    max_seq_len = df_valid["seq_len"].max()
    df_max = df_valid[df_valid["seq_len"] == max_seq_len]

    if not df_max.empty:
        baseline_throughput = df_max[df_max["model"] == "ImprovedDilated"][
            "throughput"
        ].values[0]
        df_max["relative_throughput"] = df_max["throughput"] / baseline_throughput

        _ = ax.bar(range(len(df_max)), df_max["relative_throughput"])
        ax.set_xticks(range(len(df_max)))
        ax.set_xticklabels(df_max["model"], rotation=45, ha="right")
        ax.set_ylabel("Relative Throughput vs Baseline")
        ax.set_title(f"Relative Performance at Seq Length {max_seq_len}")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for i, (idx, row) in enumerate(df_max.iterrows()):
            ax.text(
                i,
                row["relative_throughput"] + 0.05,
                f"{row['relative_throughput']:.1f}x",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    # Save plot
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_path = (
        Path("docs/benchmarks") / f"ring-attention-comprehensive-{timestamp}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.close()


def save_results(df, all_results):
    """Save benchmark results to file."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as markdown
    md_path = output_dir / f"ring-attention-comprehensive-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Comprehensive Ring Attention Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Summary\n\n")
        f.write("Performance comparison of all Ring Attention implementations:\n")
        f.write("- **RingDilatedV2**: Main implementation\n")
        f.write("- **RingDilatedV2_Ring4**: With ring size 4 (simulated)\n")
        f.write(
            "- **Production_NoCP**: Production version without gradient checkpointing\n"
        )
        f.write("- **Production_CP**: Production version with gradient checkpointing\n")
        f.write("- **Production_Ring4**: Production version with ring size 4\n")
        f.write("- **ImprovedDilated**: Baseline dilated attention\n\n")

        f.write("## Results by Sequence Length\n\n")

        for seq_len in sorted(df["seq_len"].unique()):
            f.write(f"### Sequence Length: {seq_len:,}\n\n")

            seq_df = df[df["seq_len"] == seq_len].sort_values("time_ms")

            f.write(
                "| Model | Time (ms) | Memory (GB) | Throughput (tokens/sec) | Relative Speed |\n"
            )
            f.write(
                "|-------|-----------|-------------|------------------------|----------------|\n"
            )

            baseline_time = seq_df[seq_df["model"] == "ImprovedDilated"][
                "time_ms"
            ].values[0]

            for _, row in seq_df.iterrows():
                relative_speed = (
                    baseline_time / row["time_ms"] if row["time_ms"] > 0 else 0
                )
                f.write(
                    f"| {row['model']} | {row['time_ms']:.2f} | "
                    f"{row['memory_gb']:.3f} | {row['throughput']:,.0f} | "
                    f"{relative_speed:.2f}x |\n"
                )

            f.write("\n")

        f.write("## Key Findings\n\n")

        # Best performer analysis
        best_by_throughput = df.loc[df.groupby("seq_len")["throughput"].idxmax()]
        f.write("### Best Throughput by Sequence Length\n\n")
        for _, row in best_by_throughput.iterrows():
            f.write(
                f"- **{row['seq_len']:,} tokens**: {row['model']} "
                f"({row['throughput']:,.0f} tokens/sec)\n"
            )

        f.write("\n### Memory Efficiency\n\n")
        # Compare memory usage at max sequence length
        max_seq = df["seq_len"].max()
        mem_df = df[df["seq_len"] == max_seq].sort_values("memory_gb")
        if not mem_df.empty:
            baseline_mem = mem_df[mem_df["model"] == "ImprovedDilated"][
                "memory_gb"
            ].values[0]
            for _, row in mem_df.iterrows():
                reduction = (1 - row["memory_gb"] / baseline_mem) * 100
                f.write(
                    f"- **{row['model']}**: {row['memory_gb']:.3f} GB "
                    f"({reduction:+.1f}% vs baseline)\n"
                )

    print(f"\nResults saved to: {md_path}")

    # Also save as CSV for further analysis
    csv_path = output_dir / f"ring-attention-comprehensive-{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV data saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Ring Attention Benchmark"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--seq_lengths",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192, 16384],
        help="Sequence lengths to test",
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument(
        "--plot", action="store_true", help="Create visualization plots"
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )

    args = parser.parse_args()

    # Run benchmark
    df = run_comprehensive_benchmark(args)

    return df


if __name__ == "__main__":
    main()
