"""
Benchmark Ring Attention implementations on extra long sequences.

This script tests the limits of Ring Attention with sequences up to 1M tokens,
demonstrating the O(n/ring_size) memory scaling.
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import psutil

import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
)


def get_memory_usage():
    """Get current memory usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    else:
        # CPU memory
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)


def test_sequence_length(
    model,
    seq_len,
    batch_size=1,
    num_heads=8,
    head_dim=64,
    device="cuda",
    dtype=torch.float16,
    warmup=1,
    runs=3,
):
    """Test a specific sequence length and return metrics."""
    print(f"\n  Testing seq_len={seq_len:,}...")

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    initial_memory = get_memory_usage()

    try:
        # Create inputs
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        input_memory = get_memory_usage() - initial_memory
        print(f"    Input tensors: {input_memory:.3f} GB")

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(query, key, value, is_causal=False)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Time forward pass
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

        avg_time = np.mean(times)
        peak_memory = get_memory_usage()
        model_memory = peak_memory - initial_memory - input_memory

        # Calculate throughput
        throughput = (batch_size * seq_len) / avg_time

        print("    ✓ Success!")
        print(f"    Time: {avg_time * 1000:.1f} ms")
        print(f"    Model memory: {model_memory:.3f} GB")
        print(f"    Total memory: {peak_memory:.3f} GB")
        print(f"    Throughput: {throughput:,.0f} tokens/sec")

        return {
            "success": True,
            "seq_len": seq_len,
            "time_ms": avg_time * 1000,
            "model_memory_gb": model_memory,
            "total_memory_gb": peak_memory,
            "throughput": throughput,
        }

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return {
            "success": False,
            "seq_len": seq_len,
            "error": str(e),
        }
    finally:
        # Cleanup
        if "query" in locals():
            del query, key, value
        if "output" in locals():
            del output
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


def run_extra_long_benchmark(args):
    """Run benchmark on extra long sequences."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print("Ring Attention Extra Long Sequence Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data type: {dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU memory: {total_memory:.1f} GB")

    # Results storage
    results = {
        "RingDilatedV2_r1": [],
        "RingDilatedV2_r4": [],
        "RingDilatedV2_r8": [],
        "Production_r1": [],
        "Production_r4": [],
        "Production_r8": [],
        "Production_r16": [],
    }

    # Test different configurations
    configs = [
        ("RingDilatedV2_r1", 1, False),
        ("RingDilatedV2_r4", 4, False),
        ("RingDilatedV2_r8", 8, False),
        ("Production_r1", 1, True),
        ("Production_r4", 4, True),
        ("Production_r8", 8, True),
        ("Production_r16", 16, True),
    ]

    for seq_len in args.seq_lengths:
        print(f"\n{'=' * 80}")
        print(f"SEQUENCE LENGTH: {seq_len:,}")
        print(f"{'=' * 80}")

        # Adaptive segment lengths for very long sequences
        if seq_len <= 16384:
            segment_lengths = [2048, 4096, 8192]
        elif seq_len <= 65536:
            segment_lengths = [4096, 8192, 16384]
        elif seq_len <= 262144:
            segment_lengths = [8192, 16384, 32768]
        else:
            segment_lengths = [16384, 32768, 65536]

        dilation_rates = [1, 2, 4]

        print(f"Segment lengths: {segment_lengths}")
        print(f"Dilation rates: {dilation_rates}")

        for name, ring_size, use_production in configs:
            print(f"\n{name}:")

            try:
                if use_production:
                    # Production version
                    config = RingAttentionConfig(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                        ring_size=ring_size,
                        use_gradient_checkpointing=True,
                        use_memory_pool=True,
                        mixed_precision=(dtype == torch.float16),
                        log_memory_usage=False,
                    )
                    model = RingDilatedAttentionProduction(config)
                else:
                    # Standard version
                    model = RingDilatedAttentionV2(
                        segment_lengths=segment_lengths,
                        dilation_rates=dilation_rates,
                        dropout=0.0,
                        ring_size=ring_size,
                        device=device,
                        dtype=dtype,
                    )

                # Test this configuration
                result = test_sequence_length(
                    model,
                    seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    device,
                    dtype,
                    warmup=1,
                    runs=args.runs,
                )

                results[name].append(result)

                # Cleanup model
                del model

            except Exception as e:
                print(f"  Failed to create model: {e}")
                results[name].append(
                    {
                        "success": False,
                        "seq_len": seq_len,
                        "error": f"Model creation failed: {e}",
                    }
                )

    # Create visualizations and save results
    create_extra_long_plots(results, args)
    save_extra_long_results(results, args)

    return results


def create_extra_long_plots(results, args):
    """Create visualization plots for extra long sequences."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Ring Attention on Extra Long Sequences", fontsize=16)

    # Extract successful results
    for name, data in results.items():
        successful = [r for r in data if r.get("success", False)]
        if not successful:
            continue

        seq_lens = [r["seq_len"] for r in successful]
        times = [r["time_ms"] for r in successful]
        memories = [r["total_memory_gb"] for r in successful]
        throughputs = [r["throughput"] for r in successful]

        # 1. Time vs Sequence Length
        ax = axes[0, 0]
        ax.plot(seq_lens, times, marker="o", label=name, linewidth=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title("Forward Pass Time Scaling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 2. Memory vs Sequence Length
        ax = axes[0, 1]
        ax.plot(seq_lens, memories, marker="o", label=name, linewidth=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Total Memory (GB)")
        ax.set_title("Memory Usage Scaling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # 3. Throughput vs Sequence Length
        ax = axes[1, 0]
        ax.plot(seq_lens, throughputs, marker="o", label=name, linewidth=2)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput Scaling")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 4. Memory efficiency comparison (bar chart)
    ax = axes[1, 1]

    # Find the largest sequence length tested successfully by all
    max_seq_len = 0
    for name, data in results.items():
        successful = [r["seq_len"] for r in data if r.get("success", False)]
        if successful:
            max_seq_len = max(max_seq_len, max(successful))

    if max_seq_len > 0:
        # Get memory usage at max sequence length
        memories_at_max = {}
        for name, data in results.items():
            for r in data:
                if r.get("success", False) and r["seq_len"] == max_seq_len:
                    memories_at_max[name] = r["total_memory_gb"]
                    break

        if memories_at_max:
            names = list(memories_at_max.keys())
            mems = list(memories_at_max.values())

            _ = ax.bar(range(len(names)), mems)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylabel("Memory Usage (GB)")
            ax.set_title(f"Memory Usage at {max_seq_len:,} Tokens")
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for i, mem in enumerate(mems):
                ax.text(i, mem + 0.1, f"{mem:.2f}", ha="center", va="bottom")

    plt.tight_layout()

    # Save plot
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_path = Path("docs/benchmarks") / f"ring-attention-extra-long-{timestamp}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    plt.close()


def save_extra_long_results(results, args):
    """Save benchmark results for extra long sequences."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path("docs/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / f"ring-attention-extra-long-{timestamp}.md"

    with open(md_path, "w") as f:
        f.write("# Ring Attention Extra Long Sequence Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(
            f"- Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n"
        )
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Num heads: {args.num_heads}\n")
        f.write(f"- Head dim: {args.head_dim}\n")
        f.write(f"- Sequence lengths tested: {args.seq_lengths}\n\n")

        f.write("## Results\n\n")

        # Create summary table
        f.write("### Summary Table\n\n")
        f.write(
            "| Model | Ring Size | Max Seq Length | Memory at Max | Throughput at Max |\n"
        )
        f.write(
            "|-------|-----------|----------------|---------------|-------------------|\n"
        )

        for name, data in results.items():
            successful = [r for r in data if r.get("success", False)]
            if successful:
                max_result = max(successful, key=lambda x: x["seq_len"])
                ring_size = name.split("_r")[1]
                f.write(
                    f"| {name} | {ring_size} | {max_result['seq_len']:,} | "
                    f"{max_result['total_memory_gb']:.2f} GB | "
                    f"{max_result['throughput']:,.0f} tok/s |\n"
                )

        f.write("\n### Detailed Results by Sequence Length\n\n")

        # Group by sequence length
        seq_lens = set()
        for data in results.values():
            for r in data:
                seq_lens.add(r["seq_len"])

        for seq_len in sorted(seq_lens):
            f.write(f"\n#### Sequence Length: {seq_len:,} tokens\n\n")

            f.write(
                "| Model | Success | Time (ms) | Memory (GB) | Throughput (tok/s) |\n"
            )
            f.write(
                "|-------|---------|-----------|-------------|--------------------||\n"
            )

            for name, data in results.items():
                for r in data:
                    if r["seq_len"] == seq_len:
                        if r.get("success", False):
                            f.write(
                                f"| {name} | ✓ | {r['time_ms']:.1f} | "
                                f"{r['total_memory_gb']:.2f} | "
                                f"{r['throughput']:,.0f} |\n"
                            )
                        else:
                            f.write(f"| {name} | ✗ | - | - | - |\n")
                        break

        f.write("\n## Key Findings\n\n")

        # Analysis
        f.write("### Memory Scaling\n\n")
        f.write("Ring Attention demonstrates O(n/ring_size) memory scaling:\n\n")

        # Calculate memory reduction ratios
        for seq_len in sorted(seq_lens):
            r1_mem = None
            for r in results.get("RingDilatedV2_r1", []):
                if r["seq_len"] == seq_len and r.get("success", False):
                    r1_mem = r["total_memory_gb"]
                    break

            if r1_mem:
                f.write(f"At {seq_len:,} tokens:\n")
                for name in [
                    "RingDilatedV2_r4",
                    "RingDilatedV2_r8",
                    "Production_r4",
                    "Production_r8",
                    "Production_r16",
                ]:
                    for r in results.get(name, []):
                        if r["seq_len"] == seq_len and r.get("success", False):
                            reduction = (1 - r["total_memory_gb"] / r1_mem) * 100
                            f.write(f"- {name}: {reduction:.1f}% memory reduction\n")
                            break
                f.write("\n")

        f.write("### Maximum Sequence Lengths Achieved\n\n")
        for name, data in results.items():
            successful = [r["seq_len"] for r in data if r.get("success", False)]
            if successful:
                max_len = max(successful)
                f.write(f"- **{name}**: {max_len:,} tokens\n")

    print(f"\nResults saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ring Attention on extra long sequences"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--seq_lengths",
        nargs="+",
        type=int,
        default=[16384, 32768, 65536, 131072, 262144, 524288, 1048576],
        help="Sequence lengths to test",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    args = parser.parse_args()

    # Run benchmark
    results = run_extra_long_benchmark(args)

    return results


if __name__ == "__main__":
    main()
