"""
Benchmark script for production Ring Attention implementation.

Compares performance and memory usage of:
1. RingDilatedAttentionV2 (existing implementation)
2. RingDilatedAttentionProduction (new production-ready implementation)
3. ImprovedDilatedAttention (baseline)
"""

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
import numpy as np

import torch

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
    RingAttentionConfig,
)
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def measure_performance(model, inputs, warmup=3, runs=10):
    """Measure forward and backward pass performance."""
    query, key, value = inputs

    # Warmup
    for _ in range(warmup):
        q_temp = query.clone().requires_grad_(True)
        k_temp = key.clone().requires_grad_(True)
        v_temp = value.clone().requires_grad_(True)
        output = model(q_temp, k_temp, v_temp, is_causal=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Forward timing
    forward_times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(query, key, value, is_causal=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_times.append(time.perf_counter() - start)

    # Backward timing
    backward_times = []
    for _ in range(runs):
        # Create fresh tensors with gradients for each backward pass
        q_temp = query.clone().requires_grad_(True)
        k_temp = key.clone().requires_grad_(True)
        v_temp = value.clone().requires_grad_(True)

        output = model(q_temp, k_temp, v_temp, is_causal=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        loss = output.sum()
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        backward_times.append(time.perf_counter() - start)

    # Memory stats
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        peak_memory = 0

    return {
        "forward_ms": np.mean(forward_times) * 1000,
        "forward_std": np.std(forward_times) * 1000,
        "backward_ms": np.mean(backward_times) * 1000,
        "backward_std": np.std(backward_times) * 1000,
        "peak_memory_gb": peak_memory,
    }


def create_models(segment_lengths, dilation_rates, device, dtype):
    """Create all models for comparison."""
    models = {}

    # 1. Existing RingDilatedAttentionV2
    models["RingDilatedV2"] = RingDilatedAttentionV2(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
        device=device,
        dtype=dtype,
    )

    # 2. Production implementation (no gradient checkpointing)
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

    # 3. Production implementation (with gradient checkpointing)
    config_with_cp = RingAttentionConfig(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        ring_size=1,
        use_gradient_checkpointing=True,
        use_memory_pool=True,
        mixed_precision=(dtype == torch.float16),
    )
    models["Production_WithCP"] = RingDilatedAttentionProduction(config_with_cp)

    # 4. Production with larger ring size (simulated)
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

    # 5. Baseline ImprovedDilatedAttention
    models["ImprovedDilated"] = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        dropout=0.0,
        device=device,
        dtype=dtype,
    )

    return models


def run_benchmark(args):
    """Run comprehensive benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    print(f"Running benchmark on {device} with dtype={dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num heads: {args.num_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print()

    # Create models
    segment_lengths = [min(seq_len // 4, 2048) for seq_len in args.seq_lengths[:2]]
    dilation_rates = [1, 2]

    models = create_models(segment_lengths, dilation_rates, device, dtype)

    # Results storage
    results = {}

    for seq_len in args.seq_lengths:
        print(f"\n{'=' * 60}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'=' * 60}")

        # Create test tensors
        torch.manual_seed(42)
        query = torch.randn(
            args.batch_size,
            seq_len,
            args.num_heads,
            args.head_dim,
            device=device,
            dtype=dtype,
        )
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        inputs = (query, key, value)

        seq_results = {}

        for name, model in models.items():
            print(f"\nBenchmarking {name}...")

            try:
                # Clear memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                # Run benchmark
                perf = measure_performance(
                    model, inputs, warmup=args.warmup, runs=args.runs
                )

                seq_results[name] = perf

                print(
                    f"  Forward:  {perf['forward_ms']:.2f} ± {perf['forward_std']:.2f} ms"
                )
                print(
                    f"  Backward: {perf['backward_ms']:.2f} ± {perf['backward_std']:.2f} ms"
                )
                print(f"  Memory:   {perf['peak_memory_gb']:.3f} GB")

                # Additional stats for production model
                if hasattr(model, "get_memory_stats"):
                    stats = model.get_memory_stats()
                    if "memory_pool" in stats:
                        pool_stats = stats["memory_pool"]
                        print(f"  Pool reuse rate: {pool_stats['reuse_rate']:.2%}")

            except Exception as e:
                print(f"  Failed: {e}")
                seq_results[name] = None

        results[seq_len] = seq_results

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    for seq_len, seq_results in results.items():
        print(f"\nSequence Length: {seq_len}")
        print(f"{'-' * 70}")
        print(
            f"{'Model':<20} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Memory (GB)':<12}"
        )
        print(f"{'-' * 70}")

        for name, perf in seq_results.items():
            if perf is not None:
                print(
                    f"{name:<20} "
                    f"{perf['forward_ms']:>6.2f} ± {perf['forward_std']:>4.2f}  "
                    f"{perf['backward_ms']:>6.2f} ± {perf['backward_std']:>4.2f}  "
                    f"{perf['peak_memory_gb']:>8.3f}"
                )
            else:
                print(f"{name:<20} {'FAILED':<15} {'FAILED':<15} {'N/A':<12}")

    # Save results
    if args.save_results:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
        output_dir = Path("docs/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"ring-production-benchmark-{timestamp}.md"

        with open(output_file, "w") as f:
            f.write("# Ring Attention Production Benchmark\n\n")
            f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
            f.write("## Configuration\n")
            f.write(f"- Device: {device}\n")
            f.write(f"- Data type: {dtype}\n")
            f.write(f"- Batch size: {args.batch_size}\n")
            f.write(f"- Num heads: {args.num_heads}\n")
            f.write(f"- Head dim: {args.head_dim}\n")
            f.write(f"- Segment lengths: {segment_lengths}\n")
            f.write(f"- Dilation rates: {dilation_rates}\n\n")

            f.write("## Results\n\n")

            for seq_len, seq_results in results.items():
                f.write(f"### Sequence Length: {seq_len}\n\n")
                f.write("| Model | Forward (ms) | Backward (ms) | Memory (GB) |\n")
                f.write("|-------|--------------|---------------|-------------|\n")

                for name, perf in seq_results.items():
                    if perf is not None:
                        f.write(
                            f"| {name} | "
                            f"{perf['forward_ms']:.2f} ± {perf['forward_std']:.2f} | "
                            f"{perf['backward_ms']:.2f} ± {perf['backward_std']:.2f} | "
                            f"{perf['peak_memory_gb']:.3f} |\n"
                        )
                    else:
                        f.write(f"| {name} | FAILED | FAILED | N/A |\n")

                f.write("\n")

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark production Ring Attention")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--seq_lengths",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096],
        help="Sequence lengths to test",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to file"
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
