#!/usr/bin/env python3
"""
Comprehensive benchmark comparing all Ring Attention implementations.
"""

import os
import torch
import time
import pandas as pd
from datetime import datetime

# Check if we're in distributed mode
IS_DISTRIBUTED = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

if IS_DISTRIBUTED:
    import torch.distributed as dist


def benchmark_single_gpu():
    """Benchmark all implementations in single GPU mode."""
    print("=" * 80)
    print("Single GPU Benchmark - Ring Attention Implementations")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import all implementations
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )
    from dilated_attention_pytorch.ring_dilated_attention_v2_robust import (
        RingDilatedAttentionV2Robust,
    )

    # Test parameters
    seq_lengths = [2048, 4096, 8192]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    results = []

    implementations = [
        ("Collective (baseline)", RingDilatedAttentionV2Collective),
        ("PyTorch Robust", RingDilatedAttentionV2Robust),
    ]

    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        print("-" * 40)

        for impl_name, impl_class in implementations:
            try:
                # Create model
                model = impl_class(
                    segment_lengths=[1024, 1024, 1024, 1024],
                    dilation_rates=[1, 2, 4, 8],
                    ring_size=1,
                    device=device,
                )

                dtype = model.dtype

                # Create input
                x = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(x, x, x, is_causal=True)
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                # Time
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.time()

                iterations = 10
                with torch.no_grad():
                    for _ in range(iterations):
                        _ = model(x, x, x, is_causal=True)
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                avg_time = (time.time() - start) / iterations

                # Memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    with torch.no_grad():
                        _ = model(x, x, x, is_causal=True)

                    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                else:
                    peak_memory_mb = 0

                print(
                    f"{impl_name:20} - Time: {avg_time * 1000:6.1f} ms, Memory: {peak_memory_mb:7.1f} MB"
                )

                results.append(
                    {
                        "Implementation": impl_name,
                        "Sequence Length": seq_len,
                        "Time (ms)": avg_time * 1000,
                        "Memory (MB)": peak_memory_mb,
                        "World Size": 1,
                    }
                )

            except Exception as e:
                print(f"{impl_name:20} - Error: {e}")
                results.append(
                    {
                        "Implementation": impl_name,
                        "Sequence Length": seq_len,
                        "Time (ms)": float("inf"),
                        "Memory (MB)": float("inf"),
                        "World Size": 1,
                    }
                )

    return results


def benchmark_multi_gpu():
    """Benchmark all implementations in multi-GPU mode."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize DeepSpeed if needed
    try:
        import deepspeed

        deepspeed.init_distributed()
    except ImportError:
        pass

    if rank == 0:
        print("=" * 80)
        print(f"Multi-GPU Benchmark ({world_size} GPUs)")
        print("=" * 80)

    # Import implementations
    from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
        RingDilatedAttentionV2Collective,
    )
    from dilated_attention_pytorch.ring_dilated_attention_v2_robust import (
        RingDilatedAttentionV2Robust,
    )

    # Test parameters
    seq_lengths = [2048, 4096, 8192]
    batch_size = 1
    num_heads = 8
    head_dim = 64

    results = []

    implementations = [
        ("Collective (all-gather)", RingDilatedAttentionV2Collective),
        ("PyTorch Ring", RingDilatedAttentionV2Robust),
    ]

    for seq_len in seq_lengths:
        if rank == 0:
            print(f"\nSequence length: {seq_len}")
            print("-" * 60)

        for impl_name, impl_class in implementations:
            try:
                # Create model
                model = impl_class(
                    segment_lengths=[1024, 1024, 1024, 1024],
                    dilation_rates=[1, 2, 4, 8],
                    ring_size=world_size,
                    device=device,
                )

                # Force distributed mode
                if hasattr(model, "min_seq_length_for_ring"):
                    model.min_seq_length_for_ring = 1

                dtype = model.dtype

                # Create input
                x = torch.randn(
                    batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
                )

                # Warmup
                with torch.no_grad():
                    for _ in range(2):
                        _ = model(x, x, x, is_causal=False)
                        torch.cuda.synchronize()

                # Time
                torch.cuda.synchronize()
                dist.barrier()
                start = time.time()

                iterations = 5
                with torch.no_grad():
                    for _ in range(iterations):
                        _ = model(x, x, x, is_causal=False)
                        torch.cuda.synchronize()

                dist.barrier()
                avg_time = (time.time() - start) / iterations

                # Memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                with torch.no_grad():
                    _ = model(x, x, x, is_causal=False)

                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                if rank == 0:
                    print(
                        f"{impl_name:25} - Time: {avg_time * 1000:6.1f} ms, Memory: {peak_memory_mb:7.1f} MB"
                    )

                if rank == 0:
                    results.append(
                        {
                            "Implementation": impl_name,
                            "Sequence Length": seq_len,
                            "Time (ms)": avg_time * 1000,
                            "Memory (MB)": peak_memory_mb,
                            "World Size": world_size,
                        }
                    )

            except Exception as e:
                if rank == 0:
                    print(f"{impl_name:25} - Error: {str(e)[:50]}...")
                    results.append(
                        {
                            "Implementation": impl_name,
                            "Sequence Length": seq_len,
                            "Time (ms)": float("inf"),
                            "Memory (MB)": float("inf"),
                            "World Size": world_size,
                        }
                    )

            # Sync before next test
            dist.barrier()

    dist.destroy_process_group()
    return results if rank == 0 else None


def save_results(results):
    """Save benchmark results to CSV and print summary."""
    if not results:
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    csv_file = f"benchmarks/ring_attention_comparison_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by world size
    for ws in df["World Size"].unique():
        ws_df = df[df["World Size"] == ws]
        print(f"\nWorld Size: {ws} GPU(s)")
        print("-" * 60)

        # Pivot table for easier comparison
        pivot = ws_df.pivot_table(
            index="Implementation",
            columns="Sequence Length",
            values=["Time (ms)", "Memory (MB)"],
        )

        print("\nTime (ms):")
        print(pivot["Time (ms)"].to_string())

        print("\nMemory (MB):")
        print(pivot["Memory (MB)"].to_string())

        # Calculate speedup relative to baseline
        baseline = "Collective (baseline)" if ws == 1 else "Collective (all-gather)"
        if baseline in ws_df["Implementation"].values:
            print("\nSpeedup relative to baseline:")
            for seq_len in ws_df["Sequence Length"].unique():
                seq_df = ws_df[ws_df["Sequence Length"] == seq_len]
                baseline_time = seq_df[seq_df["Implementation"] == baseline][
                    "Time (ms)"
                ].values[0]

                print(f"\nSequence {seq_len}:")
                for _, row in seq_df.iterrows():
                    if row["Implementation"] != baseline and row["Time (ms)"] != float(
                        "inf"
                    ):
                        speedup = baseline_time / row["Time (ms)"]
                        print(f"  {row['Implementation']:25} - {speedup:.2f}x")


if __name__ == "__main__":
    if IS_DISTRIBUTED:
        results = benchmark_multi_gpu()
        save_results(results)
    else:
        results = benchmark_single_gpu()
        save_results(results)
