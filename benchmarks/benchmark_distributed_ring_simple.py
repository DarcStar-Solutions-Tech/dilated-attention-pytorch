"""
Simple distributed benchmark for Ring Attention on 2 GPUs.

This version:
1. Uses simulated ring mode to avoid distributed API issues
2. Tests smaller sequences that fit in GTX 1080 memory
3. Compares single vs multi-GPU simulation performance
"""

import os
import torch
import torch.multiprocessing as mp
import time
import argparse
from typing import List
from dataclasses import dataclass
import json
from datetime import datetime

from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


@dataclass
class SimpleBenchmarkResult:
    """Result for simple benchmark."""

    implementation: str
    sequence_length: int
    batch_size: int
    num_heads: int
    head_dim: int
    ring_size: int
    mode: str
    gpu_id: int
    time_ms: float
    memory_mb: float
    throughput_tokens_per_sec: float
    pattern_cache_enabled: bool = False
    memory_pool_enabled: bool = False


def benchmark_on_gpu(
    gpu_id: int,
    implementation: str,
    seq_len: int,
    batch_size: int,
    num_heads: int = 8,
    head_dim: int = 64,
    ring_size: int = 1,
    enable_pattern_cache: bool = False,
    enable_memory_pool: bool = False,
    warmup_steps: int = 3,
    benchmark_steps: int = 10,
) -> SimpleBenchmarkResult:
    """Benchmark on a single GPU."""

    # Set device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Adjust sequence length based on memory constraints
    if seq_len > 8192:
        # For larger sequences, use smaller dimensions
        num_heads = 4
        head_dim = 32

    # Get segment configuration
    if seq_len <= 4096:
        segment_lengths = [512, 1024, 2048]
        dilation_rates = [1, 2, 4]
    elif seq_len <= 8192:
        segment_lengths = [1024, 2048, 4096]
        dilation_rates = [1, 2, 4]
    else:
        segment_lengths = [2048, 4096, 8192]
        dilation_rates = [1, 2, 4]

    # Ensure divisibility
    max_segment = max(segment_lengths)
    seq_len = (seq_len // max_segment) * max_segment

    try:
        # Create model
        kwargs = {
            "segment_lengths": segment_lengths,
            "dilation_rates": dilation_rates,
            "ring_size": ring_size,
            "device": device,
            "dtype": torch.float16,
            "enable_memory_pool": enable_memory_pool,
            "use_pattern_cache": enable_pattern_cache,
        }

        if implementation == "ring_v2":
            model = RingDilatedAttentionV2(**kwargs)
        elif implementation == "ring_v3":
            # V3 doesn't support distributed, but we can still test simulated mode
            kwargs["cache_on_gpu"] = True
            model = RingDilatedAttentionV3(**kwargs)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        # Get actual mode
        mode = model.mode

        # Create input tensors
        shape = (batch_size, seq_len, num_heads, head_dim)
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(warmup_steps):
            _ = model(q, k, v)

        torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device) / 1024 / 1024

        # Benchmark
        start_time = time.time()

        for _ in range(benchmark_steps):
            _ = model(q, k, v)

        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate metrics
        time_ms = (end_time - start_time) * 1000 / benchmark_steps

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        memory_mb = peak_memory - start_memory

        total_tokens = batch_size * seq_len
        throughput = total_tokens / (time_ms / 1000)

        return SimpleBenchmarkResult(
            implementation=implementation,
            sequence_length=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            ring_size=ring_size,
            mode=mode,
            gpu_id=gpu_id,
            time_ms=time_ms,
            memory_mb=memory_mb,
            throughput_tokens_per_sec=throughput,
            pattern_cache_enabled=enable_pattern_cache,
            memory_pool_enabled=enable_memory_pool,
        )

    except Exception as e:
        print(f"Error on GPU {gpu_id}: {e}")
        raise
    finally:
        # Cleanup
        if "model" in locals():
            del model
        if "q" in locals():
            del q, k, v
        torch.cuda.empty_cache()


def run_parallel_benchmark(
    gpu_id: int,
    queue: mp.Queue,
    implementation: str,
    seq_len: int,
    batch_size: int,
    ring_size: int,
    enable_pattern_cache: bool,
    enable_memory_pool: bool,
):
    """Worker function for parallel benchmarking."""
    try:
        result = benchmark_on_gpu(
            gpu_id=gpu_id,
            implementation=implementation,
            seq_len=seq_len,
            batch_size=batch_size,
            ring_size=ring_size,
            enable_pattern_cache=enable_pattern_cache,
            enable_memory_pool=enable_memory_pool,
        )
        queue.put(result)
    except Exception as e:
        queue.put(e)


def main():
    parser = argparse.ArgumentParser(
        description="Simple distributed Ring Attention benchmark"
    )
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=["ring_v2", "ring_v3"],
        help="Implementations to benchmark",
    )
    parser.add_argument(
        "--sequence-lengths",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192],
        help="Sequence lengths to test",
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[1], help="Batch sizes to test"
    )
    parser.add_argument(
        "--output-file",
        default="benchmark_results/distributed_ring_simple.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Test both single GPU and simulated multi-GPU
    ring_sizes = [1, 2, 4]  # Test different ring sizes
    results = []

    print("Running benchmarks on available GPUs...")
    print(f"Implementations: {args.implementations}")
    print(f"Sequence lengths: {args.sequence_lengths}")
    print(f"Ring sizes: {ring_sizes}")
    print("=" * 80)

    for seq_len in args.sequence_lengths:
        for batch_size in args.batch_sizes:
            for impl in args.implementations:
                for ring_size in ring_sizes:
                    print(
                        f"\n{impl} - seq_len={seq_len}, batch={batch_size}, ring_size={ring_size}"
                    )

                    # Test different optimization combinations
                    configs = [
                        (False, False, "baseline"),
                        (True, False, "cache"),
                        (True, True, "cache+pool"),
                    ]

                    for pattern_cache, memory_pool, config_name in configs:
                        if ring_size == 1:
                            # Single GPU test
                            try:
                                result = benchmark_on_gpu(
                                    gpu_id=0,
                                    implementation=impl,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    ring_size=ring_size,
                                    enable_pattern_cache=pattern_cache,
                                    enable_memory_pool=memory_pool,
                                )
                                results.append(result)
                                print(
                                    f"  GPU 0 ({config_name}): {result.time_ms:.1f}ms, "
                                    f"{result.memory_mb:.1f}MB, mode={result.mode}"
                                )
                            except Exception as e:
                                print(f"  GPU 0 ({config_name}): Failed - {e}")

                        elif ring_size == 2:
                            # Parallel test on 2 GPUs
                            # Each GPU processes the full sequence with simulated ring
                            processes = []
                            queue = mp.Queue()

                            for gpu_id in range(2):
                                p = mp.Process(
                                    target=run_parallel_benchmark,
                                    args=(
                                        gpu_id,
                                        queue,
                                        impl,
                                        seq_len,
                                        batch_size,
                                        ring_size,
                                        pattern_cache,
                                        memory_pool,
                                    ),
                                )
                                p.start()
                                processes.append(p)

                            # Collect results
                            gpu_results = []
                            for _ in range(2):
                                result = queue.get()
                                if isinstance(result, Exception):
                                    print(f"  Error: {result}")
                                else:
                                    gpu_results.append(result)

                            # Wait for processes
                            for p in processes:
                                p.join()

                            # Report average performance
                            if gpu_results:
                                avg_time = sum(r.time_ms for r in gpu_results) / len(
                                    gpu_results
                                )
                                avg_memory = sum(
                                    r.memory_mb for r in gpu_results
                                ) / len(gpu_results)
                                print(
                                    f"  2 GPUs ({config_name}): {avg_time:.1f}ms avg, "
                                    f"{avg_memory:.1f}MB avg, mode={gpu_results[0].mode}"
                                )
                                results.extend(gpu_results)

                        else:
                            # Single GPU with larger simulated ring
                            try:
                                result = benchmark_on_gpu(
                                    gpu_id=0,
                                    implementation=impl,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    ring_size=ring_size,
                                    enable_pattern_cache=pattern_cache,
                                    enable_memory_pool=memory_pool,
                                )
                                results.append(result)
                                print(
                                    f"  Ring-{ring_size} ({config_name}): {result.time_ms:.1f}ms, "
                                    f"{result.memory_mb:.1f}MB, mode={result.mode}"
                                )
                            except Exception as e:
                                print(
                                    f"  Ring-{ring_size} ({config_name}): Failed - {e}"
                                )

    # Save results
    save_results(results, args.output_file)

    # Generate report
    generate_report(results)


def save_results(results: List[SimpleBenchmarkResult], output_file: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    json_results = []
    for r in results:
        json_results.append(
            {
                "implementation": r.implementation,
                "sequence_length": r.sequence_length,
                "batch_size": r.batch_size,
                "num_heads": r.num_heads,
                "head_dim": r.head_dim,
                "ring_size": r.ring_size,
                "mode": r.mode,
                "gpu_id": r.gpu_id,
                "time_ms": r.time_ms,
                "memory_mb": r.memory_mb,
                "throughput_tokens_per_sec": r.throughput_tokens_per_sec,
                "pattern_cache_enabled": r.pattern_cache_enabled,
                "memory_pool_enabled": r.memory_pool_enabled,
            }
        )

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "results": json_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_file}")


def generate_report(results: List[SimpleBenchmarkResult]):
    """Generate summary report."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    # Group by configuration
    configs = {}
    for r in results:
        key = (
            r.implementation,
            r.sequence_length,
            r.ring_size,
            r.pattern_cache_enabled,
            r.memory_pool_enabled,
        )
        if key not in configs:
            configs[key] = []
        configs[key].append(r)

    # Compare V2 vs V3
    print("\n## V2 vs V3 Performance Comparison")

    for seq_len in sorted(set(r.sequence_length for r in results)):
        print(f"\n### Sequence Length: {seq_len}")

        for ring_size in sorted(set(r.ring_size for r in results)):
            v2_results = [
                r
                for r in results
                if r.implementation == "ring_v2"
                and r.sequence_length == seq_len
                and r.ring_size == ring_size
            ]
            v3_results = [
                r
                for r in results
                if r.implementation == "ring_v3"
                and r.sequence_length == seq_len
                and r.ring_size == ring_size
            ]

            if v2_results and v3_results:
                print(f"\n  Ring Size {ring_size}:")

                # Compare baseline
                v2_baseline = next(
                    (
                        r
                        for r in v2_results
                        if not r.pattern_cache_enabled and not r.memory_pool_enabled
                    ),
                    None,
                )
                v3_baseline = next(
                    (
                        r
                        for r in v3_results
                        if not r.pattern_cache_enabled and not r.memory_pool_enabled
                    ),
                    None,
                )

                if v2_baseline and v3_baseline:
                    speedup = v2_baseline.time_ms / v3_baseline.time_ms
                    print(
                        f"    Baseline: V3 is {speedup:.2f}x "
                        f"{'faster' if speedup > 1 else 'slower'} than V2"
                    )

                # Compare with optimizations
                v2_optimized = next(
                    (
                        r
                        for r in v2_results
                        if r.pattern_cache_enabled and r.memory_pool_enabled
                    ),
                    None,
                )
                v3_optimized = next(
                    (
                        r
                        for r in v3_results
                        if r.pattern_cache_enabled and r.memory_pool_enabled
                    ),
                    None,
                )

                if v2_optimized and v3_optimized:
                    speedup = v2_optimized.time_ms / v3_optimized.time_ms
                    print(
                        f"    Optimized: V3 is {speedup:.2f}x "
                        f"{'faster' if speedup > 1 else 'slower'} than V2"
                    )

    # Memory efficiency with ring size
    print("\n## Memory Efficiency with Ring Size")
    for impl in ["ring_v2", "ring_v3"]:
        impl_results = [r for r in results if r.implementation == impl]
        if not impl_results:
            continue

        print(f"\n### {impl}")
        for seq_len in sorted(set(r.sequence_length for r in impl_results)):
            seq_results = [r for r in impl_results if r.sequence_length == seq_len]
            print(f"\n  Sequence {seq_len}:")

            for ring_size in [1, 2, 4]:
                ring_results = [r for r in seq_results if r.ring_size == ring_size]
                if ring_results:
                    avg_memory = sum(r.memory_mb for r in ring_results) / len(
                        ring_results
                    )
                    print(f"    Ring-{ring_size}: {avg_memory:.1f} MB average")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set
    main()
