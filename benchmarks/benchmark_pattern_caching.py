"""
Benchmark script to measure performance improvements from pattern caching.

This script compares the performance of dilated attention with and without
pattern caching enabled, measuring both memory usage and execution time.
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

# Add parent directory to path to import benchmarks module
sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch import DilatedAttention, ImprovedDilatedAttention
from dilated_attention_pytorch.core import clear_global_cache, get_global_pattern_cache
from benchmarks.core import BenchmarkOutputManager


def measure_pattern_cache_overhead(
    attention_module,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    device: str,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> Dict[str, float]:
    """Measure the overhead of pattern caching."""

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = attention_module(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Clear pattern cache
    clear_global_cache()
    cache = get_global_pattern_cache()

    # Measure with cold cache (first run)
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    _ = attention_module(q, k, v)
    if device == "cuda":
        torch.cuda.synchronize()
    cold_cache_time = time.perf_counter() - start_time

    _ = cache.get_stats()

    if device == "cuda":
        cold_cache_memory = torch.cuda.max_memory_allocated()
    else:
        cold_cache_memory = 0

    # Measure with warm cache (subsequent runs)
    warm_cache_times = []
    for _ in range(num_iterations):
        if device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        _ = attention_module(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        warm_cache_times.append(time.perf_counter() - start_time)

    warm_cache_stats = cache.get_stats()
    avg_warm_cache_time = sum(warm_cache_times) / len(warm_cache_times)

    return {
        "cold_cache_time": cold_cache_time,
        "warm_cache_time": avg_warm_cache_time,
        "speedup": cold_cache_time / avg_warm_cache_time,
        "cache_size": warm_cache_stats["size"],
        "cache_hits": warm_cache_stats["hits"],
        "cache_misses": warm_cache_stats["misses"],
        "cache_hit_rate": warm_cache_stats["hit_rate"],
        "cold_cache_memory": cold_cache_memory,
    }


def benchmark_attention_configurations(
    device: str = "cuda",
    batch_sizes: List[int] = [1, 2, 4],
    seq_lengths: List[int] = [1024, 2048, 4096, 8192],
    num_heads: List[int] = [8, 16],
    head_dim: int = 64,
) -> Dict:
    """Benchmark various attention configurations."""

    results = {}

    # Test configurations
    configs = [
        {
            "name": "Small",
            "segment_lengths": [256, 512],
            "dilation_rates": [1, 2],
        },
        {
            "name": "Medium",
            "segment_lengths": [512, 1024, 2048],
            "dilation_rates": [1, 2, 4],
        },
        {
            "name": "Large",
            "segment_lengths": [1024, 2048, 4096],
            "dilation_rates": [1, 2, 4],
        },
    ]

    for config in configs:
        config_results = {}

        for AttentionClass in [DilatedAttention, ImprovedDilatedAttention]:
            class_name = AttentionClass.__name__
            config_results[class_name] = {}

            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    # Skip if sequence length is too small for config
                    if seq_len < max(config["segment_lengths"]):
                        continue

                    for num_head in num_heads:
                        key = f"B{batch_size}_S{seq_len}_H{num_head}"

                        try:
                            # Create attention module
                            attention = AttentionClass(
                                segment_lengths=config["segment_lengths"],
                                dilation_rates=config["dilation_rates"],
                            )
                            attention = attention.to(device)
                            attention.eval()

                            # Measure performance
                            metrics = measure_pattern_cache_overhead(
                                attention,
                                batch_size,
                                seq_len,
                                num_head,
                                head_dim,
                                device,
                            )

                            config_results[class_name][key] = metrics

                            print(
                                f"{config['name']} - {class_name} - {key}: "
                                f"Speedup: {metrics['speedup']:.2f}x, "
                                f"Cache hit rate: {metrics['cache_hit_rate']:.1%}"
                            )

                        except Exception as e:
                            print(f"Error with {class_name} - {key}: {e}")
                            config_results[class_name][key] = {"error": str(e)}

        results[config["name"]] = config_results

    return results


def create_summary_report(results: Dict) -> Dict:
    """Create a summary report of the benchmark results."""

    summary = {
        "overall_metrics": {},
        "by_attention_type": {},
        "by_configuration": {},
        "recommendations": [],
    }

    # Calculate overall metrics
    all_speedups = []
    all_hit_rates = []

    for config_name, config_results in results.items():
        for attention_type, attention_results in config_results.items():
            if attention_type not in summary["by_attention_type"]:
                summary["by_attention_type"][attention_type] = {
                    "speedups": [],
                    "hit_rates": [],
                }

            for key, metrics in attention_results.items():
                if "speedup" in metrics:
                    speedup = metrics["speedup"]
                    hit_rate = metrics["cache_hit_rate"]

                    all_speedups.append(speedup)
                    all_hit_rates.append(hit_rate)

                    summary["by_attention_type"][attention_type]["speedups"].append(
                        speedup
                    )
                    summary["by_attention_type"][attention_type]["hit_rates"].append(
                        hit_rate
                    )

    # Overall metrics
    if all_speedups:
        summary["overall_metrics"] = {
            "avg_speedup": sum(all_speedups) / len(all_speedups),
            "min_speedup": min(all_speedups),
            "max_speedup": max(all_speedups),
            "avg_hit_rate": sum(all_hit_rates) / len(all_hit_rates),
        }

    # Per-attention-type metrics
    for attention_type, data in summary["by_attention_type"].items():
        if data["speedups"]:
            summary["by_attention_type"][attention_type] = {
                "avg_speedup": sum(data["speedups"]) / len(data["speedups"]),
                "avg_hit_rate": sum(data["hit_rates"]) / len(data["hit_rates"]),
                "num_configs": len(data["speedups"]),
            }

    # Generate recommendations
    if summary["overall_metrics"]:
        avg_speedup = summary["overall_metrics"]["avg_speedup"]
        if avg_speedup > 1.5:
            summary["recommendations"].append(
                "Pattern caching provides significant speedup (>1.5x). "
                "Recommend enabling for production use."
            )
        elif avg_speedup > 1.2:
            summary["recommendations"].append(
                "Pattern caching provides moderate speedup (>1.2x). "
                "Consider enabling for repeated forward passes."
            )
        else:
            summary["recommendations"].append(
                "Pattern caching provides minimal speedup. "
                "May not be worth the memory overhead for single-pass inference."
            )

        avg_hit_rate = summary["overall_metrics"]["avg_hit_rate"]
        if avg_hit_rate > 0.9:
            summary["recommendations"].append(
                "Excellent cache hit rate (>90%). Pattern reuse is very effective."
            )
        elif avg_hit_rate > 0.7:
            summary["recommendations"].append(
                "Good cache hit rate (>70%). Consider increasing cache size for better performance."
            )
        else:
            summary["recommendations"].append(
                "Low cache hit rate (<70%). May indicate too many unique patterns or small cache size."
            )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pattern caching performance"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--seq-lengths", type=int, nargs="+", default=[1024, 2048, 4096]
    )
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--output-format", choices=["json", "markdown", "both"], default="both"
    )

    args = parser.parse_args()

    print(f"Running pattern cache benchmarks on {args.device}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Number of heads: {args.num_heads}")
    print(f"Head dimension: {args.head_dim}")
    print()

    # Run benchmarks
    results = benchmark_attention_configurations(
        device=args.device,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    # Create summary
    summary = create_summary_report(results)

    # Create output manager
    output_manager = BenchmarkOutputManager(
        benchmark_type="pattern-caching",
        parameters={
            "device": args.device,
            "batch_sizes": args.batch_sizes,
            "seq_lengths": args.seq_lengths,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
        },
    )

    # Add results
    output_manager.add_result("detailed_results", results)
    output_manager.add_result("summary", summary)

    # Save results
    output_paths = output_manager.save_results()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"Overall average speedup: {summary['overall_metrics'].get('avg_speedup', 0):.2f}x"
    )
    print(
        f"Overall cache hit rate: {summary['overall_metrics'].get('avg_hit_rate', 0):.1%}"
    )
    print("\nRecommendations:")
    for rec in summary["recommendations"]:
        print(f"- {rec}")

    print(f"\nResults saved to: {output_paths['markdown']}")


if __name__ == "__main__":
    main()
