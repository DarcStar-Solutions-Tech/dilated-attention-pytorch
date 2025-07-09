#!/usr/bin/env python3
"""
Compare multi-GPU results from original vs optimized implementations.
"""

import json
import glob


def analyze_results():
    # Find original results
    original_files = glob.glob("benchmarks/hybrid_results_rank*_2gpu_*.json")
    optimized_files = glob.glob("benchmarks/hybrid_optimized_results_rank*_2gpu_*.json")

    print("=== Multi-GPU Performance Comparison ===\n")

    # Load original results
    print("Original Implementation Results:")
    print("-" * 40)

    original_data = []
    for f in sorted(original_files):
        with open(f, "r") as file:
            data = json.load(file)
            if data.get("results"):
                print(f"\nRank {data['rank']}:")
                for r in data["results"]:
                    print(
                        f"  Seq {r['seq_len']:,}: {r['time_ms']:.1f}ms, "
                        f"{r['memory_mb']:.1f}MB, {r['throughput']:,.0f} tok/s"
                    )
                original_data.append(data)

    print("\nOptimized Implementation Results:")
    print("-" * 40)

    optimized_data = []
    for f in sorted(optimized_files):
        with open(f, "r") as file:
            data = json.load(file)
            if data.get("results"):
                print(f"\nRank {data['rank']}:")
                for r in data["results"]:
                    print(
                        f"  Seq {r['seq_len']:,}: {r['time_ms']:.1f}ms, "
                        f"{r['memory_mb']:.1f}MB, {r['throughput']:,.0f} tok/s"
                    )
                optimized_data.append(data)

    # If we have both, compare
    if original_data and not optimized_data:
        print("\n⚠️  No optimized multi-GPU results found yet")
        print("The optimized implementation encountered errors in multi-GPU mode")

    # Analyze original results
    if original_data and original_data[0]["results"]:
        print("\n" + "=" * 60)
        print("Original Multi-GPU Memory Scaling Analysis:")
        print("=" * 60)

        results = original_data[0]["results"]
        for r in results:
            print(f"Seq {r['seq_len']:,}: {r['memory_per_token_kb']:.2f} KB/token")

        if len(results) >= 2:
            first_mem = results[0]["memory_per_token_kb"]
            last_mem = results[-1]["memory_per_token_kb"]
            ratio = last_mem / first_mem
            print(f"\nMemory ratio: {ratio:.2f}")

            if ratio < 1.3:
                print("✅ Excellent O(n/p) memory scaling")
            else:
                print("⚠️  Memory scaling could be improved")


if __name__ == "__main__":
    analyze_results()
