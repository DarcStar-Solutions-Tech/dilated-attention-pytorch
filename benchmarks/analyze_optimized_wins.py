#!/usr/bin/env python3
"""Analyze where Optimized outperforms other implementations."""

# Benchmark results
results = {
    # Format: (seq_len, pattern, dilation): (unified_ms, optimized_ms, enhanced_ms)
    # Dense patterns
    (512, "dense", 1): (0.87, 24.47, 2.09),
    (1024, "dense", 1): (3.45, 37.66, 2.07),
    (2048, "dense", 1): (7.37, 24.69, 2.77),
    (4096, "dense", 1): (33.64, 13.34, 4.89),
    (8192, "dense", 1): (125.10, 25.46, 9.37),
    (16384, "dense", 1): (None, 49.51, 36.49),
    # Sparse patterns (after Enhanced fix)
    (2048, "sparse", 2): (2.34, 70.96, 4.51),
    (4096, "sparse", 2): (7.94, 37.76, 10.43),
    (4096, "sparse", 4): (7.50, 121.49, 9.82),
    (8192, "sparse", 2): (30.55, 89.39, 38.71),
    (8192, "sparse", 4): (33.42, 186.78, 44.23),
}

print("=== Where Optimized Outperforms Others ===\n")

optimized_wins = []
optimized_beats_one = []

for config, (unified, optimized, enhanced) in results.items():
    seq_len, pattern, dilation = config
    if unified is None:
        unified = float("inf")

    # Check if Optimized wins over both
    if optimized < unified and optimized < enhanced:
        ratio_vs_unified = unified / optimized if unified != float("inf") else None
        ratio_vs_enhanced = enhanced / optimized
        optimized_wins.append(
            (config, optimized, unified, enhanced, ratio_vs_unified, ratio_vs_enhanced)
        )
        print(f"{pattern.upper()} {seq_len} (d={dilation}):")
        print(f"  Optimized: {optimized:.2f}ms WINS OVER BOTH")
        if ratio_vs_unified:
            print(f"  vs Unified: {unified:.2f}ms ({ratio_vs_unified:.1f}x faster)")
        else:
            print("  vs Unified: Not tested")
        print(f"  vs Enhanced: {enhanced:.2f}ms ({ratio_vs_enhanced:.1f}x faster)\n")

    # Check if Optimized beats at least one
    elif optimized < unified or optimized < enhanced:
        optimized_beats_one.append((config, optimized, unified, enhanced))

print(f"Total configurations where Optimized wins over BOTH: {len(optimized_wins)}\n")

if optimized_beats_one:
    print("=== Where Optimized Beats At Least One ===\n")
    for config, optimized, unified, enhanced in optimized_beats_one:
        seq_len, pattern, dilation = config
        print(f"{pattern.upper()} {seq_len} (d={dilation}):")
        print(f"  Optimized: {optimized:.2f}ms")
        if optimized < unified:
            print(
                f"  Beats Unified: {unified:.2f}ms ({unified / optimized:.1f}x faster)"
            )
        else:
            print(f"  Loses to Unified: {unified:.2f}ms")
        if optimized < enhanced:
            print(
                f"  Beats Enhanced: {enhanced:.2f}ms ({enhanced / optimized:.1f}x faster)"
            )
        else:
            print(f"  Loses to Enhanced: {enhanced:.2f}ms")
        print()

print("\n=== Analysis by Pattern ===\n")

# Dense analysis
print("DENSE PATTERNS:")
dense_performance = []
for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    config = (seq_len, "dense", 1)
    if config in results:
        unified, optimized, enhanced = results[config]
        if unified is None:
            unified = float("inf")

        # Determine ranking
        times = [("Unified", unified), ("Optimized", optimized), ("Enhanced", enhanced)]
        times.sort(key=lambda x: x[1])

        rank = None
        for i, (name, _) in enumerate(times):
            if name == "Optimized":
                rank = i + 1
                break

        print(f"  {seq_len}: Optimized ranks #{rank} - {optimized:.2f}ms", end="")
        if rank == 1:
            print(" *** WINNER ***")
        else:
            winner_name, winner_time = times[0]
            print(f" (loses to {winner_name} at {winner_time:.2f}ms)")

print("\nSPARSE PATTERNS:")
for config, (unified, optimized, enhanced) in results.items():
    seq_len, pattern, dilation = config
    if pattern == "sparse":
        times = [("Unified", unified), ("Optimized", optimized), ("Enhanced", enhanced)]
        times.sort(key=lambda x: x[1])

        rank = None
        for i, (name, _) in enumerate(times):
            if name == "Optimized":
                rank = i + 1
                break

        print(
            f"  {seq_len} d={dilation}: Optimized ranks #{rank} - {optimized:.2f}ms",
            end="",
        )
        if rank == 1:
            print(" *** WINNER ***")
        else:
            winner_name, winner_time = times[0]
            print(f" (loses to {winner_name} at {winner_time:.2f}ms)")

print("\n=== Summary ===\n")
print("Optimized implementation:")
print("- NEVER wins over both competitors simultaneously")
print("- Only competitive at Dense 4K and 8K (beats Unified but loses to Enhanced)")
print("- Performs poorly on ALL sparse patterns (2nd or 3rd place)")
print("- Generally the worst performer across most configurations")
print("\nWhy Optimized struggles:")
print('- Too many "optimizations" that add overhead without benefit')
print("- Not as simple/efficient as Unified for small sequences")
print("- Not as sophisticated as Enhanced for large sequences")
print("- Poorly suited for sparse patterns despite attempted optimizations")
