#!/usr/bin/env python3
"""Analyze where Enhanced outperforms other implementations."""

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
    # Sparse patterns (after the fix)
    (2048, "sparse", 2): (2.34, 70.96, 4.51),  # Enhanced improved after fix
    (4096, "sparse", 2): (7.94, 37.76, 10.43),  # Enhanced improved after fix
    (4096, "sparse", 4): (7.50, 121.49, 9.82),  # Enhanced improved after fix
    (8192, "sparse", 2): (30.55, 89.39, 38.71),  # Enhanced improved after fix
    (8192, "sparse", 4): (33.42, 186.78, 44.23),  # Enhanced improved after fix
}

print("=== Where Enhanced Outperforms Others ===\n")

enhanced_wins = []
for config, (unified, optimized, enhanced) in results.items():
    seq_len, pattern, dilation = config
    if unified is None:
        unified = float("inf")

    if enhanced < unified and enhanced < optimized:
        ratio_vs_unified = unified / enhanced if unified != float("inf") else None
        ratio_vs_optimized = optimized / enhanced
        enhanced_wins.append(
            (config, enhanced, unified, optimized, ratio_vs_unified, ratio_vs_optimized)
        )
        print(f"{pattern.upper()} {seq_len} (d={dilation}):")
        print(f"  Enhanced: {enhanced:.2f}ms")
        if ratio_vs_unified:
            print(f"  vs Unified: {unified:.2f}ms ({ratio_vs_unified:.1f}x faster)")
        else:
            print("  vs Unified: Not tested")
        print(f"  vs Optimized: {optimized:.2f}ms ({ratio_vs_optimized:.1f}x faster)\n")

print(f"Total configurations where Enhanced wins: {len(enhanced_wins)}\n")

# Additional analysis
print("=== Key Findings ===\n")

print("1. DENSE PATTERNS:")
for seq_len in [512, 1024, 2048, 4096, 8192, 16384]:
    config = (seq_len, "dense", 1)
    if config in results:
        unified, optimized, enhanced = results[config]
        if unified is None:
            print(f"   {seq_len}: Enhanced WINS (not tested against Unified)")
        elif enhanced < unified and enhanced < optimized:
            print(
                f"   {seq_len}: Enhanced WINS ({enhanced:.2f}ms - {unified / enhanced:.1f}x faster than Unified)"
            )
        elif enhanced < optimized:
            print(f"   {seq_len}: Enhanced beats Optimized but loses to Unified")
        else:
            winner = "Unified" if unified < optimized else "Optimized"
            print(f"   {seq_len}: {winner} wins")

print("\n2. SPARSE PATTERNS (after fix):")
print("   Enhanced is now competitive but Unified still wins")
print("   - 2K d=2: Unified 2.34ms vs Enhanced 4.51ms (1.9x slower)")
print("   - 4K d=4: Unified 7.50ms vs Enhanced 9.82ms (1.3x slower)")
print("   - 8K d=4: Unified 33.42ms vs Enhanced 44.23ms (1.3x slower)")

print("\n=== Summary ===\n")
print("Enhanced excels at:")
print("- Medium to large DENSE sequences (4K-16K tokens)")
print("- Particularly strong at 8K dense: 9.37ms vs 125.10ms (13.4x faster!)")
print("- 16K dense: 36.49ms vs 49.51ms (Unified not tested)")
print("\nEnhanced struggles with:")
print("- Small sequences (<2K) where overhead dominates")
print("- ALL sparse patterns (even after removing PyTorch fallback)")
