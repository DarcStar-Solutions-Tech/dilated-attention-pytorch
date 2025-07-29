#!/usr/bin/env python3
"""
Analyze kernel redundancy to determine which can be safely removed.
"""

import sys
from pathlib import Path

sys.path.append("..")

# Performance data from benchmarks
kernel_performance = {
    "hilbert_attention_unified_optimized_enhanced.py": {
        "time_1k": 1.18,
        "class": "UnifiedHilbertAttentionOptimizedEnhanced",
        "features": [
            "unified",
            "optimized",
            "enhanced",
            "8k_optimization",
            "multi_row",
            "strided_sparse",
        ],
        "wins": 13,  # Out of 18 tests
    },
    "hilbert_attention_simple.py": {
        "time_1k": 1.40,
        "class": "HilbertAttentionCore",  # Note: has 2 classes
        "features": ["simple", "basic"],
        "wins": 0,
    },
    "hilbert_attention_unified_optimized.py": {
        "time_1k": 1.62,
        "class": "UnifiedHilbertAttentionOptimized",
        "features": ["unified", "optimized"],
        "wins": 3,  # XLarge, Sparse d=8, Max dim
    },
    "hilbert_attention_enhanced.py": {
        "time_1k": 2.44,
        "class": "HilbertAttentionEnhanced",
        "features": ["enhanced"],
        "wins": 0,
    },
    "hilbert_attention.py": {
        "time_1k": 2.55,
        "class": "HilbertAttention",
        "features": ["basic"],
        "wins": 0,
    },
    "hilbert_attention_unified.py": {
        "time_1k": 3.44,
        "class": "UnifiedHilbertAttention",
        "features": ["unified"],
        "wins": 3,  # Large heads, Sparse d=4, Huge seg
    },
    "hilbert_attention_core.py": {
        "time_1k": 3.49,
        "class": "HilbertAttentionCore",
        "features": ["core", "autograd"],
        "wins": 0,
    },
}


def analyze_redundancy():
    print("=== Kernel Redundancy Analysis ===\n")

    # Sort by performance
    sorted_kernels = sorted(kernel_performance.items(), key=lambda x: x[1]["time_1k"])

    print("Performance Ranking (1K tokens):")
    for i, (kernel, info) in enumerate(sorted_kernels):
        print(
            f"{i + 1}. {kernel:<45} - {info['time_1k']:>5.2f}ms - Wins: {info['wins']}"
        )

    print("\n\nFeature Analysis:")

    # Identify feature coverage
    all_features = set()
    for info in kernel_performance.values():
        all_features.update(info["features"])

    print(f"Total unique features: {all_features}")

    # Check feature coverage by top performers
    print("\nFeature coverage by top 3 performers:")
    top_3_features = set()
    for kernel, info in sorted_kernels[:3]:
        print(f"  {kernel}: {info['features']}")
        top_3_features.update(info["features"])

    print(f"\nFeatures covered by top 3: {top_3_features}")
    print(f"Features NOT covered by top 3: {all_features - top_3_features}")

    # Removal recommendations
    print("\n\n=== Removal Recommendations ===\n")

    removals = []
    keep = []

    for kernel, info in sorted_kernels:
        if info["wins"] == 0 and info["time_1k"] > 2.0:
            removals.append((kernel, "No performance wins and slower than 2ms"))
        elif kernel == "hilbert_attention_simple.py":
            removals.append((kernel, "Redundant with unified_optimized_enhanced"))
        elif kernel == "hilbert_attention_enhanced.py":
            removals.append(
                (kernel, "Features integrated into unified_optimized_enhanced")
            )
        elif kernel == "hilbert_attention.py":
            removals.append((kernel, "Basic implementation superseded by unified"))
        elif kernel == "hilbert_attention_core.py":
            keep.append((kernel, "Keep for backward compatibility / reference"))
        else:
            keep.append((kernel, "Unique performance characteristics"))

    print("REMOVE:")
    for kernel, reason in removals:
        print(f"  ✗ {kernel:<45} - {reason}")

    print("\nKEEP:")
    for kernel, reason in keep:
        print(f"  ✓ {kernel:<45} - {reason}")

    # Check imports and dependencies
    print("\n\n=== Dependency Check ===")

    # Search for imports of these modules
    src_dir = Path("../src/dilated_attention_pytorch")
    test_dir = Path("../tests")

    for kernel, _ in removals:
        kernel_name = kernel.replace(".py", "")
        print(f"\nChecking usage of {kernel_name}:")

        # Check in source files
        src_imports = list(src_dir.rglob("*.py"))
        test_imports = list(test_dir.rglob("*.py")) if test_dir.exists() else []

        found_imports = False
        for file_list, dir_name in [(src_imports, "src"), (test_imports, "tests")]:
            for file in file_list:
                if file.name == kernel:
                    continue
                try:
                    content = file.read_text()
                    if kernel_name in content:
                        print(
                            f"  Found in {dir_name}/{file.relative_to(file.parent.parent)}"
                        )
                        found_imports = True
                except:
                    pass

        if not found_imports:
            print("  No imports found - safe to remove")

    print("\n\n=== Summary ===")
    print(f"Total kernels: {len(kernel_performance)}")
    print(f"Recommended for removal: {len(removals)}")
    print(f"Recommended to keep: {len(keep)}")

    # Final recommendation
    print("\n\n=== Final Recommendation ===")
    print("\nKernels to KEEP:")
    print(
        "1. hilbert_attention_unified_optimized_enhanced.py - Best overall performance"
    )
    print("2. hilbert_attention_unified_optimized.py - Best for very large sequences")
    print("3. hilbert_attention_unified.py - Stable baseline, wins on some edge cases")
    print("4. hilbert_attention_core.py - Reference implementation")

    print("\nKernels to REMOVE:")
    print("1. hilbert_attention_simple.py - Redundant")
    print(
        "2. hilbert_attention_enhanced.py - Features merged into unified_optimized_enhanced"
    )
    print("3. hilbert_attention.py - Basic implementation superseded")

    return removals


if __name__ == "__main__":
    removals = analyze_redundancy()
