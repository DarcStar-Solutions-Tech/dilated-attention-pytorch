#!/usr/bin/env python3
"""
Analyze ring attention implementations to identify safe removals.

This script checks which ring implementations are:
1. Exported in the main __init__.py
2. Used as legacy imports
3. Imported by other modules
4. Safe to remove
"""

import re
from pathlib import Path


def find_imports_of_module(module_name: str, search_dir: Path) -> list[Path]:
    """Find all files that import a given module."""
    importing_files = []
    pattern = re.compile(
        rf"(from\s+.*\s+import\s+.*{module_name}|import\s+.*{module_name})"
    )

    for py_file in search_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            if pattern.search(content):
                importing_files.append(py_file)
        except Exception:
            pass

    return importing_files


def analyze_ring_implementations():
    """Analyze ring implementations for redundancy."""

    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src" / "dilated_attention_pytorch"
    ring_dir = src_dir / "ring"

    # Classes that are exported in main __init__.py
    exported_classes = {
        "RingDilatedAttentionCorrect",
        "RingDilatedAttentionSDPA",
        "RingDilatedAttentionHilbertGPUOptimized",
    }

    # Classes in legacy imports (ring/__init__.py)
    legacy_classes = {
        "RingDilatedAttentionCorrect",
        "RingAttentionWrapper",
        "RingDilatedAttentionV3",
        "RingDilatedAttentionMemoryEfficient",
        "RingDilatedAttentionSDPA",
        "RingDilatedAttentionFixedSimple",
        "RingDilatedAttentionHilbertCore",
        "RingDilatedAttentionHilbertOptimizedFixed",
        "RingDilatedAttentionHilbertProper",
        "RingDilatedAttentionHilbertGPUOptimized",
    }

    # Map files to their main class
    file_to_class = {
        "base/ring_dilated_attention_correct.py": "RingDilatedAttentionCorrect",
        "base/ring_dilated_attention_v3.py": "RingDilatedAttentionV3",
        "base/ring_dilated_attention_fixed_simple.py": "RingDilatedAttentionFixedSimple",
        "base/ring_dilated_attention_memory_efficient.py": "RingDilatedAttentionMemoryEfficient",
        "base/ring_dilated_attention_sdpa.py": "RingDilatedAttentionSDPA",
        "hilbert/ring_dilated_attention_hilbert_core.py": "RingDilatedAttentionHilbertCore",
        "hilbert/ring_dilated_attention_hilbert_core_fixed.py": "RingDilatedAttentionHilbertCoreFixed",
        "hilbert/ring_dilated_attention_hilbert_gpu_optimized.py": "RingDilatedAttentionHilbertGPUOptimized",
        "hilbert/ring_dilated_attention_hilbert_optimized_correct.py": "RingDilatedAttentionHilbertOptimizedCorrect",
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed.py": "RingDilatedAttentionHilbertOptimizedFixed",
        "hilbert/ring_dilated_attention_hilbert_optimized_fixed_v2.py": "RingDilatedAttentionHilbertOptimizedFixedV2",
        "hilbert/ring_dilated_attention_hilbert_proper.py": "RingDilatedAttentionHilbertProper",
    }

    print("Ring Implementation Analysis")
    print("=" * 80)

    # Analyze each file
    safe_to_remove = []
    must_keep = []

    for rel_path, class_name in file_to_class.items():
        full_path = ring_dir / rel_path
        if not full_path.exists():
            continue

        status_parts = []

        # Check if exported
        if class_name in exported_classes:
            status_parts.append("EXPORTED")
            must_keep.append(rel_path)

        # Check if in legacy imports
        if class_name in legacy_classes:
            status_parts.append("LEGACY")

        # Check who imports this file
        importers = find_imports_of_module(class_name, src_dir)
        external_importers = [
            f for f in importers if f != full_path and "ring/__init__.py" not in str(f)
        ]

        if external_importers:
            status_parts.append(f"IMPORTED({len(external_importers)})")

        # Determine if safe to remove
        if class_name not in exported_classes and not external_importers:
            safe_to_remove.append(rel_path)
            status = "SAFE TO REMOVE"
        else:
            status = "KEEP - " + ", ".join(status_parts)

        print(f"{rel_path:60} {status}")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total analyzed: {len(file_to_class)}")
    print(f"  Must keep (exported): {len(must_keep)}")
    print(f"  Safe to remove: {len(safe_to_remove)}")

    print("\nFiles that MUST be kept (exported in main __init__.py):")
    for f in must_keep:
        print(f"  - {f}")

    print("\nFiles SAFE to remove:")
    for f in safe_to_remove:
        print(f"  - {f}")

    # Additional checks
    print("\nAdditional checks:")

    # Check for all_gather usage (bad pattern)
    print("\nFiles using all_gather (should be removed):")
    for rel_path in file_to_class:
        full_path = ring_dir / rel_path
        if full_path.exists():
            content = full_path.read_text()
            if "all_gather" in content:
                print(f"  - {rel_path} (uses all_gather - breaks O(n/k) memory!)")


if __name__ == "__main__":
    analyze_ring_implementations()
