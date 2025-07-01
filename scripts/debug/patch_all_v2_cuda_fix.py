#!/usr/bin/env python3
"""
Apply CUDA illegal memory access fix to all affected v2 implementations.
"""

import os
import re


def fix_file(filepath):
    """Apply the contiguous fix to a file."""
    print(f"\nProcessing {filepath}...")

    with open(filepath, "r") as f:
        content = f.read()

    # Pattern to find and replace
    pattern = r"(\s+)(k_dilated\[.*?\] = k\[.*?\]\.index_select\(1, dilated_indices\))\n(\s+)(v_dilated\[.*?\] = v\[.*?\]\.index_select\(1, dilated_indices\))"

    # Check if pattern exists
    if re.search(pattern, content):
        # Extract the head slice pattern
        match = re.search(r"k\[(.*?)\]\.index_select", content)
        if match:
            slice_pattern = match.group(1)

            # Create replacement
            replacement = rf"""\1# Fix: Make slices contiguous before index_select to avoid CUDA errors
\1k_heads = k[{slice_pattern}].contiguous()
\1v_heads = v[{slice_pattern}].contiguous()
\1k_dilated[{slice_pattern}] = k_heads.index_select(1, dilated_indices)
\3v_dilated[{slice_pattern}] = v_heads.index_select(1, dilated_indices)"""

            # Apply replacement
            new_content = re.sub(pattern, replacement, content)

            if new_content != content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"  ✅ Fixed {filepath}")
                return True
            else:
                print(f"  ⚠️  No changes made to {filepath}")
                return False
    else:
        print(f"  ℹ️  Pattern not found in {filepath}")
        return False


def main():
    """Apply fixes to all affected files."""
    print("Applying CUDA Illegal Memory Access Fix to All V2 Implementations")
    print("=" * 70)

    # Files that need fixing
    files_to_fix = [
        "dilated_attention_pytorch/ring_dilated_attention_v2_fsdp.py",
        "dilated_attention_pytorch/ring_dilated_attention_v2_robust.py",
    ]

    # Files already fixed
    files_already_fixed = [
        "dilated_attention_pytorch/ring_dilated_attention_v2_fairscale.py",
        "dilated_attention_pytorch/ring_dilated_attention_v2_deepspeed.py",
    ]

    # Files that use a different pattern (don't need fixing)
    files_different_pattern = [
        "dilated_attention_pytorch/ring_dilated_attention_v2_collective.py",
        "dilated_attention_pytorch/ring_dilated_attention_v2_fixed.py",
    ]

    print("\nFiles to fix:")
    for f in files_to_fix:
        print(f"  - {f}")

    print("\nFiles already fixed:")
    for f in files_already_fixed:
        print(f"  ✅ {f}")

    print("\nFiles using different pattern (no fix needed):")
    for f in files_different_pattern:
        print(f"  ℹ️  {f}")

    # Apply fixes
    print("\nApplying fixes...")
    fixed_count = 0

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"  ❌ File not found: {filepath}")

    print("\n" + "=" * 70)
    print(f"Summary: Fixed {fixed_count} out of {len(files_to_fix)} files")

    if fixed_count > 0:
        print("\n✅ Fixes applied successfully!")
        print("\nTo verify the fixes, run:")
        print("  python scripts/debug/verify_cuda_fix.py")
        print("  torchrun --nproc_per_node=2 scripts/debug/verify_cuda_fix.py")


if __name__ == "__main__":
    main()
