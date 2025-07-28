#!/usr/bin/env python3
"""
Analyze V2 Collective for redundant methods.
"""

import re


def analyze_method_usage(file_path):
    """Analyze which methods are actually used in the file."""
    with open(file_path, "r") as f:
        content = f.read()

    # Find all method definitions
    method_pattern = r"def\s+(\w+)\s*\("
    methods = re.findall(method_pattern, content)

    # Exclude __init__ and other special methods
    methods = [m for m in methods if not m.startswith("__")]

    # Track usage of each method
    usage_count = {}
    for method in methods:
        # Count how many times each method is called (excluding its definition)
        # Look for self.method_name( or method_name(
        call_pattern = rf"(?:self\.)?{method}\s*\("
        calls = re.findall(call_pattern, content)
        # Subtract 1 for the definition itself
        usage_count[method] = len(calls) - 1

    # Categorize methods
    unused_methods = []
    rarely_used_methods = []
    used_methods = []

    for method, count in usage_count.items():
        if count == 0:
            unused_methods.append(method)
        elif count == 1:
            rarely_used_methods.append(method)
        else:
            used_methods.append((method, count))

    return unused_methods, rarely_used_methods, used_methods


def main():
    file_path = "dilated_attention_pytorch/ring_dilated_attention_v2_collective.py"

    print("Analyzing V2 Collective for redundant methods...")
    print("=" * 60)

    unused, rarely_used, used = analyze_method_usage(file_path)

    print("\nUNUSED METHODS (can be removed):")
    print("-" * 40)
    for method in unused:
        print(f"  - {method}")

    print("\nRARELY USED METHODS (check if needed):")
    print("-" * 40)
    for method in rarely_used:
        print(f"  - {method}")

    print("\nFREQUENTLY USED METHODS:")
    print("-" * 40)
    for method, count in sorted(used, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {method} ({count} calls)")

    # Now let's check specific potentially redundant methods
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS OF POTENTIALLY REDUNDANT METHODS:")
    print("=" * 60)

    with open(file_path, "r") as f:
        content = f.read()

    # Methods that seem redundant based on manual inspection
    suspect_methods = [
        "_compute_chunk_attention_simple",
        "_combine_chunk_outputs",
        "_apply_dilation_to_tensor",
        "_compute_attention_chunk",
        "_compute_attention_dilated",
    ]

    for method in suspect_methods:
        # Find if method is called anywhere
        pattern = rf"self\.{method}\s*\("
        matches = re.findall(pattern, content)

        print(f"\n{method}:")
        if not matches:
            print("  ✗ NOT USED - Can be removed")
            # Find the method definition to understand what it does
            def_pattern = rf"def {method}.*?(?=\n    def|\n\nclass|\Z)"
            def_match = re.search(def_pattern, content, re.DOTALL)
            if def_match:
                lines = def_match.group(0).split("\n")
                docstring = None
                for i, line in enumerate(lines):
                    if '"""' in line and i > 0:
                        docstring = line.strip()
                        break
                if docstring:
                    print(f"  Description: {docstring}")
        else:
            print(f"  ✓ Used {len(matches)} times")


if __name__ == "__main__":
    main()
