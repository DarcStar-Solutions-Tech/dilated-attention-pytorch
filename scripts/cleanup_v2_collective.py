#!/usr/bin/env python3
"""
Remove redundant methods from V2 Collective.
"""

import re


def remove_method_from_content(content, method_name):
    """Remove a method definition from the content."""
    # Pattern to match the entire method definition
    # This handles methods with decorators, docstrings, and nested functions
    pattern = rf"(\n\s*)def {method_name}\s*\([^)]*\).*?(?=\n\s*def\s+\w+|(?=\n\s*(@|class))|$)"

    # Remove the method
    content = re.sub(pattern, "", content, flags=re.DOTALL)

    return content


def main():
    file_path = "dilated_attention_pytorch/ring_dilated_attention_v2_collective.py"

    # Methods to remove (confirmed unused)
    methods_to_remove = [
        "_compute_chunk_attention_simple",
        "_combine_chunk_outputs",
        "_apply_dilation_to_tensor",
        "_compute_attention_chunk",
        "_compute_attention_dilated",
    ]

    print("Removing redundant methods from V2 Collective...")
    print("=" * 60)

    # Read the file
    with open(file_path, "r") as f:
        content = f.read()

    original_length = len(content)

    # Remove each method
    for method in methods_to_remove:
        print(f"Removing {method}...")
        content = remove_method_from_content(content, method)

    # Clean up any multiple blank lines
    content = re.sub(r"\n\n\n+", "\n\n", content)

    # Write back
    with open(file_path, "w") as f:
        f.write(content)

    new_length = len(content)

    print(f"\nRemoved {original_length - new_length} characters")
    print("✓ Cleanup complete!")

    # Also remove the standalone factory function if it's not used elsewhere
    print(
        "\nNote: The factory function 'create_ring_dilated_attention_v2_collective' was also found to be unused."
    )
    print("It can be removed if not used externally.")


if __name__ == "__main__":
    main()
