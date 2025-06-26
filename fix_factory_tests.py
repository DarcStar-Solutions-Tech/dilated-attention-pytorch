#!/usr/bin/env python
"""Script to fix factory registration issues in tests."""

import os
import re


def fix_factory_test_file(filepath):
    """Fix factory test file to preserve registrations."""

    with open(filepath) as f:
        content = f.read()

    original_content = content

    # Pattern to find setup_method that clears registries
    setup_pattern = r'(def setup_method\(self\):\s*\n\s*"""[^"]*"""\s*\n)(\s*_ATTENTION_REGISTRY\.clear\(\)\s*\n\s*_MULTIHEAD_REGISTRY\.clear\(\)\s*\n)'

    # Replace with code that saves and restores state
    replacement = r"\1        # Save current registry state\n        self._saved_attention_registry = _ATTENTION_REGISTRY.copy()\n        self._saved_multihead_registry = _MULTIHEAD_REGISTRY.copy()\n        _ATTENTION_REGISTRY.clear()\n        _MULTIHEAD_REGISTRY.clear()\n"

    content = re.sub(setup_pattern, replacement, content)

    # Add teardown_method if not present
    if "def teardown_method" not in content:
        # Find the class that has setup_method
        class_pattern = r"(class\s+\w+.*?:\n(?:.*?\n)*?)(    def setup_method.*?\n(?:.*?\n)*?)(\n    def|\nclass|\Z)"

        def add_teardown(match):
            class_def = match.group(1)
            setup_section = match.group(2)
            next_section = match.group(3)

            teardown = '''
    def teardown_method(self):
        """Restore registry state after test."""
        _ATTENTION_REGISTRY.clear()
        _MULTIHEAD_REGISTRY.clear()
        _ATTENTION_REGISTRY.update(self._saved_attention_registry)
        _MULTIHEAD_REGISTRY.update(self._saved_multihead_registry)
'''
            return class_def + setup_section + teardown + next_section

        content = re.sub(class_pattern, add_teardown, content, flags=re.MULTILINE | re.DOTALL)

    # Ensure implementations are registered at module level
    if "_ensure_implementations_registered()" not in content:
        # Add after imports
        import_section_end = content.rfind("from dilated_attention_pytorch")
        if import_section_end != -1:
            # Find the end of the import section
            lines = content[:import_section_end].split("\n")
            import_end_line = len(lines)

            # Find where to insert
            insert_pos = content.find("\n\n", import_section_end) + 2

            registration_code = """# Ensure implementations are registered for tests
from dilated_attention_pytorch.core.factory import _ensure_implementations_registered
_ensure_implementations_registered()

"""
            content = content[:insert_pos] + registration_code + content[insert_pos:]

    if content != original_content:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False


def main():
    """Fix all test files with factory registration issues."""

    test_files = ["tests/test_core_factory.py", "tests/test_factory_integration.py"]

    fixed_count = 0
    for filepath in test_files:
        if os.path.exists(filepath):
            if fix_factory_test_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filepath}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
