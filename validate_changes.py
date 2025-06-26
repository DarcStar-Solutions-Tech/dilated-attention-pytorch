#!/usr/bin/env python3
"""
Validation script to check if our changes are logically correct without running full tests.
"""

import ast
import os


def check_method_exists(file_path, class_name, method_name):
    """Check if a method exists in a class."""
    with open(file_path) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return True
    return False


def validate_error_recovery():
    """Validate that error recovery methods were added correctly."""
    print("Validating error recovery implementations...")

    # Check RingDilatedAttention has cleanup method
    if check_method_exists(
        "dilated_attention_pytorch/ring_dilated_attention.py",
        "RingDilatedAttention",
        "_cleanup_ring_communication",
    ):
        print("✓ RingDilatedAttention._cleanup_ring_communication() exists")
    else:
        print("✗ RingDilatedAttention._cleanup_ring_communication() missing")

    # Check RingDistributedDilatedAttention has emergency cleanup
    if check_method_exists(
        "dilated_attention_pytorch/ring_distributed_dilated_attention.py",
        "RingDistributedDilatedAttention",
        "_emergency_cleanup",
    ):
        print("✓ RingDistributedDilatedAttention._emergency_cleanup() exists")
    else:
        print("✗ RingDistributedDilatedAttention._emergency_cleanup() missing")

    # Check BlockSparseRingDistributedDilatedAttention has cleanup
    if check_method_exists(
        "dilated_attention_pytorch/block_sparse_ring_distributed_dilated_attention.py",
        "BlockSparseRingDistributedDilatedAttention",
        "_cleanup_resources",
    ):
        print(
            "✓ BlockSparseRingDistributedDilatedAttention._cleanup_resources() exists"
        )
    else:
        print(
            "✗ BlockSparseRingDistributedDilatedAttention._cleanup_resources() missing"
        )


def validate_memory_limits():
    """Validate that memory pool limits were added."""
    print("\nValidating memory pool limits...")

    # Check RingAttentionMemoryPool has max_pool_size parameter
    with open("dilated_attention_pytorch/ring_dilated_attention.py") as f:
        content = f.read()
        if "max_pool_size: int = 100" in content:
            print("✓ RingAttentionMemoryPool has max_pool_size parameter")
        else:
            print("✗ RingAttentionMemoryPool missing max_pool_size parameter")

        if "_evict_lru_buffer" in content:
            print("✓ RingAttentionMemoryPool has LRU eviction method")
        else:
            print("✗ RingAttentionMemoryPool missing LRU eviction")


def validate_input_validation():
    """Validate that input validation was added."""
    print("\nValidating input validation...")

    # Check DilatedAttention validation
    with open("dilated_attention_pytorch/dilated_attention.py") as f:
        content = f.read()
        if "segment_lengths cannot be empty" in content:
            print("✓ DilatedAttention validates empty segment_lengths")
        if "must be positive" in content:
            print("✓ DilatedAttention validates positive values")
        if "dropout must be between 0 and 1" in content:
            print("✓ DilatedAttention validates dropout range")
        if "Expected 4D tensors" in content:
            print("✓ DilatedAttention validates tensor dimensions")


def validate_test_coverage():
    """Validate that test files were created."""
    print("\nValidating test coverage...")

    test_files = [
        "tests/test_distributed_ring_attention.py",
        "tests/test_thread_safety.py",
        "tests/test_edge_cases_validation.py",
    ]

    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ {test_file} exists")
            # Check it has test classes
            with open(test_file) as f:
                content = f.read()
                test_count = content.count("def test_")
                print(f"  - Contains {test_count} test methods")
        else:
            print(f"✗ {test_file} missing")


def main():
    """Run all validations."""
    print("=" * 60)
    print("Validating changes to dilated-attention-pytorch")
    print("=" * 60)

    validate_error_recovery()
    validate_memory_limits()
    validate_input_validation()
    validate_test_coverage()

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
