#!/usr/bin/env python3
"""Script to run all block-sparse related tests and provide a summary."""

import subprocess
import sys
from pathlib import Path


def run_test(test_path, description):
    """Run a test and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Test: {test_path}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Extract summary from output
        lines = result.stdout.split("\n")
        summary_line = None
        for line in lines:
            if "passed" in line or "failed" in line or "skipped" in line:
                if "==" in line:
                    summary_line = line
                    break

        if result.returncode == 0:
            print(f"✅ PASSED: {summary_line if summary_line else 'All tests passed'}")
        else:
            print(f"❌ FAILED: {summary_line if summary_line else 'Some tests failed'}")
            # Print failure details
            if result.stderr:
                print("\nError output:")
                print(result.stderr[:500])  # First 500 chars of error

        return result.returncode == 0, summary_line

    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Test took too long to run")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False, str(e)


def main():
    """Run all block-sparse tests."""
    project_root = Path(__file__).parent.parent

    # Define all block-sparse related tests
    tests = [
        (
            "tests/sparse/test_block_sparse_attention.py",
            "Main block-sparse attention tests",
        ),
        ("tests/sparse/test_block_sparse_adaptive.py", "Block-sparse adaptive tests"),
        (
            "tests/sparse/test_block_sparse_multihead_memory.py",
            "Block-sparse multihead memory tests",
        ),
        (
            "tests/sparse/test_distributed_block_sparse_simple.py",
            "Distributed block-sparse simple tests",
        ),
        (
            "tests/ring/hilbert/test_post_pattern_hilbert.py",
            "Hilbert post-pattern tests",
        ),
        (
            "tests/ring/test_standardized_ring_attention.py::TestBlockSparseRingAttention",
            "Standardized block-sparse ring tests",
        ),
        (
            "tests/ring/distributed/test_distributed_ring_attention.py::TestBlockSparseDistributed::test_distributed_sparse_config_validation",
            "Distributed block-sparse config test",
        ),
        (
            "tests/ring/distributed/test_distributed_ring_attention.py::TestBlockSparseDistributed::test_forward_error_cleanup",
            "Distributed block-sparse error cleanup test",
        ),
    ]

    # Run all tests
    results = []
    for test_path, description in tests:
        full_path = (
            str(project_root / test_path)
            if not test_path.startswith("tests/")
            else test_path
        )
        success, summary = run_test(full_path, description)
        results.append((description, success, summary))

    # Print summary
    print("\n" + "=" * 80)
    print("BLOCK-SPARSE TEST SUMMARY")
    print("=" * 80)

    total_tests = len(results)
    passed_tests = sum(1 for _, success, _ in results if success)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal test suites: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

    print("\nDetailed Results:")
    for description, success, summary in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"\n{status}: {description}")
        if summary:
            print(f"   {summary}")

    # List active block-sparse implementations
    print("\n" + "=" * 80)
    print("ACTIVE BLOCK-SPARSE IMPLEMENTATIONS")
    print("=" * 80)

    implementations = [
        "BlockSparseRingDilatedAttention - Core block-sparse ring attention with multiple pattern types",
        "BlockSparseRingDilatedAttentionFixed - Fixed version with standardized API wrapper",
        "BlockSparseRingMultiheadDilatedAttention - Drop-in replacement for nn.MultiheadAttention",
        "BlockSparseRingDistributedDilatedAttention - Enterprise distributed implementation",
        "BlockSparseAdaptive - Content-adaptive sparsity patterns that learn optimal attention",
        "BlockSparseAdaptiveFixed - Fixed API wrapper for BlockSparseAdaptive",
        "BlockSparseRingDilatedAttentionHilbertPostPattern - Hilbert curve optimization (up to 2.53x speedup)",
    ]

    for impl in implementations:
        print(f"• {impl}")

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
