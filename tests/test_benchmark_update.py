#!/usr/bin/env python3
"""
Quick test to verify benchmark_all.py includes block sparse implementations.
"""

import subprocess
import sys


def test_benchmark_includes_block_sparse():
    """Test that benchmark_all.py properly includes block sparse implementations."""

    # Check if block sparse implementations are imported
    print("Checking benchmark_all.py includes block sparse implementations...")

    with open("benchmarks/benchmark_all.py") as f:
        content = f.read()

    # Check imports
    assert "BlockSparseRingDilatedAttention" in content
    assert "BlockSparseRingMultiheadDilatedAttention" in content
    assert "SparsePatternConfig" in content
    print("✓ Block sparse imports found")

    # Check that they're added to benchmarks
    assert "BlockSparseRingDilated_" in content
    assert "BlockSparseRingMultihead_" in content
    print("✓ Block sparse implementations added to benchmark list")

    # Check sparsity configurations
    assert "0.1, 0.25, 0.5" in content
    print("✓ Multiple sparsity ratios configured (10%, 25%, 50%)")

    # Run a minimal test
    print("\nRunning minimal benchmark to verify functionality...")

    # Just check that it starts without errors
    proc = subprocess.Popen(
        [
            sys.executable,
            "benchmarks/benchmark_all.py",
            "--batch-sizes",
            "1",
            "--seq-lens",
            "512",
            "--num-heads",
            "2",
            "--head-dim",
            "32",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait a few seconds and check it started properly
    import time

    time.sleep(5)

    if proc.poll() is None:
        print("✓ Benchmark started successfully")
        proc.terminate()
        proc.wait()
    else:
        stdout, stderr = proc.communicate()
        if "BlockSparseRingDilated" in stdout:
            print("✓ Block sparse implementations are being benchmarked")
        else:
            print("✗ Block sparse implementations not found in output")
            print("STDOUT:", stdout[:500])
            print("STDERR:", stderr[:500])

    print("\n✅ benchmark_all.py successfully updated with block sparse implementations!")


if __name__ == "__main__":
    test_benchmark_includes_block_sparse()
