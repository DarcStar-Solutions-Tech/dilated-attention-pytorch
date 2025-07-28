#!/usr/bin/env python3
"""
Launch script for distributed Ring Hilbert Attention benchmark.
Handles both single-GPU and multi-GPU scenarios.
"""

import os
import sys
import subprocess
import torch


def launch_benchmark():
    """Launch the benchmark with appropriate configuration."""

    # Check if we have multiple GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
    else:
        print("No GPUs found. Ring Attention requires CUDA.")
        return

    # Path to benchmark script
    benchmark_script = os.path.join(
        os.path.dirname(__file__),
        "..",
        "benchmarks",
        "benchmark_ring_hilbert_attention_single.py",
    )

    if num_gpus > 1:
        # Multi-GPU: use torchrun
        print(f"\nLaunching distributed benchmark on {num_gpus} GPUs...")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node",
            str(num_gpus),
            "--use_env",
            benchmark_script,
        ]
    else:
        # Single GPU: run directly but with mocked ring behavior
        print("\nLaunching single-GPU benchmark (simulated ring)...")
        cmd = [sys.executable, benchmark_script, "--single-gpu"]

    # Run the benchmark
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    launch_benchmark()
