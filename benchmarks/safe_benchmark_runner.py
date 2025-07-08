#!/usr/bin/env python3
"""
Safe benchmark runner that prevents system lockups.

This script wraps benchmark execution with memory safety checks and progressive testing.
"""

import sys
import argparse
import importlib.util
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.core.utils.safety import (
    SafeBenchmarkRunner,
    SafetyConfig,
    MemorySafetyChecker,
)


def load_benchmark_module(benchmark_path: str):
    """Dynamically load a benchmark module."""
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    spec = importlib.util.spec_from_file_location("benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmarks with safety limits to prevent lockups"
    )
    parser.add_argument("benchmark", help="Path to benchmark script or module name")
    parser.add_argument(
        "--max-memory-fraction",
        type=float,
        default=0.8,
        help="Maximum fraction of GPU memory to use (default: 0.8)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="Minimum free GPU memory to maintain in GB (default: 2.0)",
    )
    parser.add_argument(
        "--progressive-steps",
        type=int,
        default=5,
        help="Number of progressive size steps (default: 5)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32768,
        help="Maximum sequence length to test (default: 32768)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show memory estimates without running",
    )

    args = parser.parse_args()

    # Create safety config
    config = SafetyConfig(
        max_memory_fraction=args.max_memory_fraction,
        min_free_memory_gb=args.min_free_gb,
        progressive_steps=args.progressive_steps,
    )

    # Check current system state
    checker = MemorySafetyChecker(config)

    print("=== System Memory Status ===")
    if torch.cuda.is_available():
        used, free, total = checker.get_gpu_memory_info()
        print(f"GPU: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available")

    cpu_used, cpu_free, cpu_total = checker.get_cpu_memory_info()
    print(f"CPU: {cpu_used:.1f}GB used, {cpu_free:.1f}GB free, {cpu_total:.1f}GB total")

    if args.dry_run:
        print("\n=== Memory Estimates ===")
        # Estimate memory for different sequence lengths
        seq_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
        batch_size = 2
        num_heads = 8
        head_dim = 64

        for seq_len in seq_lengths:
            if seq_len > args.max_seq_len:
                break

            shape = (batch_size, seq_len, num_heads, head_dim)
            # Q, K, V tensors
            memory_gb = checker.estimate_tensor_memory(shape, torch.float16, 3)
            # Add overhead
            total_gb = memory_gb * 1.5

            can_run, message = checker.check_memory_available(total_gb)
            status = "✓" if can_run else "✗"

            print(f"Seq {seq_len:,}: ~{total_gb:.1f}GB required {status}")
            if not can_run:
                print(f"  {message}")

        return

    # Load and run benchmark
    try:
        # Load benchmark module
        if args.benchmark.endswith(".py"):
            module = load_benchmark_module(args.benchmark)
        else:
            # Import as module name
            module = importlib.import_module(args.benchmark)

        # Create safe runner
        runner = SafeBenchmarkRunner(config)

        # Look for main function or run_benchmark
        if hasattr(module, "main"):
            # Wrap the main function
            print(f"\nRunning {args.benchmark} with safety limits...")
            print(f"Max sequence length: {args.max_seq_len:,}")
            print(
                f"Memory limits: {args.max_memory_fraction:.0%} of GPU, min {args.min_free_gb}GB free\n"
            )

            # Monkey patch any sequence length limits
            if hasattr(module, "MAX_SEQ_LEN"):
                module.MAX_SEQ_LEN = min(module.MAX_SEQ_LEN, args.max_seq_len)

            # Run with safety wrapper
            module.main()

        elif hasattr(module, "run_benchmark"):
            # Use the runner to execute benchmarks
            configs = getattr(module, "BENCHMARK_CONFIGS", [])
            if configs:
                # Limit sequence lengths in configs
                for config in configs:
                    if "seq_len" in config:
                        config["seq_len"] = min(config["seq_len"], args.max_seq_len)

                results = runner.run_benchmark(module.run_benchmark, configs)
                print(f"\nCompleted {len(results)} benchmarks")
            else:
                print("No benchmark configs found")
        else:
            print(f"No main() or run_benchmark() function found in {args.benchmark}")

    except Exception as e:
        print(f"\nError running benchmark: {e}")
        checker.force_cleanup()
        raise
    finally:
        # Final cleanup
        print("\nPerforming final cleanup...")
        checker.force_cleanup()

        # Show final memory state
        if torch.cuda.is_available():
            used, free, total = checker.get_gpu_memory_info()
            print(f"Final GPU state: {used:.1f}GB used, {free:.1f}GB free")


if __name__ == "__main__":
    main()
