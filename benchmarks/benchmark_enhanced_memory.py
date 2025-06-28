#!/usr/bin/env python3
"""
Benchmark enhanced memory management system.

This script tests the performance and efficiency of the new fragment-aware
and bucketed memory pools introduced in Phase 1.4.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from dilated_attention_pytorch.core.enhanced_memory_pool import (
    EnhancedMemoryPool,
)
from dilated_attention_pytorch.core.bucketed_memory_pool import BucketedMemoryPool


def benchmark_memory_pool(
    pool,
    pool_name: str,
    allocation_patterns: list,
    num_iterations: int = 100,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark a memory pool with various allocation patterns.

    Args:
        pool: Memory pool to benchmark
        pool_name: Name for reporting
        allocation_patterns: List of (shape, dtype) patterns
        num_iterations: Number of iterations
        device: Device to test on

    Returns:
        Benchmark results
    """
    print(f"\nBenchmarking {pool_name}...")

    results = {
        "pool_name": pool_name,
        "device": str(device),
        "iterations": num_iterations,
        "patterns": len(allocation_patterns),
        "allocation_times": [],
        "deallocation_times": [],
        "memory_usage": [],
        "success_rate": 0.0,
    }

    successful_allocations = 0
    allocated_tensors = []

    # Warm up
    for _ in range(5):
        try:
            if hasattr(pool, "allocate"):
                shape, dtype = allocation_patterns[0]
                tensor = pool.allocate(shape, dtype, device)
                if tensor is not None:
                    if hasattr(pool, "deallocate"):
                        pool.deallocate(tensor)
        except Exception:
            pass

    # Allocation benchmark
    for i in range(num_iterations):
        pattern_idx = i % len(allocation_patterns)
        shape, dtype = allocation_patterns[pattern_idx]

        # Measure allocation time
        start_time = time.perf_counter()

        try:
            if hasattr(pool, "allocate"):
                # Enhanced pool interface
                tensor = pool.allocate(shape, dtype, device)
            else:
                # Standard PyTorch allocation
                tensor = torch.empty(shape, dtype=dtype, device=device)

            alloc_time = time.perf_counter() - start_time

            if tensor is not None:
                successful_allocations += 1
                allocated_tensors.append(tensor)
                results["allocation_times"].append(alloc_time * 1000)  # ms

                # Record memory usage
                if device.type == "cuda" and torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024**2)
                    results["memory_usage"].append(memory_mb)

        except Exception as e:
            results["allocation_times"].append(float("inf"))
            print(f"Allocation failed: {e}")

    # Deallocation benchmark
    for tensor in allocated_tensors:
        start_time = time.perf_counter()

        try:
            if hasattr(pool, "deallocate"):
                pool.deallocate(tensor)
            else:
                del tensor  # Standard deallocation

            dealloc_time = time.perf_counter() - start_time
            results["deallocation_times"].append(dealloc_time * 1000)  # ms

        except Exception as e:
            results["deallocation_times"].append(float("inf"))
            print(f"Deallocation failed: {e}")

    results["success_rate"] = successful_allocations / num_iterations

    # Calculate statistics
    if results["allocation_times"]:
        valid_times = [t for t in results["allocation_times"] if t != float("inf")]
        if valid_times:
            results["avg_alloc_time"] = np.mean(valid_times)
            results["std_alloc_time"] = np.std(valid_times)
            results["min_alloc_time"] = np.min(valid_times)
            results["max_alloc_time"] = np.max(valid_times)
        else:
            results["avg_alloc_time"] = float("inf")

    if results["deallocation_times"]:
        valid_times = [t for t in results["deallocation_times"] if t != float("inf")]
        if valid_times:
            results["avg_dealloc_time"] = np.mean(valid_times)
            results["std_dealloc_time"] = np.std(valid_times)
        else:
            results["avg_dealloc_time"] = float("inf")

    if results["memory_usage"]:
        results["peak_memory"] = np.max(results["memory_usage"])
        results["avg_memory"] = np.mean(results["memory_usage"])

    return results


def create_allocation_patterns():
    """Create allocation patterns for testing."""
    patterns = []

    # Small allocations (transformer components)
    patterns.extend(
        [
            ((64,), torch.float32),  # Small bias
            ((256,), torch.float32),  # Medium bias
            ((512, 64), torch.float32),  # Small weight matrix
            ((1024, 64), torch.float32),  # Medium weight matrix
        ]
    )

    # Medium allocations (activations)
    patterns.extend(
        [
            ((32, 512), torch.float16),  # Batch activations
            ((32, 1024), torch.float16),  # Larger activations
            ((16, 2048), torch.float16),  # Sequence activations
            ((8, 4096), torch.float16),  # Long sequence
        ]
    )

    # Large allocations (attention matrices)
    patterns.extend(
        [
            ((8, 16, 512, 512), torch.float16),  # Attention weights
            ((4, 16, 1024, 1024), torch.float16),  # Large attention
            ((2, 16, 2048, 2048), torch.float16),  # Very large attention
            ((1, 16, 4096, 64), torch.float16),  # Sequence x head_dim
        ]
    )

    # Irregular sizes (fragmentation test)
    patterns.extend(
        [
            ((777,), torch.float32),  # Odd size
            ((123, 456), torch.float32),  # Irregular matrix
            ((1337, 42), torch.float16),  # Another irregular
            ((2049,), torch.float32),  # Just over power of 2
        ]
    )

    return patterns


def test_fragmentation_handling(device: torch.device):
    """Test fragmentation handling capabilities."""
    print(f"\nTesting fragmentation handling on {device}...")

    # Create pools
    enhanced_pool = EnhancedMemoryPool(
        enable_fragment_aware=True,
        enable_bucketed=True,
        fragmentation_threshold=0.2,
    )

    # Create fragmented allocation pattern
    tensors = []

    # Allocate many small blocks
    for i in range(50):
        try:
            tensor = enhanced_pool.allocate((256, 64), torch.float32, device)
            if tensor is not None:
                tensors.append(tensor)
        except Exception as e:
            print(f"Allocation {i} failed: {e}")
            break

    print(f"Allocated {len(tensors)} tensors")

    # Deallocate every other block to create fragmentation
    for i in range(0, len(tensors), 2):
        enhanced_pool.deallocate(tensors[i])

    print("Deallocated every other tensor to create fragmentation")

    # Get fragmentation stats
    stats = enhanced_pool.get_stats()
    print(f"Memory pool stats: {stats}")

    # Try to allocate a large block
    try:
        large_tensor = enhanced_pool.allocate((1024, 512), torch.float32, device)
        if large_tensor is not None:
            print("Successfully allocated large tensor despite fragmentation")
            enhanced_pool.deallocate(large_tensor)
        else:
            print("Failed to allocate large tensor")
    except Exception as e:
        print(f"Large allocation failed: {e}")

    # Cleanup remaining tensors
    for tensor in tensors[1::2]:  # Odd indices (not yet deallocated)
        enhanced_pool.deallocate(tensor)

    print("Fragmentation test complete")


def main():
    parser = argparse.ArgumentParser(description="Benchmark enhanced memory management")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to test (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )
    parser.add_argument(
        "--test-fragmentation", action="store_true", help="Test fragmentation handling"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Enhanced Memory Management Benchmark")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")

    # Create allocation patterns
    patterns = create_allocation_patterns()
    print(f"Testing {len(patterns)} allocation patterns")

    # Test pools
    pools_to_test = []

    # Enhanced pool (fragment-aware + bucketed)
    pools_to_test.append(
        (
            EnhancedMemoryPool(enable_fragment_aware=True, enable_bucketed=True),
            "Enhanced (Fragment + Bucketed)",
        )
    )

    # Bucketed only
    pools_to_test.append((BucketedMemoryPool(adaptive_buckets=True), "Bucketed Only"))

    # Standard PyTorch allocation (baseline)
    class StandardAllocator:
        def allocate(self, shape, dtype, device):
            return torch.empty(shape, dtype=dtype, device=device)

        def deallocate(self, tensor):
            del tensor

    pools_to_test.append((StandardAllocator(), "Standard PyTorch"))

    # Run benchmarks
    all_results = []

    for pool, name in pools_to_test:
        try:
            result = benchmark_memory_pool(
                pool, name, patterns, args.iterations, device
            )
            all_results.append(result)

            # Print summary
            if "avg_alloc_time" in result:
                print(f"  Avg allocation: {result['avg_alloc_time']:.3f} ms")
            if "avg_dealloc_time" in result:
                print(f"  Avg deallocation: {result['avg_dealloc_time']:.3f} ms")
            print(f"  Success rate: {result['success_rate']:.1%}")

        except Exception as e:
            print(f"Benchmark failed for {name}: {e}")

    # Test fragmentation handling
    if args.test_fragmentation:
        test_fragmentation_handling(device)

    # Generate report
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"enhanced-memory-benchmark-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Enhanced Memory Management Benchmark\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Iterations: {args.iterations}\n")
        f.write(f"- Allocation patterns: {len(patterns)}\n\n")

        f.write("## Results\n\n")
        f.write(
            "| Pool | Avg Alloc (ms) | Avg Dealloc (ms) | Success Rate | Peak Memory (MB) |\n"
        )
        f.write(
            "|------|----------------|-------------------|--------------|------------------|\n"
        )

        for result in all_results:
            name = result["pool_name"]
            alloc_time = result.get("avg_alloc_time", float("inf"))
            dealloc_time = result.get("avg_dealloc_time", float("inf"))
            success_rate = result["success_rate"]
            peak_memory = result.get("peak_memory", 0)

            if alloc_time == float("inf"):
                alloc_str = "FAILED"
            else:
                alloc_str = f"{alloc_time:.3f}"

            if dealloc_time == float("inf"):
                dealloc_str = "FAILED"
            else:
                dealloc_str = f"{dealloc_time:.3f}"

            f.write(
                f"| {name} | {alloc_str} | {dealloc_str} | {success_rate:.1%} | {peak_memory:.1f} |\n"
            )

        f.write("\n## Key Findings\n\n")

        if all_results:
            # Find best performers
            valid_results = [
                r
                for r in all_results
                if r.get("avg_alloc_time", float("inf")) != float("inf")
            ]

            if valid_results:
                fastest = min(
                    valid_results, key=lambda x: x.get("avg_alloc_time", float("inf"))
                )
                most_reliable = max(all_results, key=lambda x: x["success_rate"])

                f.write(
                    f"- Fastest allocation: {fastest['pool_name']} ({fastest.get('avg_alloc_time', 0):.3f} ms)\n"
                )
                f.write(
                    f"- Most reliable: {most_reliable['pool_name']} ({most_reliable['success_rate']:.1%} success)\n"
                )

        f.write("\n### Enhanced Pool Features:\n")
        f.write("- Automatic strategy selection based on allocation size\n")
        f.write("- Fragment-aware allocation reduces memory fragmentation\n")
        f.write("- Bucketed allocation optimizes common transformer patterns\n")
        f.write("- Adaptive bucket creation handles irregular allocation sizes\n")

    print(f"\nBenchmark results saved to: {report_path}")

    # Print efficiency reports for enhanced pools
    for pool, name in pools_to_test:
        if hasattr(pool, "get_efficiency_report"):
            print(f"\n{name} Efficiency Report:")
            print("=" * 50)
            try:
                report = pool.get_efficiency_report()
                print(report)
            except Exception as e:
                print(f"Failed to generate efficiency report: {e}")


if __name__ == "__main__":
    main()
