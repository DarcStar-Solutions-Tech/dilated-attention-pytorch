#!/usr/bin/env python3
"""
Benchmark NUMA-aware memory allocation performance.

This script evaluates the performance characteristics of NUMA-aware memory
allocation compared to standard allocation strategies.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import torch
import numpy as np

from dilated_attention_pytorch.core.numa_aware_pool import (
    NUMAAwareMemoryPool,
    NUMATopologyDetector,
)


def benchmark_numa_allocation(
    pool,
    pool_name: str,
    allocation_patterns: list,
    num_iterations: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark NUMA-aware memory allocation.

    Args:
        pool: Memory pool to benchmark
        pool_name: Name for reporting
        allocation_patterns: List of (shape, dtype) patterns
        num_iterations: Number of iterations
        device: Device to test on

    Returns:
        Benchmark results
    """
    print(f"\\nBenchmarking {pool_name}...")

    results = {
        "pool_name": pool_name,
        "device": str(device),
        "iterations": num_iterations,
        "patterns": len(allocation_patterns),
        "allocation_times": [],
        "allocation_throughput": [],
        "memory_usage": [],
        "success_rate": 0.0,
        "numa_stats": {},
    }

    successful_allocations = 0
    allocated_tensors = []

    # Warm up
    for _ in range(5):
        try:
            shape, dtype = allocation_patterns[0]
            if hasattr(pool, "allocate"):
                tensor = pool.allocate(shape, dtype, device)
            else:
                tensor = torch.empty(shape, dtype=dtype, device=device)

            if tensor is not None:
                allocated_tensors.append(tensor)
                if hasattr(pool, "stats"):
                    pool.stats.clear()  # Reset stats after warmup

        except Exception:
            pass

    # Clear warmup tensors
    allocated_tensors.clear()

    # Allocation benchmark
    total_elements = 0
    start_time = time.perf_counter()

    for i in range(num_iterations):
        pattern_idx = i % len(allocation_patterns)
        shape, dtype = allocation_patterns[pattern_idx]

        # Measure allocation time
        alloc_start = time.perf_counter()

        try:
            if hasattr(pool, "allocate"):
                # NUMA-aware pool interface
                tensor = pool.allocate(shape, dtype, device)
            else:
                # Standard PyTorch allocation
                tensor = torch.empty(shape, dtype=dtype, device=device)

            alloc_time = time.perf_counter() - alloc_start

            if tensor is not None:
                successful_allocations += 1
                allocated_tensors.append(tensor)
                results["allocation_times"].append(alloc_time * 1000)  # ms

                # Calculate throughput (elements per second)
                num_elements = torch.prod(torch.tensor(shape)).item()
                total_elements += num_elements
                throughput = num_elements / alloc_time if alloc_time > 0 else 0
                results["allocation_throughput"].append(throughput)

                # Record memory usage
                if device.type == "cuda" and torch.cuda.is_available():
                    memory_mb = torch.cuda.memory_allocated() / (1024**2)
                    results["memory_usage"].append(memory_mb)

        except Exception as e:
            results["allocation_times"].append(float("inf"))
            print(f"  Allocation failed: {e}")

    total_time = time.perf_counter() - start_time
    results["total_time"] = total_time
    results["success_rate"] = successful_allocations / num_iterations

    # Calculate overall throughput
    if total_time > 0:
        results["overall_throughput"] = total_elements / total_time

    # Get NUMA-specific stats
    if hasattr(pool, "get_stats"):
        try:
            numa_stats = pool.get_stats()
            results["numa_stats"] = numa_stats
        except Exception as e:
            print(f"  Failed to get NUMA stats: {e}")

    # Calculate statistics
    if results["allocation_times"]:
        valid_times = [t for t in results["allocation_times"] if t != float("inf")]
        if valid_times:
            results["avg_alloc_time"] = np.mean(valid_times)
            results["std_alloc_time"] = np.std(valid_times)
            results["min_alloc_time"] = np.min(valid_times)
            results["max_alloc_time"] = np.max(valid_times)
            results["p95_alloc_time"] = np.percentile(valid_times, 95)
        else:
            results["avg_alloc_time"] = float("inf")

    if results["allocation_throughput"]:
        valid_throughput = [t for t in results["allocation_throughput"] if t > 0]
        if valid_throughput:
            results["avg_throughput"] = np.mean(valid_throughput)
            results["peak_throughput"] = np.max(valid_throughput)

    if results["memory_usage"]:
        results["peak_memory"] = np.max(results["memory_usage"])
        results["avg_memory"] = np.mean(results["memory_usage"])

    # Cleanup
    for tensor in allocated_tensors:
        del tensor

    return results


def create_numa_allocation_patterns():
    """Create allocation patterns optimized for NUMA testing."""
    patterns = []

    # Small allocations (likely to stay on single NUMA node)
    patterns.extend(
        [
            ((64,), torch.float32),  # 256 bytes
            ((256,), torch.float32),  # 1KB
            ((1024,), torch.float32),  # 4KB
            ((64, 64), torch.float32),  # 16KB
        ]
    )

    # Medium allocations (may span NUMA nodes)
    patterns.extend(
        [
            ((512, 512), torch.float32),  # 1MB
            ((1024, 512), torch.float32),  # 2MB
            ((1024, 1024), torch.float32),  # 4MB
            ((2048, 1024), torch.float32),  # 8MB
        ]
    )

    # Large allocations (definitely span NUMA nodes)
    patterns.extend(
        [
            ((4096, 1024), torch.float32),  # 16MB
            ((4096, 2048), torch.float32),  # 32MB
            ((8192, 2048), torch.float32),  # 64MB
            ((8192, 4096), torch.float32),  # 128MB
        ]
    )

    # GPU-typical allocations
    patterns.extend(
        [
            ((32, 768), torch.float16),  # Transformer hidden states
            ((32, 2048), torch.float16),  # Transformer FFN
            ((8, 16, 512, 512), torch.float16),  # Attention weights
            ((16, 12, 1024, 64), torch.float16),  # Multi-head attention
        ]
    )

    return patterns


def test_numa_topology():
    """Test and report NUMA topology detection."""
    print("NUMA Topology Detection")
    print("=" * 25)

    detector = NUMATopologyDetector()
    numa_detected = detector.detect()

    if numa_detected:
        topology = detector.get_topology_info()

        print(f"✅ NUMA Available: {topology['num_nodes']} nodes detected")
        print()

        for node_id, node_info in topology["nodes"].items():
            gpus = node_info["gpu_devices"]
            gpu_str = f", GPUs: {gpus}" if gpus else ""
            print(
                f"Node {node_id}: {node_info['num_cores']} cores, "
                f"{node_info['memory_gb']:.1f}GB{gpu_str}"
            )

        if topology["gpu_affinity"]:
            print("\\nGPU-NUMA Affinity:")
            for gpu_id, numa_node in topology["gpu_affinity"].items():
                print(f"  GPU {gpu_id} -> NUMA Node {numa_node}")

        return True
    else:
        print("❌ NUMA not available or not detected")
        return False


def test_numa_affinity_impact(device: torch.device, iterations: int = 20):
    """Test impact of NUMA affinity on allocation performance."""
    print(f"\\nTesting NUMA Affinity Impact on {device}...")

    # Test without NUMA awareness
    pool_standard = NUMAAwareMemoryPool(enable_numa=False)

    # Test with NUMA awareness
    pool_numa = NUMAAwareMemoryPool(enable_numa=True)

    # Simple allocation pattern
    test_shapes = [
        (1024, 1024),  # 4MB
        (2048, 1024),  # 8MB
        (4096, 1024),  # 16MB
    ]

    results = {"standard": [], "numa_aware": []}

    for shape in test_shapes:
        print(f"  Testing shape {shape}...")

        # Test standard allocation
        times_standard = []
        for _ in range(iterations):
            start = time.perf_counter()
            tensor = pool_standard.allocate(shape, torch.float32, device)
            end = time.perf_counter()
            times_standard.append((end - start) * 1000)  # ms
            del tensor

        # Test NUMA-aware allocation
        times_numa = []
        for _ in range(iterations):
            start = time.perf_counter()
            tensor = pool_numa.allocate(shape, torch.float32, device)
            end = time.perf_counter()
            times_numa.append((end - start) * 1000)  # ms
            del tensor

        # Calculate and store results
        std_avg = np.mean(times_standard)
        numa_avg = np.mean(times_numa)
        speedup = std_avg / numa_avg if numa_avg > 0 else 1.0

        results["standard"].append(std_avg)
        results["numa_aware"].append(numa_avg)

        print(f"    Standard: {std_avg:.3f}ms")
        print(f"    NUMA-aware: {numa_avg:.3f}ms")
        print(f"    Speedup: {speedup:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NUMA-aware memory allocation"
    )
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to test (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )
    parser.add_argument(
        "--test-topology", action="store_true", help="Test NUMA topology detection"
    )
    parser.add_argument(
        "--test-affinity", action="store_true", help="Test NUMA affinity impact"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("NUMA-Aware Memory Allocation Benchmark")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")

    # Test NUMA topology
    numa_available = False
    if args.test_topology:
        numa_available = test_numa_topology()

    # Test NUMA affinity impact
    if args.test_affinity:
        test_numa_affinity_impact(device, args.iterations // 2)

    # Create allocation patterns
    patterns = create_numa_allocation_patterns()
    print(f"\\nTesting {len(patterns)} allocation patterns")

    # Test pools
    pools_to_test = []

    # NUMA-aware pool
    pools_to_test.append(
        (
            NUMAAwareMemoryPool(enable_numa=True, prefer_local_allocation=True),
            "NUMA-Aware (Local Preferred)",
        )
    )

    # NUMA-aware pool without local preference
    pools_to_test.append(
        (
            NUMAAwareMemoryPool(enable_numa=True, prefer_local_allocation=False),
            "NUMA-Aware (No Preference)",
        )
    )

    # Standard allocation (baseline)
    class StandardAllocator:
        def allocate(self, shape, dtype, device):
            return torch.empty(shape, dtype=dtype, device=device)

    pools_to_test.append((StandardAllocator(), "Standard PyTorch"))

    # Disabled NUMA pool
    pools_to_test.append((NUMAAwareMemoryPool(enable_numa=False), "NUMA Disabled"))

    # Run benchmarks
    all_results = []

    for pool, name in pools_to_test:
        try:
            result = benchmark_numa_allocation(
                pool, name, patterns, args.iterations, device
            )
            all_results.append(result)

            # Print summary
            if "avg_alloc_time" in result:
                print(f"  Avg allocation: {result['avg_alloc_time']:.3f} ms")
            if "p95_alloc_time" in result:
                print(f"  P95 allocation: {result['p95_alloc_time']:.3f} ms")
            if "avg_throughput" in result:
                print(f"  Avg throughput: {result['avg_throughput']:.0f} elements/sec")
            print(f"  Success rate: {result['success_rate']:.1%}")

            # Print NUMA-specific stats
            if result["numa_stats"]:
                numa_stats = result["numa_stats"]
                if "local_allocations" in numa_stats:
                    total_allocs = numa_stats.get("total_allocations", 1)
                    local_pct = (numa_stats["local_allocations"] / total_allocs) * 100
                    print(f"  Local allocations: {local_pct:.1f}%")

        except Exception as e:
            print(f"Benchmark failed for {name}: {e}")

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"numa-memory-benchmark-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# NUMA-Aware Memory Allocation Benchmark\\n\\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\\n\\n")

        f.write("## Configuration\\n\\n")
        f.write(f"- Device: {device}\\n")
        f.write(f"- Iterations: {args.iterations}\\n")
        f.write(f"- Allocation patterns: {len(patterns)}\\n")
        f.write(f"- NUMA Available: {'✅' if numa_available else '❌'}\\n\\n")

        f.write("## Results\\n\\n")
        f.write(
            "| Pool | Avg Alloc (ms) | P95 Alloc (ms) | Throughput (elem/s) | Success Rate | Local % |\\n"
        )
        f.write(
            "|------|----------------|----------------|---------------------|--------------|---------|\\n"
        )

        for result in all_results:
            name = result["pool_name"]
            avg_time = result.get("avg_alloc_time", float("inf"))
            p95_time = result.get("p95_alloc_time", float("inf"))
            throughput = result.get("avg_throughput", 0)
            success_rate = result["success_rate"]

            # Calculate local allocation percentage
            local_pct = ""
            numa_stats = result.get("numa_stats", {})
            if numa_stats and "total_allocations" in numa_stats:
                total = numa_stats["total_allocations"]
                local = numa_stats.get("local_allocations", 0)
                if total > 0:
                    local_pct = f"{(local / total * 100):.1f}%"

            avg_str = f"{avg_time:.3f}" if avg_time != float("inf") else "FAILED"
            p95_str = f"{p95_time:.3f}" if p95_time != float("inf") else "FAILED"
            throughput_str = f"{throughput:.0f}" if throughput > 0 else "N/A"

            f.write(
                f"| {name} | {avg_str} | {p95_str} | {throughput_str} | "
                f"{success_rate:.1%} | {local_pct} |\\n"
            )

        f.write("\\n## Key Findings\\n\\n")

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
                highest_throughput = max(
                    valid_results, key=lambda x: x.get("avg_throughput", 0)
                )

                f.write(
                    f"- Fastest allocation: {fastest['pool_name']} "
                    f"({fastest.get('avg_alloc_time', 0):.3f} ms)\\n"
                )
                f.write(
                    f"- Highest throughput: {highest_throughput['pool_name']} "
                    f"({highest_throughput.get('avg_throughput', 0):.0f} elements/sec)\\n"
                )

        f.write("\\n### NUMA-Aware Features:\\n")
        f.write("- Automatic NUMA topology detection and GPU affinity mapping\\n")
        f.write("- NUMA-local memory allocation for improved bandwidth\\n")
        f.write("- Cross-NUMA allocation tracking and optimization\\n")
        f.write("- Context-aware NUMA node selection based on device type\\n")

        if numa_available:
            f.write("\\n### NUMA Topology Detected:\\n")
            detector = NUMATopologyDetector()
            if detector.detect():
                topology = detector.get_topology_info()
                for node_id, node_info in topology["nodes"].items():
                    f.write(
                        f"- Node {node_id}: {node_info['num_cores']} cores, "
                        f"{node_info['memory_gb']:.1f}GB\\n"
                    )

    print(f"\\nBenchmark results saved to: {report_path}")

    # Generate NUMA reports
    for pool, name in pools_to_test:
        if hasattr(pool, "get_numa_report"):
            try:
                print(f"\\n{name} NUMA Report:")
                print("=" * (len(name) + 12))
                report = pool.get_numa_report()
                print(report)
            except Exception as e:
                print(f"Failed to generate NUMA report: {e}")


if __name__ == "__main__":
    main()
