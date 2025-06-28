#!/usr/bin/env python3
"""
Benchmark memory profiling system performance and overhead.

This script evaluates the performance impact and capabilities of the memory
profiling system introduced in Phase 1.4.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import torch

from dilated_attention_pytorch.core.memory_profiler import (
    MemoryProfiler,
)
from dilated_attention_pytorch.core.enhanced_memory_pool import EnhancedMemoryPool


def benchmark_profiler_overhead(
    num_allocations: int = 1000,
    allocation_sizes: list = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark the overhead of memory profiling.

    Args:
        num_allocations: Number of allocations to test
        allocation_sizes: List of allocation sizes to test
        device: Device to test on

    Returns:
        Benchmark results
    """
    if allocation_sizes is None:
        allocation_sizes = [
            (64, 32),  # Small
            (256, 128),  # Medium
            (1024, 512),  # Large
        ]

    print(f"\\nBenchmarking profiler overhead with {num_allocations} allocations...")

    results = {
        "num_allocations": num_allocations,
        "device": str(device),
        "allocation_sizes": allocation_sizes,
        "with_profiling": {},
        "without_profiling": {},
        "overhead": {},
    }

    # Test without profiling (baseline)
    print("  Testing without profiling...")
    tensors = []

    start_time = time.perf_counter()
    for i in range(num_allocations):
        shape = allocation_sizes[i % len(allocation_sizes)]
        tensor = torch.empty(shape, dtype=torch.float32, device=device)
        tensors.append(tensor)

    alloc_time_baseline = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for tensor in tensors:
        del tensor
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dealloc_time_baseline = time.perf_counter() - start_time

    results["without_profiling"] = {
        "allocation_time": alloc_time_baseline,
        "deallocation_time": dealloc_time_baseline,
        "total_time": alloc_time_baseline + dealloc_time_baseline,
    }

    # Test with profiling
    print("  Testing with profiling...")
    profiler = MemoryProfiler(
        enable_stack_traces=False,  # Disable for fair comparison
        max_events=num_allocations * 2,
    )
    profiler.start_profiling()

    tensors = []

    start_time = time.perf_counter()
    for i in range(num_allocations):
        shape = allocation_sizes[i % len(allocation_sizes)]
        tensor = torch.empty(shape, dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="benchmark")
        tensors.append(tensor)

    alloc_time_profiled = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for tensor in tensors:
        profiler.record_deallocation(tensor)
        del tensor
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dealloc_time_profiled = time.perf_counter() - start_time

    profiler.stop_profiling()

    results["with_profiling"] = {
        "allocation_time": alloc_time_profiled,
        "deallocation_time": dealloc_time_profiled,
        "total_time": alloc_time_profiled + dealloc_time_profiled,
        "profiler_overhead": profiler.stats["profiling_overhead"],
        "events_recorded": len(profiler.allocation_events),
    }

    # Calculate overhead
    alloc_overhead = (
        (alloc_time_profiled - alloc_time_baseline) / alloc_time_baseline
    ) * 100
    dealloc_overhead = (
        (dealloc_time_profiled - dealloc_time_baseline) / dealloc_time_baseline
    ) * 100
    total_overhead = (
        (
            results["with_profiling"]["total_time"]
            - results["without_profiling"]["total_time"]
        )
        / results["without_profiling"]["total_time"]
    ) * 100

    results["overhead"] = {
        "allocation_overhead_percent": alloc_overhead,
        "deallocation_overhead_percent": dealloc_overhead,
        "total_overhead_percent": total_overhead,
        "per_allocation_overhead_ms": (alloc_time_profiled - alloc_time_baseline)
        * 1000
        / num_allocations,
    }

    # Print results
    print(f"    Baseline allocation: {alloc_time_baseline:.4f}s")
    print(f"    Profiled allocation: {alloc_time_profiled:.4f}s")
    print(f"    Allocation overhead: {alloc_overhead:.2f}%")
    print(
        f"    Per-allocation overhead: {results['overhead']['per_allocation_overhead_ms']:.3f}ms"
    )
    print(f"    Total overhead: {total_overhead:.2f}%")

    return results


def benchmark_enhanced_pool_profiling(
    num_allocations: int = 500,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark enhanced memory pool with profiling enabled.

    Args:
        num_allocations: Number of allocations to test
        device: Device to test on

    Returns:
        Benchmark results
    """
    print("\\nBenchmarking enhanced pool with profiling...")

    # Test allocation patterns
    allocation_patterns = [
        ((64,), "Small vector"),
        ((256, 64), "Medium matrix"),
        ((1024, 512), "Large matrix"),
        ((2048, 1024), "Very large matrix"),
        ((32, 768), "Transformer hidden"),
        ((8, 16, 512, 512), "Attention weights"),
    ]

    results = {
        "num_allocations": num_allocations,
        "device": str(device),
        "patterns": len(allocation_patterns),
        "without_profiling": {},
        "with_profiling": {},
        "profiling_analysis": {},
    }

    # Test without profiling
    print("  Testing enhanced pool without profiling...")
    pool_no_profiling = EnhancedMemoryPool(
        enable_fragment_aware=True,
        enable_bucketed=True,
        enable_numa=True,
        enable_profiling=False,
    )

    tensors = []
    start_time = time.perf_counter()

    for i in range(num_allocations):
        shape, _ = allocation_patterns[i % len(allocation_patterns)]
        tensor = pool_no_profiling.allocate(shape, torch.float32, device)
        tensors.append(tensor)

    alloc_time_no_profiling = time.perf_counter() - start_time

    # Cleanup
    for tensor in tensors:
        pool_no_profiling.deallocate(tensor)

    results["without_profiling"] = {
        "allocation_time": alloc_time_no_profiling,
        "avg_per_allocation": alloc_time_no_profiling / num_allocations * 1000,  # ms
    }

    # Test with profiling
    print("  Testing enhanced pool with profiling...")
    pool_with_profiling = EnhancedMemoryPool(
        enable_fragment_aware=True,
        enable_bucketed=True,
        enable_numa=True,
        enable_profiling=True,
    )

    tensors = []
    start_time = time.perf_counter()

    for i in range(num_allocations):
        shape, _ = allocation_patterns[i % len(allocation_patterns)]
        tensor = pool_with_profiling.allocate(shape, torch.float32, device)
        tensors.append(tensor)

    alloc_time_with_profiling = time.perf_counter() - start_time

    # Get profiling data
    profiling_summary = pool_with_profiling.profiler.get_allocation_summary()

    # Cleanup
    for tensor in tensors:
        pool_with_profiling.deallocate(tensor)

    results["with_profiling"] = {
        "allocation_time": alloc_time_with_profiling,
        "avg_per_allocation": alloc_time_with_profiling / num_allocations * 1000,  # ms
        "events_recorded": len(pool_with_profiling.profiler.allocation_events),
    }

    # Calculate overhead
    overhead_percent = (
        (alloc_time_with_profiling - alloc_time_no_profiling) / alloc_time_no_profiling
    ) * 100

    results["profiling_analysis"] = {
        "overhead_percent": overhead_percent,
        "overhead_per_allocation_ms": (
            alloc_time_with_profiling - alloc_time_no_profiling
        )
        * 1000
        / num_allocations,
        "pool_breakdown": profiling_summary.get("pool_breakdown", {}),
        "operation_breakdown": profiling_summary.get("operation_breakdown", {}),
    }

    print(
        f"    Without profiling: {alloc_time_no_profiling:.4f}s ({results['without_profiling']['avg_per_allocation']:.3f}ms/alloc)"
    )
    print(
        f"    With profiling: {alloc_time_with_profiling:.4f}s ({results['with_profiling']['avg_per_allocation']:.3f}ms/alloc)"
    )
    print(f"    Profiling overhead: {overhead_percent:.2f}%")
    print(f"    Events recorded: {results['with_profiling']['events_recorded']}")

    return results


def benchmark_pattern_detection(
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark pattern detection capabilities.

    Args:
        device: Device to test on

    Returns:
        Benchmark results
    """
    print("\\nBenchmarking pattern detection...")

    profiler = MemoryProfiler(
        max_events=1000,
        pattern_analysis_window=50,
    )
    profiler.start_profiling()

    results = {
        "device": str(device),
        "patterns_created": [],
        "patterns_detected": 0,
        "detection_accuracy": 0.0,
    }

    # Create burst pattern
    print("  Creating burst allocation pattern...")
    burst_start = time.perf_counter()
    for i in range(30):
        tensor = torch.empty((32, 32), dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="burst_test")
        time.sleep(0.001)  # Very fast allocations

    results["patterns_created"].append(
        {
            "type": "burst",
            "count": 30,
            "duration": time.perf_counter() - burst_start,
        }
    )

    # Create large allocation pattern
    print("  Creating large allocation pattern...")
    for i in range(10):
        size = (1024 * (i + 1), 512)  # Increasingly large
        tensor = torch.empty(size, dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type="large_test")
        time.sleep(0.01)

    results["patterns_created"].append(
        {
            "type": "large_allocation",
            "count": 10,
            "sizes": "1024x512 to 10240x512",
        }
    )

    # Force pattern analysis
    profiler._analyze_patterns()

    # Check detected patterns
    results["patterns_detected"] = len(profiler.detected_patterns)

    if profiler.detected_patterns:
        print(f"    Detected {len(profiler.detected_patterns)} patterns:")
        for pattern in profiler.detected_patterns:
            print(
                f"      {pattern.pattern_type}: {pattern.description} (confidence: {pattern.confidence:.2f})"
            )

    profiler.stop_profiling()

    return results


def benchmark_visualization_performance(
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Benchmark visualization generation performance.

    Args:
        device: Device to test on

    Returns:
        Benchmark results
    """
    print("\\nBenchmarking visualization performance...")

    # Create profiler with substantial data
    profiler = MemoryProfiler(max_events=500, snapshot_interval=0.05)
    profiler.start_profiling()

    # Generate diverse allocation data
    for i in range(100):
        size_factor = (i % 10) + 1
        shape = (64 * size_factor, 32)
        tensor = torch.empty(shape, dtype=torch.float32, device=device)
        profiler.record_allocation(tensor, pool_type=f"pool_{i % 5}")

        if i % 10 == 0:
            time.sleep(0.01)  # Create temporal gaps

    # Wait for snapshots
    time.sleep(0.3)
    profiler.stop_profiling()

    results = {
        "device": str(device),
        "data_size": {
            "allocation_events": len(profiler.allocation_events),
            "memory_snapshots": len(profiler.memory_snapshots),
        },
        "visualization_times": {},
    }

    try:
        from dilated_attention_pytorch.core.memory_visualizer import MemoryVisualizer

        visualizer = MemoryVisualizer(profiler)

        # Benchmark timeline plot
        start_time = time.perf_counter()
        timeline_html = visualizer.plot_memory_timeline(duration=60.0, interactive=True)
        timeline_time = time.perf_counter() - start_time
        results["visualization_times"]["timeline"] = timeline_time

        # Benchmark distribution plot
        start_time = time.perf_counter()
        dist_html = visualizer.plot_allocation_distribution(interactive=True)
        dist_time = time.perf_counter() - start_time
        results["visualization_times"]["distribution"] = dist_time

        # Benchmark dashboard creation
        start_time = time.perf_counter()
        dashboard = visualizer.create_dashboard()
        dashboard_time = time.perf_counter() - start_time
        results["visualization_times"]["dashboard"] = dashboard_time

        # Calculate data sizes
        results["output_sizes"] = {
            "timeline_html_kb": len(timeline_html) / 1024 if timeline_html else 0,
            "distribution_html_kb": len(dist_html) / 1024 if dist_html else 0,
            "dashboard_html_kb": len(dashboard) / 1024,
        }

        print(f"    Timeline plot: {timeline_time:.3f}s")
        print(f"    Distribution plot: {dist_time:.3f}s")
        print(f"    Dashboard: {dashboard_time:.3f}s")
        print(
            f"    Dashboard size: {results['output_sizes']['dashboard_html_kb']:.1f} KB"
        )

    except ImportError:
        print("    Visualization libraries not available - skipping")
        results["visualization_times"] = {"error": "Libraries not available"}

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory profiling system")
    parser.add_argument(
        "--allocations", type=int, default=1000, help="Number of allocations to test"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to test (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )
    parser.add_argument(
        "--test-overhead", action="store_true", help="Test profiler overhead"
    )
    parser.add_argument(
        "--test-enhanced", action="store_true", help="Test enhanced pool profiling"
    )
    parser.add_argument(
        "--test-patterns", action="store_true", help="Test pattern detection"
    )
    parser.add_argument(
        "--test-visualization",
        action="store_true",
        help="Test visualization performance",
    )
    parser.add_argument("--test-all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Memory Profiling System Benchmark")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Allocations: {args.allocations}")

    all_results = {}

    # Run benchmarks
    if args.test_overhead or args.test_all:
        all_results["overhead"] = benchmark_profiler_overhead(
            num_allocations=args.allocations, device=device
        )

    if args.test_enhanced or args.test_all:
        all_results["enhanced_pool"] = benchmark_enhanced_pool_profiling(
            num_allocations=args.allocations // 2, device=device
        )

    if args.test_patterns or args.test_all:
        all_results["pattern_detection"] = benchmark_pattern_detection(device=device)

    if args.test_visualization or args.test_all:
        all_results["visualization"] = benchmark_visualization_performance(
            device=device
        )

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"memory-profiler-benchmark-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Memory Profiling System Benchmark\\n\\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\\n\\n")

        f.write("## Configuration\\n\\n")
        f.write(f"- Device: {device}\\n")
        f.write(f"- Allocations: {args.allocations}\\n")
        f.write(f"- PyTorch Version: {torch.__version__}\\n\\n")

        # Overhead results
        if "overhead" in all_results:
            overhead = all_results["overhead"]
            f.write("## Profiler Overhead\\n\\n")
            f.write("| Metric | Value |\\n")
            f.write("|--------|-------|\\n")
            f.write(
                f"| Allocation Overhead | {overhead['overhead']['allocation_overhead_percent']:.2f}% |\\n"
            )
            f.write(
                f"| Deallocation Overhead | {overhead['overhead']['deallocation_overhead_percent']:.2f}% |\\n"
            )
            f.write(
                f"| Total Overhead | {overhead['overhead']['total_overhead_percent']:.2f}% |\\n"
            )
            f.write(
                f"| Per-Allocation Overhead | {overhead['overhead']['per_allocation_overhead_ms']:.3f}ms |\\n"
            )
            f.write(
                f"| Events Recorded | {overhead['with_profiling']['events_recorded']} |\\n\\n"
            )

        # Enhanced pool results
        if "enhanced_pool" in all_results:
            enhanced = all_results["enhanced_pool"]
            f.write("## Enhanced Pool Profiling\\n\\n")
            f.write("| Pool Configuration | Time (ms/alloc) |\\n")
            f.write("|--------------------|-----------------|\\n")
            f.write(
                f"| Without Profiling | {enhanced['without_profiling']['avg_per_allocation']:.3f} |\\n"
            )
            f.write(
                f"| With Profiling | {enhanced['with_profiling']['avg_per_allocation']:.3f} |\\n"
            )
            f.write(
                f"| Overhead | {enhanced['profiling_analysis']['overhead_percent']:.2f}% |\\n\\n"
            )

            if enhanced["profiling_analysis"]["pool_breakdown"]:
                f.write("### Pool Usage Breakdown\\n\\n")
                for pool_type, stats in enhanced["profiling_analysis"][
                    "pool_breakdown"
                ].items():
                    f.write(f"- **{pool_type}**: {stats['count']} allocations\\n")
                f.write("\\n")

        # Pattern detection results
        if "pattern_detection" in all_results:
            patterns = all_results["pattern_detection"]
            f.write("## Pattern Detection\\n\\n")
            f.write(f"- Patterns Created: {len(patterns['patterns_created'])}\\n")
            f.write(f"- Patterns Detected: {patterns['patterns_detected']}\\n\\n")

        # Visualization results
        if "visualization" in all_results:
            viz = all_results["visualization"]
            if "error" not in viz["visualization_times"]:
                f.write("## Visualization Performance\\n\\n")
                f.write("| Visualization | Time (s) | Size (KB) |\\n")
                f.write("|---------------|----------|-----------|\\n")

                times = viz["visualization_times"]
                sizes = viz.get("output_sizes", {})

                f.write(
                    f"| Timeline | {times.get('timeline', 0):.3f} | {sizes.get('timeline_html_kb', 0):.1f} |\\n"
                )
                f.write(
                    f"| Distribution | {times.get('distribution', 0):.3f} | {sizes.get('distribution_html_kb', 0):.1f} |\\n"
                )
                f.write(
                    f"| Dashboard | {times.get('dashboard', 0):.3f} | {sizes.get('dashboard_html_kb', 0):.1f} |\\n\\n"
                )

        f.write("## Key Findings\\n\\n")

        if "overhead" in all_results:
            overhead_pct = all_results["overhead"]["overhead"]["total_overhead_percent"]
            if overhead_pct < 5:
                f.write(
                    f"- ✅ **Low overhead**: Profiling adds only {overhead_pct:.1f}% overhead\\n"
                )
            elif overhead_pct < 15:
                f.write(
                    f"- ⚠️ **Moderate overhead**: Profiling adds {overhead_pct:.1f}% overhead\\n"
                )
            else:
                f.write(
                    f"- ❌ **High overhead**: Profiling adds {overhead_pct:.1f}% overhead\\n"
                )

        if "pattern_detection" in all_results:
            patterns_detected = all_results["pattern_detection"]["patterns_detected"]
            if patterns_detected > 0:
                f.write(
                    f"- ✅ **Pattern detection working**: Detected {patterns_detected} patterns\\n"
                )
            else:
                f.write("- ⚠️ **Pattern detection**: No patterns detected\\n")

        f.write("\\n### Profiling System Features:\\n")
        f.write("- Real-time allocation and deallocation tracking\\n")
        f.write("- Memory timeline with snapshots\\n")
        f.write("- Allocation pattern detection and analysis\\n")
        f.write("- Integration with enhanced memory pools\\n")
        f.write("- Interactive visualization and reporting\\n")

    print(f"\\nBenchmark results saved to: {report_path}")

    # Print summary
    print("\\nBenchmark Summary:")
    print("=" * 20)

    if "overhead" in all_results:
        overhead_pct = all_results["overhead"]["overhead"]["total_overhead_percent"]
        print(f"Profiler Overhead: {overhead_pct:.2f}%")

    if "enhanced_pool" in all_results:
        enhanced_overhead = all_results["enhanced_pool"]["profiling_analysis"][
            "overhead_percent"
        ]
        print(f"Enhanced Pool Overhead: {enhanced_overhead:.2f}%")

    if "pattern_detection" in all_results:
        patterns = all_results["pattern_detection"]["patterns_detected"]
        print(f"Patterns Detected: {patterns}")


if __name__ == "__main__":
    main()
