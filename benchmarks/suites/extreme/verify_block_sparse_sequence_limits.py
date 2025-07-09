#!/usr/bin/env python3
"""
Verify maximum achievable sequence lengths for each block-sparse variant.
Carefully measures performance without using gather operations.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use single GPU for consistent testing

import torch
import gc
import time
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


@dataclass
class SequenceTestResult:
    """Result of testing a specific sequence length."""

    sequence_length: int
    success: bool
    forward_time_ms: Optional[float] = None
    memory_used_mb: Optional[float] = None
    error: Optional[str] = None
    theoretical_attention_memory_gb: Optional[float] = None
    actual_memory_gb: Optional[float] = None


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0


def test_sequence_length(
    variant_name: str,
    variant_config: Dict,
    seq_len: int,
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 3,
) -> SequenceTestResult:
    """Test if a specific sequence length works for a variant."""

    # Clear memory before test
    torch.cuda.empty_cache()
    gc.collect()

    # Calculate theoretical memory for full attention
    theoretical_memory_gb = (seq_len * seq_len * num_heads * 2) / 1024**3  # float16

    try:
        # Create model
        model = create_block_sparse_attention(**variant_config)
        model = model.to(device="cuda", dtype=torch.float16)

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Measure initial memory
        allocated_before, _ = get_gpu_memory_info()

        # Warmup
        for _ in range(2):
            _ = model(q, k, v)
        torch.cuda.synchronize()

        # Time forward pass
        start_time = time.time()
        for _ in range(num_iterations):
            output = model(q, k, v)
        torch.cuda.synchronize()
        forward_time = (time.time() - start_time) / num_iterations * 1000

        # Measure memory after forward
        allocated_after, _ = get_gpu_memory_info()
        memory_used_gb = allocated_after - allocated_before

        # Cleanup
        del model, q, k, v, output
        torch.cuda.empty_cache()
        gc.collect()

        return SequenceTestResult(
            sequence_length=seq_len,
            success=True,
            forward_time_ms=forward_time,
            memory_used_mb=memory_used_gb * 1024,
            theoretical_attention_memory_gb=theoretical_memory_gb,
            actual_memory_gb=memory_used_gb,
        )

    except torch.cuda.OutOfMemoryError:
        # Cleanup on OOM
        torch.cuda.empty_cache()
        gc.collect()

        return SequenceTestResult(
            sequence_length=seq_len,
            success=False,
            error="Out of memory",
            theoretical_attention_memory_gb=theoretical_memory_gb,
        )

    except Exception as e:
        # Cleanup on error
        torch.cuda.empty_cache()
        gc.collect()

        return SequenceTestResult(
            sequence_length=seq_len,
            success=False,
            error=str(e),
            theoretical_attention_memory_gb=theoretical_memory_gb,
        )


def find_max_sequence_length(
    variant_name: str,
    variant_config: Dict,
    start_len: int = 4096,
    max_len: int = 1024 * 1024,  # 1M tokens
) -> Tuple[int, Dict[int, SequenceTestResult]]:
    """Binary search to find maximum sequence length."""

    print(f"\nFinding max sequence length for {variant_name}...")

    results = {}

    # Start with powers of 2
    test_lengths = []
    current = start_len
    while current <= max_len:
        test_lengths.append(current)
        current *= 2

    # Test each length
    max_working = 0
    for seq_len in test_lengths:
        print(f"  Testing {seq_len:,} tokens...", end=" ", flush=True)

        result = test_sequence_length(variant_name, variant_config, seq_len)
        results[seq_len] = result

        if result.success:
            max_working = seq_len
            print(f"✓ ({result.forward_time_ms:.1f}ms, {result.memory_used_mb:.0f}MB)")
        else:
            print(f"✗ ({result.error})")
            break

    # Binary search between last working and first failing
    if max_working > 0 and max_working < test_lengths[-1]:
        low = max_working
        high = test_lengths[test_lengths.index(max_working) + 1]

        print(f"  Binary search between {low:,} and {high:,}...")

        while high - low > 1024:  # Stop when within 1K tokens
            mid = (low + high) // 2
            mid = (mid // 1024) * 1024  # Round to nearest 1K

            if mid in results:  # Skip if already tested
                if results[mid].success:
                    low = mid
                else:
                    high = mid
                continue

            print(f"    Testing {mid:,}...", end=" ", flush=True)
            result = test_sequence_length(variant_name, variant_config, mid)
            results[mid] = result

            if result.success:
                low = mid
                max_working = mid
                print("✓")
            else:
                high = mid
                print("✗")

    return max_working, results


def measure_performance_scaling(
    variant_name: str,
    variant_config: Dict,
    max_seq_len: int,
) -> Dict[int, SequenceTestResult]:
    """Measure performance at different sequence lengths."""

    print(f"\nMeasuring performance scaling for {variant_name}...")

    # Test at different scales
    test_lengths = [4096, 8192, 16384, 32768]
    if max_seq_len >= 65536:
        test_lengths.append(65536)
    if max_seq_len >= 131072:
        test_lengths.append(131072)

    # Only test up to max working length
    test_lengths = [length for length in test_lengths if length <= max_seq_len]

    results = {}
    for seq_len in test_lengths:
        print(f"  {seq_len:,} tokens...", end=" ", flush=True)
        result = test_sequence_length(
            variant_name, variant_config, seq_len, num_iterations=10
        )
        results[seq_len] = result

        if result.success:
            print(f"{result.forward_time_ms:.1f}ms ({result.memory_used_mb:.0f}MB)")
        else:
            print(f"Failed: {result.error}")

    return results


def main():
    """Run sequence length verification for all block-sparse variants."""

    print("Block-Sparse Maximum Sequence Length Verification")
    print("=" * 70)

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name}")
        print(f"Total Memory: {total_memory:.1f}GB")
    else:
        print("\nNo GPU available!")
        return

    # Define variants to test
    variants = {
        "Dense (baseline)": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 1.0,  # No sparsity
        },
        "90% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.1,
        },
        "95% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.05,
        },
        "99% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.01,
        },
        "99.9% Sparse": {
            "variant": "base",
            "segment_lengths": [2048],
            "dilation_rates": [1],
            "sparsity_ratio": 0.001,
        },
        # Hierarchical removed - achieved poor memory efficiency
        "Adaptive": {
            "variant": "adaptive",
            "segment_lengths": [2048],
            "dilation_rates": [1],
        },
    }

    # Track results
    all_results = {}
    max_lengths = {}

    # Test each variant
    for variant_name, variant_config in variants.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {variant_name}")
        print(f"{'=' * 70}")

        # Find maximum sequence length
        max_len, length_results = find_max_sequence_length(variant_name, variant_config)
        max_lengths[variant_name] = max_len

        # Measure performance scaling
        if max_len > 0:
            perf_results = measure_performance_scaling(
                variant_name, variant_config, max_len
            )

            # Combine results
            all_results[variant_name] = {
                "max_length": max_len,
                "length_search": length_results,
                "performance": perf_results,
            }
        else:
            all_results[variant_name] = {
                "max_length": 0,
                "length_search": length_results,
                "performance": {},
            }

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Maximum Sequence Lengths")
    print(f"{'=' * 70}")
    print(
        f"{'Variant':<20} {'Max Length':>15} {'vs Dense':>10} {'Memory Efficiency':>20}"
    )
    print("-" * 70)

    dense_max = max_lengths.get("Dense (baseline)", 1)

    for variant_name, max_len in sorted(
        max_lengths.items(), key=lambda x: x[1], reverse=True
    ):
        if max_len > 0:
            vs_dense = max_len / dense_max if dense_max > 0 else 0

            # Get memory efficiency at max length
            if (
                variant_name in all_results
                and max_len in all_results[variant_name]["length_search"]
            ):
                result = all_results[variant_name]["length_search"][max_len]
                if result.theoretical_attention_memory_gb and result.actual_memory_gb:
                    efficiency = (
                        result.theoretical_attention_memory_gb / result.actual_memory_gb
                    )
                    mem_str = f"{efficiency:.1f}x"
                else:
                    mem_str = "N/A"
            else:
                mem_str = "N/A"

            print(f"{variant_name:<20} {max_len:>15,} {vs_dense:>10.1f}x {mem_str:>20}")
        else:
            print(f"{variant_name:<20} {'Failed':>15} {'-':>10} {'-':>20}")

    # Performance scaling analysis
    print(f"\n{'=' * 70}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'=' * 70}")

    for variant_name in ["90% Sparse", "95% Sparse", "99% Sparse"]:
        if variant_name in all_results:
            results = all_results[variant_name]["performance"]
            if results:
                print(f"\n{variant_name}:")
                print(
                    f"{'Seq Length':>12} {'Time (ms)':>12} {'Memory (MB)':>12} {'ms/K tokens':>12}"
                )
                print("-" * 50)

                for seq_len in sorted(results.keys()):
                    result = results[seq_len]
                    if result.success:
                        ms_per_k = result.forward_time_ms / (seq_len / 1000)
                        print(
                            f"{seq_len:>12,} {result.forward_time_ms:>12.1f} "
                            f"{result.memory_used_mb:>12.0f} {ms_per_k:>12.2f}"
                        )

    print("\n✅ Verification completed!")

    # Save detailed results
    save_detailed_results(all_results, max_lengths)


def save_detailed_results(all_results: Dict, max_lengths: Dict):
    """Save detailed results to a report file."""

    from datetime import datetime

    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")

    report_path = f"docs/reports/block-sparse-sequence-limits-{timestamp}.md"

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# Block-Sparse Sequence Length Limits\n\n")
        f.write(f"Generated: {timestamp}\n\n")

        # GPU info
        if torch.cuda.is_available():
            f.write("## Test Environment\n\n")
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(
                f"- Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB\n"
            )
            f.write(f"- PyTorch: {torch.__version__}\n\n")

        # Maximum lengths
        f.write("## Maximum Sequence Lengths\n\n")
        f.write("| Variant | Max Length | vs Dense | Memory Efficiency |\n")
        f.write("|---------|------------|----------|------------------|\n")

        dense_max = max_lengths.get("Dense (baseline)", 1)
        for variant, max_len in sorted(
            max_lengths.items(), key=lambda x: x[1], reverse=True
        ):
            vs_dense = max_len / dense_max if dense_max > 0 else 0
            f.write(f"| {variant} | {max_len:,} | {vs_dense:.1f}x | ")

            # Add memory efficiency
            if (
                variant in all_results
                and max_len in all_results[variant]["length_search"]
            ):
                result = all_results[variant]["length_search"][max_len]
                if result.theoretical_attention_memory_gb and result.actual_memory_gb:
                    efficiency = (
                        result.theoretical_attention_memory_gb / result.actual_memory_gb
                    )
                    f.write(f"{efficiency:.1f}x |\n")
                else:
                    f.write("N/A |\n")
            else:
                f.write("N/A |\n")

        # Performance details
        f.write("\n## Performance Scaling\n\n")

        for variant in ["90% Sparse", "95% Sparse", "99% Sparse"]:
            if variant in all_results and all_results[variant]["performance"]:
                f.write(f"### {variant}\n\n")
                f.write("| Sequence Length | Time (ms) | Memory (MB) | ms/K tokens |\n")
                f.write("|-----------------|-----------|-------------|-------------|\n")

                results = all_results[variant]["performance"]
                for seq_len in sorted(results.keys()):
                    result = results[seq_len]
                    if result.success:
                        ms_per_k = result.forward_time_ms / (seq_len / 1000)
                        f.write(
                            f"| {seq_len:,} | {result.forward_time_ms:.1f} | "
                            f"{result.memory_used_mb:.0f} | {ms_per_k:.2f} |\n"
                        )
                f.write("\n")

        # Key findings
        f.write("## Key Findings\n\n")

        # Find best performers
        if max_lengths:
            best_length = max(max_lengths.values())
            best_variants = [k for k, v in max_lengths.items() if v == best_length]

            f.write(
                f"1. **Longest sequence**: {best_length:,} tokens ({', '.join(best_variants)})\n"
            )

            if dense_max > 0:
                improvement = best_length / dense_max
                f.write(
                    f"2. **Improvement over dense**: {improvement:.1f}x longer sequences\n"
                )

            # Memory efficiency
            most_efficient = None
            best_efficiency = 0

            for variant in all_results:
                for seq_len, result in all_results[variant]["length_search"].items():
                    if (
                        result.success
                        and result.theoretical_attention_memory_gb
                        and result.actual_memory_gb
                    ):
                        efficiency = (
                            result.theoretical_attention_memory_gb
                            / result.actual_memory_gb
                        )
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            most_efficient = (variant, seq_len, efficiency)

            if most_efficient:
                f.write(
                    f"3. **Best memory efficiency**: {most_efficient[2]:.1f}x "
                    f"({most_efficient[0]} at {most_efficient[1]:,} tokens)\n"
                )

    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
