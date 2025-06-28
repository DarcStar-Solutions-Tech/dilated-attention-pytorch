#!/usr/bin/env python3
"""
Benchmark memory pool integration across realistic training scenarios.

This script tests memory pool benefits in scenarios where they provide value:
1. Long-running training with many forward/backward passes
2. Variable sequence lengths (shows pool reuse benefits)
3. Large batch training scenarios
4. Memory-constrained environments
"""

import argparse
import time
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path

from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def benchmark_training_simulation(
    num_epochs: int = 5,
    steps_per_epoch: int = 50,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Simulate training workload to show memory pool benefits over time.

    Args:
        num_epochs: Number of training epochs to simulate
        steps_per_epoch: Number of steps per epoch
        device: Device to run on

    Returns:
        Training simulation results
    """
    print(
        f"\nSimulating training workload ({num_epochs} epochs, {steps_per_epoch} steps/epoch)"
    )

    results = {
        "num_epochs": num_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": num_epochs * steps_per_epoch,
        "device": str(device),
        "without_pool": {},
        "with_pool": {},
        "improvements": {},
    }

    # Configuration (reduced for memory efficiency)
    batch_size = 2
    seq_len = 4096
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Create models
    model_no_pool = nn.Sequential(
        ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            enable_memory_pool=False,
        ),
        nn.Linear(num_heads * head_dim, num_heads * head_dim),
    ).to(device)

    model_with_pool = nn.Sequential(
        ImprovedDilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            enable_memory_pool=True,
            enable_profiling=False,
        ),
        nn.Linear(num_heads * head_dim, num_heads * head_dim),
    ).to(device)

    # Create optimizers
    optimizer_no_pool = torch.optim.AdamW(model_no_pool.parameters(), lr=1e-4)
    optimizer_with_pool = torch.optim.AdamW(model_with_pool.parameters(), lr=1e-4)

    # Test WITHOUT memory pool
    print("  Training without memory pool...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    _ = torch.cuda.memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # Generate batch data
            query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            target = torch.randn(
                batch_size, seq_len, num_heads * head_dim, device=device
            )

            # Forward pass
            output = model_no_pool[0](query, key, value)
            output = output.view(batch_size, seq_len, -1)
            output = model_no_pool[1](output)

            # Compute loss and backward pass
            loss = nn.functional.mse_loss(output, target)
            optimizer_no_pool.zero_grad()
            loss.backward()
            optimizer_no_pool.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Clean up intermediate tensors
            del query, key, value, target, output, loss

    end_time = time.perf_counter()
    peak_memory_no_pool = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["without_pool"] = {
        "total_time": end_time - start_time,
        "time_per_step": (end_time - start_time) / (num_epochs * steps_per_epoch),
        "peak_memory_mb": peak_memory_no_pool,
    }

    del model_no_pool, optimizer_no_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH memory pool
    print("  Training with memory pool...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # Generate batch data
            query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            target = torch.randn(
                batch_size, seq_len, num_heads * head_dim, device=device
            )

            # Forward pass
            output = model_with_pool[0](query, key, value)
            output = output.view(batch_size, seq_len, -1)
            output = model_with_pool[1](output)

            # Compute loss and backward pass
            loss = nn.functional.mse_loss(output, target)
            optimizer_with_pool.zero_grad()
            loss.backward()
            optimizer_with_pool.step()

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Clean up intermediate tensors
            del query, key, value, target, output, loss

    end_time = time.perf_counter()
    peak_memory_with_pool = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["with_pool"] = {
        "total_time": end_time - start_time,
        "time_per_step": (end_time - start_time) / (num_epochs * steps_per_epoch),
        "peak_memory_mb": peak_memory_with_pool,
    }

    # Calculate improvements
    time_improvement = (
        (
            results["without_pool"]["time_per_step"]
            - results["with_pool"]["time_per_step"]
        )
        / results["without_pool"]["time_per_step"]
    ) * 100
    memory_improvement = (
        (
            results["without_pool"]["peak_memory_mb"]
            - results["with_pool"]["peak_memory_mb"]
        )
        / results["without_pool"]["peak_memory_mb"]
    ) * 100

    results["improvements"] = {
        "time_improvement_percent": time_improvement,
        "memory_improvement_percent": memory_improvement,
    }

    print(
        f"    Without pool: {results['without_pool']['time_per_step']:.4f}s/step, {results['without_pool']['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    With pool:    {results['with_pool']['time_per_step']:.4f}s/step, {results['with_pool']['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    Improvements: {time_improvement:.1f}% time, {memory_improvement:.1f}% memory"
    )

    del model_with_pool, optimizer_with_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_variable_sequence_lengths(
    num_sequences: int = 100,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Test memory pool benefits with variable sequence lengths.

    Args:
        num_sequences: Number of sequences to process
        device: Device to run on

    Returns:
        Variable sequence length results
    """
    print(f"\nTesting variable sequence lengths ({num_sequences} sequences)")

    results = {
        "num_sequences": num_sequences,
        "device": str(device),
        "sequence_lengths": [],
        "without_pool": {},
        "with_pool": {},
        "improvements": {},
    }

    # Configuration
    batch_size = 2
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Generate variable sequence lengths (multiples of 4096)
    base_lengths = [4096, 8192, 12288, 16384]
    sequence_lengths = [
        base_lengths[i % len(base_lengths)] for i in range(num_sequences)
    ]
    results["sequence_lengths"] = list(set(sequence_lengths))

    # Create models
    attention_no_pool = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=False,
    ).to(device)

    attention_with_pool = ImprovedDilatedAttention(
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
        enable_memory_pool=True,
        enable_profiling=False,
    ).to(device)

    # Test WITHOUT memory pool
    print("  Processing without memory pool...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    for i, seq_len in enumerate(sequence_lengths):
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        output = attention_no_pool(query, key, value)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Clean up
        del query, key, value, output

    end_time = time.perf_counter()
    peak_memory_no_pool = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["without_pool"] = {
        "total_time": end_time - start_time,
        "time_per_sequence": (end_time - start_time) / num_sequences,
        "peak_memory_mb": peak_memory_no_pool,
    }

    del attention_no_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test WITH memory pool
    print("  Processing with memory pool...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    for i, seq_len in enumerate(sequence_lengths):
        query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

        output = attention_with_pool(query, key, value)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Clean up
        del query, key, value, output

    end_time = time.perf_counter()
    peak_memory_with_pool = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0
    )

    results["with_pool"] = {
        "total_time": end_time - start_time,
        "time_per_sequence": (end_time - start_time) / num_sequences,
        "peak_memory_mb": peak_memory_with_pool,
    }

    # Calculate improvements
    time_improvement = (
        (
            results["without_pool"]["time_per_sequence"]
            - results["with_pool"]["time_per_sequence"]
        )
        / results["without_pool"]["time_per_sequence"]
    ) * 100
    memory_improvement = (
        (
            results["without_pool"]["peak_memory_mb"]
            - results["with_pool"]["peak_memory_mb"]
        )
        / results["without_pool"]["peak_memory_mb"]
    ) * 100

    results["improvements"] = {
        "time_improvement_percent": time_improvement,
        "memory_improvement_percent": memory_improvement,
    }

    print(
        f"    Without pool: {results['without_pool']['time_per_sequence']:.4f}s/seq, {results['without_pool']['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    With pool:    {results['with_pool']['time_per_sequence']:.4f}s/seq, {results['with_pool']['peak_memory_mb']:.1f}MB peak"
    )
    print(
        f"    Improvements: {time_improvement:.1f}% time, {memory_improvement:.1f}% memory"
    )

    del attention_with_pool
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_large_batch_scenario(
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Test memory pool benefits with large batch scenarios.

    Args:
        device: Device to run on

    Returns:
        Large batch scenario results
    """
    print(f"\nTesting large batch scenarios on {device}")

    results = {
        "device": str(device),
        "scenarios": [],
    }

    # Test different batch sizes
    batch_sizes = [8, 16, 32] if device.type == "cuda" else [4, 8, 16]
    seq_len = 4096
    num_heads = 8
    head_dim = 64
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    for batch_size in batch_sizes:
        print(f"  Testing batch_size={batch_size}")

        scenario_result = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "without_pool": {},
            "with_pool": {},
            "improvements": {},
        }

        try:
            # Create models
            attention_no_pool = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                enable_memory_pool=False,
            ).to(device)

            attention_with_pool = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                enable_memory_pool=True,
                enable_profiling=False,
            ).to(device)

            # Generate test data
            query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Test without pool
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            for _ in range(10):  # Multiple iterations
                output = attention_no_pool(query, key, value)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            end_time = time.perf_counter()

            no_pool_time = (end_time - start_time) / 10
            no_pool_memory = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0
            )

            scenario_result["without_pool"] = {
                "time_per_iteration": no_pool_time,
                "peak_memory_mb": no_pool_memory,
                "success": True,
            }

            # Test with pool
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.perf_counter()
            for _ in range(10):  # Multiple iterations
                output = attention_with_pool(query, key, value)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            end_time = time.perf_counter()

            with_pool_time = (end_time - start_time) / 10
            with_pool_memory = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if device.type == "cuda"
                else 0
            )

            scenario_result["with_pool"] = {
                "time_per_iteration": with_pool_time,
                "peak_memory_mb": with_pool_memory,
                "success": True,
            }

            # Calculate improvements
            time_improvement = ((no_pool_time - with_pool_time) / no_pool_time) * 100
            memory_improvement = (
                ((no_pool_memory - with_pool_memory) / no_pool_memory) * 100
                if no_pool_memory > 0
                else 0
            )

            scenario_result["improvements"] = {
                "time_improvement_percent": time_improvement,
                "memory_improvement_percent": memory_improvement,
            }

            print(f"    Without pool: {no_pool_time:.4f}s/iter, {no_pool_memory:.1f}MB")
            print(
                f"    With pool:    {with_pool_time:.4f}s/iter, {with_pool_memory:.1f}MB"
            )
            print(
                f"    Improvements: {time_improvement:.1f}% time, {memory_improvement:.1f}% memory"
            )

            del attention_no_pool, attention_with_pool, query, key, value, output

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"    Failed with batch_size={batch_size}: {e}")
            scenario_result["without_pool"] = {"success": False, "error": str(e)}
            scenario_result["with_pool"] = {"success": False, "error": str(e)}
            scenario_result["improvements"] = {"success": False}

        results["scenarios"].append(scenario_result)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark memory pool integration across realistic scenarios"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (cpu/cuda/auto)"
    )
    parser.add_argument(
        "--test-training", action="store_true", help="Test training simulation"
    )
    parser.add_argument(
        "--test-variable-seq",
        action="store_true",
        help="Test variable sequence lengths",
    )
    parser.add_argument(
        "--test-large-batch", action="store_true", help="Test large batch scenarios"
    )
    parser.add_argument(
        "--output-dir", type=str, default="docs/benchmarks", help="Output directory"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Memory Pool Integration - Realistic Scenarios Benchmark")
    print("=" * 60)
    print(f"Device: {device}")

    all_results = {}

    # Training simulation
    if args.test_training:
        all_results["training"] = benchmark_training_simulation(device=device)

    # Variable sequence lengths
    if args.test_variable_seq:
        all_results["variable_seq"] = benchmark_variable_sequence_lengths(device=device)

    # Large batch scenarios
    if args.test_large_batch:
        all_results["large_batch"] = benchmark_large_batch_scenario(device=device)

    # If no specific tests requested, run training simulation
    if not any([args.test_training, args.test_variable_seq, args.test_large_batch]):
        all_results["training"] = benchmark_training_simulation(device=device)

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"memory-pool-realistic-scenarios-{timestamp}.md"

    with open(report_path, "w") as f:
        f.write("# Memory Pool Integration - Realistic Scenarios Benchmark\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}Z\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- PyTorch Version: {torch.__version__}\n\n")

        # Training results
        if "training" in all_results:
            training = all_results["training"]
            f.write("## Training Simulation Results\n\n")
            f.write(f"- Total Steps: {training['total_steps']}\n")
            f.write(
                f"- Time per Step: without_pool={training['without_pool']['time_per_step']:.4f}s, with_pool={training['with_pool']['time_per_step']:.4f}s\n"
            )
            f.write(
                f"- Peak Memory: without_pool={training['without_pool']['peak_memory_mb']:.1f}MB, with_pool={training['with_pool']['peak_memory_mb']:.1f}MB\n"
            )
            f.write(
                f"- **Time Improvement**: {training['improvements']['time_improvement_percent']:.1f}%\n"
            )
            f.write(
                f"- **Memory Improvement**: {training['improvements']['memory_improvement_percent']:.1f}%\n\n"
            )

        # Variable sequence results
        if "variable_seq" in all_results:
            var_seq = all_results["variable_seq"]
            f.write("## Variable Sequence Length Results\n\n")
            f.write(f"- Sequences Processed: {var_seq['num_sequences']}\n")
            f.write(f"- Sequence Lengths: {var_seq['sequence_lengths']}\n")
            f.write(
                f"- Time per Sequence: without_pool={var_seq['without_pool']['time_per_sequence']:.4f}s, with_pool={var_seq['with_pool']['time_per_sequence']:.4f}s\n"
            )
            f.write(
                f"- **Time Improvement**: {var_seq['improvements']['time_improvement_percent']:.1f}%\n"
            )
            f.write(
                f"- **Memory Improvement**: {var_seq['improvements']['memory_improvement_percent']:.1f}%\n\n"
            )

        # Large batch results
        if "large_batch" in all_results:
            large_batch = all_results["large_batch"]
            f.write("## Large Batch Scenario Results\n\n")
            f.write(
                "| Batch Size | Without Pool (ms) | With Pool (ms) | Time Improvement | Memory Improvement |\n"
            )
            f.write(
                "|------------|-------------------|----------------|------------------|--------------------|\n"
            )

            for scenario in large_batch["scenarios"]:
                if (
                    scenario["without_pool"]["success"]
                    and scenario["with_pool"]["success"]
                ):
                    batch_size = scenario["batch_size"]
                    no_pool_time = scenario["without_pool"]["time_per_iteration"] * 1000
                    with_pool_time = scenario["with_pool"]["time_per_iteration"] * 1000
                    time_imp = scenario["improvements"]["time_improvement_percent"]
                    memory_imp = scenario["improvements"]["memory_improvement_percent"]
                    f.write(
                        f"| {batch_size} | {no_pool_time:.2f} | {with_pool_time:.2f} | {time_imp:.1f}% | {memory_imp:.1f}% |\n"
                    )
            f.write("\n")

        f.write("## Key Findings\n\n")

        # Analyze results
        if "training" in all_results:
            time_imp = all_results["training"]["improvements"][
                "time_improvement_percent"
            ]
            if time_imp > 5:
                f.write(
                    f"- ✅ **Training Performance**: {time_imp:.1f}% faster with memory pool\n"
                )
            elif time_imp > -5:
                f.write(
                    f"- ✅ **Training Performance**: Negligible impact ({time_imp:.1f}%)\n"
                )
            else:
                f.write(
                    f"- ⚠️ **Training Performance**: {abs(time_imp):.1f}% slower with pool overhead\n"
                )

        f.write("\n### Memory Pool Benefits:\n")
        f.write("- Enhanced tensor allocation with strategy selection\n")
        f.write("- Automatic memory pool management and reuse\n")
        f.write("- NUMA-aware allocation for multi-socket systems\n")
        f.write("- Fragment-aware allocation to reduce fragmentation\n")
        f.write("- Optional memory profiling and monitoring\n")

    print(f"\nBenchmark results saved to: {report_path}")

    # Print summary
    print("\nRealistic Scenarios Summary:")
    print("=" * 30)

    if "training" in all_results:
        training = all_results["training"]
        print(
            f"Training time improvement: {training['improvements']['time_improvement_percent']:.1f}%"
        )
        print(
            f"Training memory improvement: {training['improvements']['memory_improvement_percent']:.1f}%"
        )


if __name__ == "__main__":
    main()
