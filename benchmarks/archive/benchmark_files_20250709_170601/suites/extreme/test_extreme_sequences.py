#!/usr/bin/env python3
"""
Test extreme sequence lengths with the fixed Hilbert implementation.
"""

import torch
import time
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core_fixed import (
    RingDilatedAttentionHilbertCoreFixed,
)
from benchmarks.core.utils.safety import (
    SafetyConfig,
    MemorySafetyChecker,
)


def test_extreme_sequences():
    """Test the absolute limits of sequence length."""
    print("\n" + "=" * 80)
    print("Extreme Sequence Length Test - Fixed Hilbert Implementation")
    print("=" * 80)

    # Safety config for extreme testing
    safety_config = SafetyConfig(
        max_memory_fraction=0.95,  # Use up to 95% GPU memory
        min_free_memory_gb=0.5,  # Only keep 0.5GB free
    )
    safety_checker = MemorySafetyChecker(safety_config)

    # Show system info
    if torch.cuda.is_available():
        used, free, total = safety_checker.get_gpu_memory_info()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {used:.1f}GB used, {free:.1f}GB free, {total:.1f}GB total")

    # Test configurations based on what we found works
    test_configs = [
        {"name": "Quarter Million (256K)", "seq_len": 262_144, "batch": 1, "heads": 8},
        {"name": "Half Million (512K)", "seq_len": 524_288, "batch": 1, "heads": 4},
        {"name": "One Million (1M)", "seq_len": 1_048_576, "batch": 1, "heads": 2},
    ]

    results = []

    for config in test_configs:
        print(f"\n--- Testing {config['name']} tokens ---")
        print(f"Sequence length: {config['seq_len']:,}")
        print(f"Batch size: {config['batch']}, Heads: {config['heads']}")

        # Memory estimate
        shape = (config["batch"], config["seq_len"], config["heads"], 64)
        memory_gb = safety_checker.estimate_tensor_memory(shape, torch.float32, 4) * 1.5
        print(f"Estimated memory: {memory_gb:.1f}GB")

        can_allocate, message = safety_checker.check_memory_available(memory_gb)
        if not can_allocate:
            print(f"Skip: {message}")
            continue

        try:
            # Create model
            model = RingDilatedAttentionHilbertCoreFixed(
                dim=config["heads"] * 64,
                heads=config["heads"],
                segment_lengths=[2048, 4096, 8192, 16384],
                dilation_rates=[1, 2, 4, 8],
                use_hilbert=True,
            )

            if torch.cuda.is_available():
                model = model.cuda()

            # Create inputs
            device = next(model.parameters()).device
            dtype = torch.float32

            print("Allocating tensors...")
            q = torch.randn(
                config["batch"],
                config["seq_len"],
                config["heads"],
                64,
                device=device,
                dtype=dtype,
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            print("Running forward pass...")
            torch.cuda.synchronize()
            start = time.perf_counter()

            output = model(q, k, v)

            torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed = (end - start) * 1000  # ms

            # Get memory stats
            peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9

            result = {
                "name": config["name"],
                "seq_len": config["seq_len"],
                "time_ms": elapsed,
                "memory_gb": peak_memory_gb,
                "throughput": config["seq_len"] / elapsed * 1000,  # tokens/sec
            }
            results.append(result)

            print("✓ Success!")
            print(f"  Time: {elapsed:.1f}ms")
            print(f"  Memory: {peak_memory_gb:.2f}GB")
            print(f"  Throughput: {result['throughput']:.0f} tokens/sec")

            # Cleanup
            del q, k, v, output, model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ Failed: {e}")
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - Maximum Sequence Lengths Achieved")
    print("=" * 80)

    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Sequence length: {result['seq_len']:,} tokens")
        print(f"  Processing time: {result['time_ms']:.1f}ms")
        print(f"  Memory used: {result['memory_gb']:.2f}GB")
        print(f"  Throughput: {result['throughput']:.0f} tokens/sec")

    if results:
        max_result = max(results, key=lambda x: x["seq_len"])
        print(f"\n🏆 Maximum achieved: {max_result['seq_len']:,} tokens!")

    # Save report
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    report_path = Path(__file__).parent / f"extreme-sequences-{timestamp}.txt"

    with open(report_path, "w") as f:
        f.write("Extreme Sequence Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n\n")

        for result in results:
            f.write(f"{result['name']}:\n")
            f.write(f"  Sequence: {result['seq_len']:,} tokens\n")
            f.write(f"  Time: {result['time_ms']:.1f}ms\n")
            f.write(f"  Memory: {result['memory_gb']:.2f}GB\n")
            f.write(f"  Throughput: {result['throughput']:.0f} tokens/sec\n\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    test_extreme_sequences()
