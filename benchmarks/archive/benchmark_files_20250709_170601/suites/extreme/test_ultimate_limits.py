#!/usr/bin/env python3
"""
Push to the absolute limits - testing multi-million token sequences.
"""

import torch
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_core_fixed import (
    RingDilatedAttentionHilbertCoreFixed,
)
from benchmarks.core.utils.safety import MemorySafetyChecker, SafetyConfig


def test_ultimate_limits():
    """Test multi-million token sequences."""
    print("\n" + "=" * 80)
    print("ULTIMATE LIMITS TEST - Multi-Million Token Sequences")
    print("=" * 80)

    # Extreme safety config
    safety_config = SafetyConfig(
        max_memory_fraction=0.98,  # Use 98% of GPU
        min_free_memory_gb=0.2,  # Only 200MB free
    )
    safety_checker = MemorySafetyChecker(safety_config)

    # Test configurations for multi-million tokens
    test_configs = [
        {"name": "2 Million", "seq_len": 2_097_152, "batch": 1, "heads": 1},
        {"name": "4 Million", "seq_len": 4_194_304, "batch": 1, "heads": 1},
        {"name": "8 Million", "seq_len": 8_388_608, "batch": 1, "heads": 1},
        {"name": "16 Million", "seq_len": 16_777_216, "batch": 1, "heads": 1},
    ]

    # Also test with optimized parameters
    optimized_configs = [
        {
            "name": "1M Optimized",
            "seq_len": 1_048_576,
            "batch": 1,
            "heads": 4,
            "segments": [4096, 8192, 16384, 32768],
        },
        {
            "name": "2M Optimized",
            "seq_len": 2_097_152,
            "batch": 1,
            "heads": 2,
            "segments": [8192, 16384, 32768, 65536],
        },
    ]

    print("\n--- Testing Multi-Million Token Sequences ---")

    for config in test_configs:
        print(f"\n{config['name']} tokens test:")
        print(f"  Sequence: {config['seq_len']:,} tokens")

        # Memory check
        shape = (config["batch"], config["seq_len"], config["heads"], 64)
        memory_gb = safety_checker.estimate_tensor_memory(shape, torch.float32, 4) * 1.2
        print(f"  Estimated memory: {memory_gb:.1f}GB")

        can_allocate, message = safety_checker.check_memory_available(memory_gb)
        if not can_allocate:
            print(f"  ✗ Skip: {message}")
            continue

        try:
            # Create model with large segments
            segment_lengths = [8192, 16384, 32768, 65536]
            dilation_rates = [1, 2, 4, 8]

            model = RingDilatedAttentionHilbertCoreFixed(
                dim=config["heads"] * 64,
                heads=config["heads"],
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_hilbert=True,
            ).cuda()

            # Allocate minimal tensors
            print("  Allocating tensors...")
            q = torch.randn(
                1,
                min(8192, config["seq_len"]),
                config["heads"],
                64,
                device="cuda",
                dtype=torch.float32,
            )

            # Test with small sequence first
            _ = model(q, q, q)
            del q
            torch.cuda.empty_cache()

            # Now try full sequence
            q = torch.randn(
                config["batch"],
                config["seq_len"],
                config["heads"],
                64,
                device="cuda",
                dtype=torch.float32,
            )
            k = q  # Reuse to save memory
            v = q

            print("  Running forward pass...")
            torch.cuda.synchronize()
            start = time.perf_counter()

            output = model(q, k, v)

            torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed = (end - start) * 1000
            throughput = config["seq_len"] / elapsed * 1000

            print(f"  ✓ SUCCESS! {config['name']} tokens processed!")
            print(f"    Time: {elapsed:.1f}ms ({elapsed / 1000:.1f}s)")
            print(f"    Throughput: {throughput:,.0f} tokens/sec")
            print(f"    Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

            del q, output, model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:100]}")
            torch.cuda.empty_cache()

    print("\n--- Testing with Optimized Configurations ---")

    for config in optimized_configs:
        print(f"\n{config['name']}:")
        try:
            model = RingDilatedAttentionHilbertCoreFixed(
                dim=config["heads"] * 64,
                heads=config["heads"],
                segment_lengths=config["segments"],
                dilation_rates=[1, 2, 4, 8],
                use_hilbert=True,
            ).cuda()

            # Create inputs
            q = torch.randn(
                config["batch"],
                config["seq_len"],
                config["heads"],
                64,
                device="cuda",
                dtype=torch.float32,
            )

            # Time it
            torch.cuda.synchronize()
            start = time.perf_counter()

            output = model(q, q, q)

            torch.cuda.synchronize()
            end = time.perf_counter()

            elapsed = (end - start) * 1000

            print(f"  ✓ Success: {elapsed:.1f}ms")
            print(f"  Throughput: {config['seq_len'] / elapsed * 1000:,.0f} tokens/sec")

            del q, output, model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:100]}")

    # Final test: absolute maximum
    print("\n--- Finding Absolute Maximum Sequence Length ---")

    max_found = 1_048_576  # Start from 1M
    step = 1_048_576  # 1M increments

    while step >= 65536:  # Stop at 64K granularity
        test_len = max_found + step

        # Quick memory check
        memory_gb = (
            safety_checker.estimate_tensor_memory(
                (1, test_len, 1, 64), torch.float32, 3
            )
            * 1.1
        )

        can_allocate, _ = safety_checker.check_memory_available(memory_gb)

        if can_allocate:
            try:
                model = RingDilatedAttentionHilbertCoreFixed(
                    dim=64,
                    heads=1,
                    segment_lengths=[16384, 32768, 65536],
                    dilation_rates=[1, 2, 4],
                ).cuda()

                q = torch.randn(1, test_len, 1, 64, device="cuda", dtype=torch.float32)
                _ = model(q, q, q)

                max_found = test_len
                print(f"  ✓ {test_len:,} tokens possible")

                del q, model
                torch.cuda.empty_cache()

            except Exception:
                step //= 2
        else:
            step //= 2

    print(f"\n🏆 ABSOLUTE MAXIMUM: {max_found:,} tokens!")
    print(
        f"   That's {max_found / 1_000_000:.1f} MILLION tokens in a single forward pass!"
    )


if __name__ == "__main__":
    test_ultimate_limits()
