#!/usr/bin/env python3
"""
Focused benchmark to demonstrate 200K+ token capability with ring attention.

This benchmark specifically tests the configurations needed to achieve 200K+ tokens.
"""

import torch
import gc
import os
import sys
import time
from datetime import datetime
from typing import Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_correct import (
    RingDilatedAttentionHilbertOptimizedCorrect,
)
from dilated_attention_pytorch.utils.gpu_utils import get_gpu_info, get_optimal_dtype


def cleanup_memory():
    """Force memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def test_configuration(
    model_class,
    model_name: str,
    total_seq_len: int,
    world_size: int,
    batch_size: int = 1,
    warmup_steps: int = 1,
    measure_steps: int = 3,
) -> Dict:
    """Test a specific configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimal_dtype = get_optimal_dtype(device)

    local_seq_len = total_seq_len // world_size

    print(f"\nTesting {model_name}:")
    print(f"  Total sequence: {total_seq_len:,} tokens")
    print(f"  World size: {world_size} GPUs")
    print(f"  Local sequence: {local_seq_len:,} tokens per GPU")

    cleanup_memory()
    start_mem = get_memory_mb()

    try:
        # Create local input
        x_local = torch.randn(
            batch_size, local_seq_len, 768, device=device, dtype=optimal_dtype
        )
        input_mem = get_memory_mb() - start_mem
        print(f"  Input created: {input_mem:.1f} MB")

        # Create model
        if model_name == "HilbertCore":
            model = model_class(
                dim=768,
                heads=12,
                segment_lengths=[4096, 8192, 16384],
                dilation_rates=[1, 2, 4],
                ring_size=world_size,
                use_hilbert=True,
                use_custom_backward=True,
            )
        else:
            model = model_class(
                embed_dim=768,
                num_heads=12,
                segment_lengths=[4096, 8192, 16384],
                dilation_rates=[1, 2, 4],
                dropout=0.0,
                use_hilbert=True,
                device=device,
                dtype=optimal_dtype,
                memory_efficient=True,
            )

        model = model.to(device)
        model.eval()
        model_mem = get_memory_mb() - start_mem
        print(f"  Model created: {model_mem:.1f} MB total")

        # Warmup
        print("  Running warmup...", end="", flush=True)
        with torch.no_grad():
            for _ in range(warmup_steps):
                _ = model(x_local, total_seq_len=total_seq_len, already_split=True)
        print(" done")

        # Measure
        print("  Measuring performance...", end="", flush=True)
        torch.cuda.synchronize()
        forward_times = []

        with torch.no_grad():
            for i in range(measure_steps):
                start_time = time.perf_counter()

                output = model(x_local, total_seq_len=total_seq_len, already_split=True)

                torch.cuda.synchronize()
                forward_time = time.perf_counter() - start_time
                forward_times.append(forward_time)
                print(f" {i + 1}", end="", flush=True)

        print(" done")

        # Calculate metrics
        avg_forward_time = sum(forward_times) / len(forward_times)
        peak_mem = get_memory_mb()
        memory_used = peak_mem - start_mem

        print("\n  Results:")
        print(f"    Memory used: {memory_used:.1f} MB")
        print(f"    Memory per token: {memory_used / local_seq_len:.4f} MB")
        print(f"    Forward pass: {avg_forward_time * 1000:.1f} ms")
        print(
            f"    Throughput: {total_seq_len / avg_forward_time / 1e6:.3f}M tokens/sec"
        )
        print(f"    ✓ SUCCESS - Can process {total_seq_len:,} tokens!")

        # Cleanup
        del x_local, model, output
        cleanup_memory()

        return {
            "success": True,
            "total_seq_len": total_seq_len,
            "world_size": world_size,
            "local_seq_len": local_seq_len,
            "memory_mb": memory_used,
            "forward_time_ms": avg_forward_time * 1000,
            "throughput_mtoks": total_seq_len / avg_forward_time / 1e6,
        }

    except torch.cuda.OutOfMemoryError:
        print(
            f"\n  ✗ OOM - Cannot process {total_seq_len:,} tokens with {world_size} GPUs"
        )
        return {
            "success": False,
            "total_seq_len": total_seq_len,
            "world_size": world_size,
            "error": "OOM",
        }
    except Exception as e:
        print(f"\n  ✗ Error: {e}")
        return {
            "success": False,
            "total_seq_len": total_seq_len,
            "world_size": world_size,
            "error": str(e),
        }
    finally:
        cleanup_memory()


def find_max_sequence_length(model_class, model_name: str, world_size: int) -> int:
    """Binary search to find maximum sequence length for given world size."""
    print(
        f"\nFinding maximum sequence length for {model_name} with world_size={world_size}..."
    )

    # Start with conservative estimates
    min_seq = 8192
    max_seq = 1024 * 1024  # 1M tokens

    # Binary search
    last_success = 0
    while min_seq <= max_seq:
        mid_seq = (
            (min_seq + max_seq) // 2 // world_size
        ) * world_size  # Ensure divisible

        result = test_configuration(
            model_class, model_name, mid_seq, world_size, measure_steps=1
        )

        if result["success"]:
            last_success = mid_seq
            min_seq = mid_seq + world_size
        else:
            max_seq = mid_seq - world_size

    return last_success


def main():
    """Run focused 200K+ token benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("CUDA not available!")
        return

    gpu_info = get_gpu_info(device)

    print("=" * 80)
    print("200K+ Token Ring Attention Benchmark")
    print("=" * 80)
    print(f"GPU: {gpu_info.name} ({gpu_info.architecture})")
    print(f"Compute capability: {gpu_info.compute_capability}")
    print(f"Total memory: {gpu_info.total_memory_gb:.1f} GB")
    print(f"Available memory: {gpu_info.available_memory_gb:.1f} GB")
    print(f"Optimal dtype: {get_optimal_dtype(device)}")
    print()

    # Test configurations specifically targeting 200K+ tokens
    test_configs = [
        # Start with configurations likely to succeed
        (RingDilatedAttentionHilbertOptimizedCorrect, "HilbertOptimizedCorrect"),
    ]

    # First, find what world sizes we need for 200K
    print("=" * 80)
    print("Testing 200K token capability")
    print("=" * 80)

    results = []

    for model_class, model_name in test_configs:
        # Test specific world sizes for 200K
        for world_size in [2, 4, 8]:
            print(f"\n{'=' * 60}")
            print(f"Testing {model_name} with {world_size} GPUs for 200K tokens")
            print(f"{'=' * 60}")

            # Test 200K specifically
            result = test_configuration(
                model_class,
                model_name,
                204800,  # 200K tokens (divisible by 2, 4, 8)
                world_size,
            )
            results.append(result)

            if result["success"]:
                print(f"\n🎉 SUCCESS! Can process 200K+ tokens with {world_size} GPUs!")
                break

    # Now find maximum capability
    print("\n" + "=" * 80)
    print("Finding maximum sequence lengths")
    print("=" * 80)

    max_lengths = {}
    for model_class, model_name in test_configs:
        print(f"\n{model_name}:")
        for world_size in [1, 2, 4, 8]:
            max_len = find_max_sequence_length(model_class, model_name, world_size)
            if max_len > 0:
                max_lengths[(model_name, world_size)] = max_len
                print(f"  World size {world_size}: Max = {max_len:,} tokens")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Check 200K capability
    success_200k = [r for r in results if r["success"] and r["total_seq_len"] >= 200000]

    if success_200k:
        print("\n✅ 200K+ Token Processing Confirmed!")
        print("-" * 40)
        for r in success_200k:
            print(f"World size {r['world_size']}: {r['total_seq_len']:,} tokens")
            print(
                f"  Memory: {r['memory_mb']:.1f} MB ({r['local_seq_len']:,} tokens/GPU)"
            )
            print(f"  Speed: {r['throughput_mtoks']:.3f}M tokens/sec")
    else:
        print("\n❌ Could not achieve 200K+ tokens with available memory")

    # Maximum capabilities
    print("\nMaximum Sequence Lengths:")
    print("-" * 40)
    for (model_name, world_size), max_len in sorted(max_lengths.items()):
        print(f"{model_name} (world_size={world_size}): {max_len:,} tokens")

    # Memory efficiency
    print("\nMemory Efficiency (O(n/k) verification):")
    print("-" * 40)

    # Check if memory scales with local sequence length
    for model_name in set(k[0] for k in max_lengths.keys()):
        model_results = [
            (ws, ml) for (mn, ws), ml in max_lengths.items() if mn == model_name
        ]
        if len(model_results) >= 2:
            print(f"\n{model_name}:")
            base_ws, base_len = model_results[0]
            for ws, max_len in model_results[1:]:
                # Local sequence lengths
                base_local = base_len // base_ws
                curr_local = max_len // ws

                # If O(n/k) holds, similar local lengths should be achievable
                ratio = curr_local / base_local
                print(
                    f"  WS {base_ws} → {ws}: Local seq {base_local:,} → {curr_local:,} "
                    f"({ratio:.2f}x) {'✓' if 0.8 <= ratio <= 1.2 else '✗'}"
                )

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M-UTC")
    summary_file = f"benchmark-200k-summary-{timestamp}.txt"

    with open(summary_file, "w") as f:
        f.write("200K+ Token Ring Attention Benchmark Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"GPU: {gpu_info.name}\n")
        f.write(f"Memory: {gpu_info.total_memory_gb:.1f} GB\n")
        f.write("\n")

        if success_200k:
            f.write("✅ Successfully processed 200K+ tokens!\n")
            for r in success_200k:
                f.write(f"\nWorld size {r['world_size']}:\n")
                f.write(f"  Sequence: {r['total_seq_len']:,} tokens\n")
                f.write(f"  Memory: {r['memory_mb']:.1f} MB\n")
                f.write(f"  Speed: {r['throughput_mtoks']:.3f}M tokens/sec\n")

        f.write("\nMaximum sequence lengths:\n")
        for (model_name, world_size), max_len in sorted(max_lengths.items()):
            f.write(f"  {model_name} (WS={world_size}): {max_len:,} tokens\n")

    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    main()
