#!/usr/bin/env python3
"""
Test maximum sequence lengths for different Ring Attention implementations.
Progressively increases sequence length until OOM or other failure.
"""

import torch
import gc
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dilated_attention_pytorch.ring_dilated_attention_production_fixed import (
    RingDilatedAttentionProductionFixed,
)
from src.dilated_attention_pytorch.ring_dilated_attention_hilbert_optimized_fixed import (
    RingDilatedAttentionHilbertOptimizedFixed,
)
from src.dilated_attention_pytorch.block_sparse_ring_dilated_attention_fixed import (
    BlockSparseRingDilatedAttentionFixed,
)
from src.dilated_attention_pytorch.core.standardized_api import StandardizedRingConfig


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return allocated, reserved, total
    return 0, 0, 0


def test_sequence_length(model_type, max_seq_len=1_000_000, batch_size=1):
    """Test how far we can push sequence length for a given model type."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_type}")
    print(f"{'=' * 60}")

    # Test configurations - exponentially increasing sequence lengths
    test_lengths = [
        1024,  # 1K
        2048,  # 2K
        4096,  # 4K
        8192,  # 8K
        16384,  # 16K
        32768,  # 32K
        65536,  # 64K
        131072,  # 128K
        262144,  # 256K
        524288,  # 512K
        1048576,  # 1M
        2097152,  # 2M
        4194304,  # 4M
    ]

    # Filter to reasonable range
    test_lengths = [l for l in test_lengths if l <= max_seq_len]

    # Fixed parameters
    num_heads = 8
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for seq_len in test_lengths:
        print(f"\nTesting sequence length: {seq_len:,}")

        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Get initial memory
        alloc_before, _, total_mem = get_gpu_memory_info()

        try:
            # Calculate appropriate segment lengths
            if seq_len <= 8192:
                segment_lengths = [min(2048, seq_len)]
                dilation_rates = [1]
            elif seq_len <= 32768:
                segment_lengths = [2048, 4096]
                dilation_rates = [1, 2]
            elif seq_len <= 131072:
                segment_lengths = [2048, 4096, 8192]
                dilation_rates = [1, 2, 4]
            else:
                segment_lengths = [4096, 8192, 16384]
                dilation_rates = [1, 2, 4]

            # Adjust segment lengths if they exceed sequence length
            segment_lengths = [min(s, seq_len) for s in segment_lengths]

            print(f"  Segment lengths: {segment_lengths}")
            print(f"  Dilation rates: {dilation_rates}")

            # Create model configuration
            if model_type == "hilbert":
                config = StandardizedRingConfig(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    use_hilbert=True,
                    hilbert_chunk_size=min(max(segment_lengths), 16384),
                )
                model = RingDilatedAttentionHilbertOptimizedFixed(config=config).to(
                    device
                )
            elif model_type == "production":
                config = StandardizedRingConfig(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                )
                model = RingDilatedAttentionProductionFixed(config=config).to(device)
            elif model_type == "block_sparse":
                config = StandardizedRingConfig(
                    dim=head_dim,
                    heads=num_heads,
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    dropout=0.0,
                    sparsity_ratio=0.1,  # 90% sparse
                    block_size=min(128, seq_len // 32),
                )
                model = BlockSparseRingDilatedAttentionFixed(config=config).to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Create inputs
            print("  Creating inputs...")
            q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)

            # Get memory after allocation
            alloc_after_input, _, _ = get_gpu_memory_info()
            input_memory = alloc_after_input - alloc_before
            print(f"  Input memory: {input_memory:.2f} GB")

            # Forward pass
            print("  Running forward pass...")
            with torch.cuda.amp.autocast(enabled=False):  # Use FP32
                output = model(q, k, v, is_causal=False)

            # Get peak memory
            alloc_peak, _, _ = get_gpu_memory_info()
            peak_memory = alloc_peak - alloc_before

            # Check output validity
            assert output.shape == (batch_size, seq_len, num_heads, head_dim)
            assert not torch.isnan(output).any()

            print("  ✓ Success!")
            print(f"  Peak memory: {peak_memory:.2f} GB")
            print(f"  Memory efficiency: {seq_len / peak_memory / 1e6:.2f} M tokens/GB")

            results.append(
                {
                    "seq_len": seq_len,
                    "success": True,
                    "peak_memory_gb": peak_memory,
                    "tokens_per_gb": seq_len / peak_memory / 1e6,
                }
            )

            # Clean up
            del model, q, k, v, output

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ OOM at {seq_len:,} tokens")
                results.append(
                    {
                        "seq_len": seq_len,
                        "success": False,
                        "error": "OOM",
                    }
                )
                break
            else:
                print(f"  ✗ Error: {e}")
                results.append(
                    {
                        "seq_len": seq_len,
                        "success": False,
                        "error": str(e),
                    }
                )
                # Try to continue with next size

        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append(
                {
                    "seq_len": seq_len,
                    "success": False,
                    "error": str(e),
                }
            )
            # Try to continue

    return results


def main():
    """Test all implementations."""
    print("=" * 80)
    print("SEQUENCE LENGTH LIMIT TESTING")
    print("=" * 80)

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        _, _, total_mem = get_gpu_memory_info()
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_mem:.1f} GB")
    else:
        print("Running on CPU (will be very limited)")

    # Test each implementation
    all_results = {}

    for model_type in ["hilbert", "production", "block_sparse"]:
        results = test_sequence_length(model_type, max_seq_len=10_000_000)
        all_results[model_type] = results

        # Find maximum successful length
        max_len = 0
        for r in results:
            if r["success"]:
                max_len = max(max_len, r["seq_len"])

        print(f"\n{model_type.upper()} Maximum sequence length: {max_len:,}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for model_type, results in all_results.items():
        successful = [r for r in results if r["success"]]
        if successful:
            max_result = max(successful, key=lambda x: x["seq_len"])
            print(f"\n{model_type.upper()}:")
            print(f"  Max sequence length: {max_result['seq_len']:,}")
            print(f"  Peak memory used: {max_result['peak_memory_gb']:.2f} GB")
            print(f"  Efficiency: {max_result['tokens_per_gb']:.2f} M tokens/GB")
        else:
            print(f"\n{model_type.upper()}: No successful runs")

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    filename = f"sequence_length_limits_{timestamp}.json"

    import json

    with open(filename, "w") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "gpu": torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "CPU",
                    "total_memory_gb": total_mem if torch.cuda.is_available() else 0,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
