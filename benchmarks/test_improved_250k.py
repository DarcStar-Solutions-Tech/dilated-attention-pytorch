#!/usr/bin/env python3
"""
Test ImprovedDilatedAttention with configuration that achieves 250K+ tokens.
Based on documented results from implementation comparison.
"""

import torch
import gc
import time

from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)


def test_250k_capability():
    """Test the 250K+ token capability with optimal configuration."""

    device = torch.device("cuda")

    print("Testing ImprovedDilatedAttention 250K+ Capability")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print()

    # Optimal configuration from the documentation
    # This is what achieved 512K tokens in the tests
    segment_lengths = [4096, 8192, 16384, 32768]
    dilation_rates = [1, 2, 4, 8]

    print("Configuration (from documented 512K results):")
    print(f"  Segments: {segment_lengths}")
    print(f"  Dilations: {dilation_rates}")
    print("  dtype: float16 (critical for memory efficiency)")
    print("  Memory pool: Enabled")
    print()

    # Test sequences including the documented 250K+ range
    test_sequences = [
        65536,  # 64K - baseline
        131072,  # 128K
        196608,  # 192K
        262144,  # 256K - the 250K+ claim
        327680,  # 320K
        393216,  # 384K
        458752,  # 448K
        524288,  # 512K - documented maximum
    ]

    max_working = 0

    for seq_len in test_sequences:
        # Ensure divisibility by largest segment
        if seq_len % 32768 != 0:
            seq_len = ((seq_len // 32768) + 1) * 32768

        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            print(f"\nTesting {seq_len:,} tokens...", end="", flush=True)

            # Create model with optimal settings
            model = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                enable_memory_pool=True,  # Critical for long sequences
                lightweight_pool=False,  # Full pool for maximum capability
            )

            # Move to float16 after creation for memory efficiency
            model = model.half()

            # Test inputs - using float16
            batch_size = 1
            num_heads = 8
            head_dim = 64

            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            k = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            v = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float16,
            )

            # Time the forward pass
            start_time = time.time()

            with torch.no_grad():
                output = model(q, k, v)

            torch.cuda.synchronize()
            elapsed = time.time() - start_time

            # Get memory usage
            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_gb = peak_mb / 1024
            kb_per_token = peak_mb * 1024 / seq_len

            print(" SUCCESS!")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Memory: {peak_gb:.2f} GB ({kb_per_token:.1f} KB/token)")

            max_working = seq_len

            # Cleanup
            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(" OOM")
            gc.collect()
            torch.cuda.empty_cache()
            break

        except Exception as e:
            print(f" Error: {type(e).__name__}: {str(e)}")
            gc.collect()
            torch.cuda.empty_cache()
            break

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nMaximum sequence achieved: {max_working:,} tokens")

    if max_working >= 250000:
        print("✅ Successfully demonstrated 250K+ token capability!")
    else:
        print("❌ Did not reach 250K tokens")

    print("\nComparison:")
    print("  Hybrid Ring (single GPU): 335,872 tokens")
    print(f"  Improved Dilated: {max_working:,} tokens")

    if max_working < 250000:
        print("\nPossible reasons for not reaching 250K:")
        print("  - Different GPU architecture (Pascal vs newer)")
        print("  - PyTorch version differences")
        print("  - Memory fragmentation")
        print("  - Background GPU memory usage")

    # Additional test with even more aggressive settings
    if max_working < 250000:
        print("\n" + "=" * 60)
        print("Testing with more aggressive memory optimization...")

        # Try with bfloat16 if available
        if torch.cuda.is_bf16_supported():
            print("Using bfloat16...")
            dtype = torch.bfloat16
        else:
            print("Using float16...")
            dtype = torch.float16

        # Test just 262144 (256K) with maximum optimization
        seq_len = 262144

        try:
            gc.collect()
            torch.cuda.empty_cache()

            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.95)

            model = ImprovedDilatedAttention(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                enable_memory_pool=True,
                lightweight_pool=False,
            ).to(dtype)

            # Minimal test
            q = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype)
            k = q  # Reuse tensor to save memory
            v = q

            with torch.no_grad():
                _ = model(q, k, v)

            peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3

            print("\n✅ 256K tokens successful with aggressive optimization!")
            print(f"   Memory used: {peak_gb:.2f} GB")

        except Exception:
            print("\n❌ 256K tokens still failed with aggressive optimization")


if __name__ == "__main__":
    test_250k_capability()
