#!/usr/bin/env python3
"""
Simple test of SDPA without ring communication.
"""

import torch
import torch.nn.functional as F
import gc


def test_sdpa_memory():
    """Test SDPA memory usage with different configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing SDPA memory usage...")
    print("-" * 60)

    # Test different sequence lengths
    for seq_len in [8192, 16384, 32768, 65536, 100000]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Test with dilated attention - only process subset of positions
            dilation_rate = 4  # Process every 4th position
            dilated_len = seq_len // dilation_rate

            # Create tensors - using FP16 for memory efficiency
            q = torch.randn(1, 12, dilated_len, 64, device=device, dtype=torch.float16)
            k = torch.randn(1, 12, dilated_len, 64, device=device, dtype=torch.float16)
            v = torch.randn(1, 12, dilated_len, 64, device=device, dtype=torch.float16)

            # Use SDPA
            with torch.no_grad():
                output = F.scaled_dot_product_attention(q, k, v)

            # Check memory
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_per_token = mem_mb / dilated_len

            print(
                f"✓ {seq_len:,} tokens (dilated to {dilated_len:,}): "
                f"{mem_mb:.1f} MB ({mem_per_token:.4f} MB/token)"
            )

            del q, k, v, output

        except torch.cuda.OutOfMemoryError:
            print(f"✗ {seq_len:,} tokens: OOM")
            break
        except Exception as e:
            print(f"✗ {seq_len:,} tokens: {e}")
            break

    # Now test what sequence length we can handle without dilation
    print("\nTesting without dilation (full attention):")
    print("-" * 60)

    for seq_len in [4096, 8192, 16384, 32768]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            # Full attention
            q = torch.randn(1, 12, seq_len, 64, device=device, dtype=torch.float16)
            k = torch.randn(1, 12, seq_len, 64, device=device, dtype=torch.float16)
            v = torch.randn(1, 12, seq_len, 64, device=device, dtype=torch.float16)

            with torch.no_grad():
                output = F.scaled_dot_product_attention(q, k, v)

            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            print(f"✓ {seq_len:,} tokens: {mem_mb:.1f} MB")

            del q, k, v, output

        except torch.cuda.OutOfMemoryError:
            print(f"✗ {seq_len:,} tokens: OOM")
            break


def estimate_max_tokens():
    """Estimate maximum tokens for 2 GPUs with ring attention."""
    print("\nEstimating maximum tokens with ring attention + dilation:")
    print("-" * 60)

    # Assumptions:
    # - 2 GPUs with ~8GB each
    # - Using FP16 (2 bytes per element)
    # - 12 heads, 64 dim per head
    # - Dilation rate of 4-8

    available_memory_gb = 7.5  # Conservative estimate
    overhead_gb = 1.0  # Model weights, activations, etc.
    usable_memory_gb = available_memory_gb - overhead_gb

    # Memory per token (empirical from above tests)
    mem_per_token_mb = 0.015  # With SDPA and FP16

    # With dilation rate of 4
    dilation_rate = 4

    # Per GPU
    tokens_per_gpu = int((usable_memory_gb * 1024) / mem_per_token_mb)
    total_tokens_dilated = tokens_per_gpu * 2  # 2 GPUs
    total_tokens_original = total_tokens_dilated * dilation_rate

    print(f"Available memory per GPU: {available_memory_gb} GB")
    print(f"Usable memory (after overhead): {usable_memory_gb} GB")
    print(f"Memory per dilated token: {mem_per_token_mb} MB")
    print(f"Dilation rate: {dilation_rate}")
    print("\nEstimated capacity:")
    print(f"  Dilated tokens per GPU: {tokens_per_gpu:,}")
    print(f"  Total dilated tokens (2 GPUs): {total_tokens_dilated:,}")
    print(f"  Original sequence length: {total_tokens_original:,}")
    print(
        f"\n{'✅' if total_tokens_original >= 200000 else '❌'} "
        f"Can handle 200K tokens: {total_tokens_original >= 200000}"
    )


if __name__ == "__main__":
    test_sdpa_memory()
    estimate_max_tokens()
