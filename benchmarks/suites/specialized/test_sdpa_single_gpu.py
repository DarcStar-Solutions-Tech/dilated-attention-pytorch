#!/usr/bin/env python3
"""
Test Ring Dilated Attention with SDPA on single GPU.

This tests the implementation without distributed communication.
"""

import torch
import gc
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dilated_attention_pytorch.ring_dilated_attention_sdpa import (
    RingDilatedAttentionSDPA,
)


def test_single_gpu():
    """Test SDPA implementation on single GPU."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Testing Ring Dilated Attention SDPA - Single GPU")
    print("=" * 80)
    print(f"Device: {device}")

    # Test different sequence lengths
    test_configs = [
        (8192, [2048, 4096], [1, 2], "8K tokens"),
        (16384, [2048, 4096], [1, 2], "16K tokens"),
        (32768, [2048, 4096, 8192], [1, 2, 4], "32K tokens"),
        (65536, [2048, 4096, 8192], [1, 2, 4], "64K tokens"),
        (100000, [2048, 4096, 8192], [1, 2, 4], "100K tokens"),
        (150000, [2048, 4096, 8192], [1, 4, 8], "150K tokens"),
    ]

    for seq_len, seg_lengths, dil_rates, label in test_configs:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        try:
            print(f"\nTesting {label}...")

            # Create input
            x = torch.randn(1, seq_len, 768, device=device, dtype=torch.float32)

            # Create model
            model = RingDilatedAttentionSDPA(
                embed_dim=768,
                num_heads=12,
                segment_lengths=seg_lengths,
                dilation_rates=dil_rates,
                dropout=0.0,
                device=device,
            )
            model.eval()

            # Forward pass
            with torch.no_grad():
                output = model(x)
                torch.cuda.synchronize()

            # Get memory
            mem_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            mem_per_token = mem_mb / seq_len

            print(f"✓ {label} - Success!")
            print(f"  Memory: {mem_mb:.1f} MB ({mem_per_token:.4f} MB/token)")
            print(f"  Output shape: {output.shape}")

            # Cleanup
            del x, model, output
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ {label} - Failed: {e}")
            break

    print("\n" + "=" * 80)
    print("Single GPU test completed!")


if __name__ == "__main__":
    test_single_gpu()
