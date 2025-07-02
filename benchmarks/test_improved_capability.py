#!/usr/bin/env python3
"""
Test the actual capability of ImprovedDilatedAttention on Pascal GPU.
"""

import torch
import gc
from dilated_attention_pytorch.improved_dilated_attention import (
    ImprovedDilatedAttention,
)
from dilated_attention_pytorch.core import DilatedAttentionConfig


def test_improved_max_sequence():
    """Test maximum sequence capability of improved implementation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Testing ImprovedDilatedAttention on {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Test with different configurations
    configs = [
        # Small config
        {"segment_lengths": [2048], "dilation_rates": [1], "start_seq": 16384},
        # Medium config
        {"segment_lengths": [2048, 4096], "dilation_rates": [1, 2], "start_seq": 32768},
        # Large config
        {
            "segment_lengths": [2048, 4096, 8192],
            "dilation_rates": [1, 2, 4],
            "start_seq": 65536,
        },
    ]

    for i, cfg in enumerate(configs):
        print(
            f"\n--- Config {i + 1}: segments={cfg['segment_lengths']}, dilation={cfg['dilation_rates']} ---"
        )

        # Create attention config
        attention_config = DilatedAttentionConfig(
            segment_lengths=cfg["segment_lengths"],
            dilation_rates=cfg["dilation_rates"],
            dropout=0.0,
        )

        current = cfg["start_seq"]
        max_working = 0

        while current <= 512000:
            # Ensure divisibility
            max_seg = max(cfg["segment_lengths"])
            if current % max_seg != 0:
                current = ((current // max_seg) + 1) * max_seg

            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # Create model with direct arguments
                model = ImprovedDilatedAttention(
                    segment_lengths=cfg["segment_lengths"],
                    dilation_rates=cfg["dilation_rates"],
                    dropout=0.0,
                    use_xformers=True,
                    use_flex_attention=False,  # Pascal doesn't support
                )

                # Test inputs
                batch_size = 1
                num_heads = 8
                head_dim = 64

                q = torch.randn(batch_size, current, num_heads, head_dim, device=device)
                k = torch.randn(batch_size, current, num_heads, head_dim, device=device)
                v = torch.randn(batch_size, current, num_heads, head_dim, device=device)

                # Forward
                with torch.no_grad():
                    output = model(q, k, v)

                # Memory
                peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                kb_per_token = peak_mb * 1024 / current

                print(
                    f"  {current:,} tokens: Success - {peak_mb:.0f} MB, {kb_per_token:.1f} KB/token"
                )
                max_working = current

                # Cleanup
                del q, k, v, output, model
                gc.collect()
                torch.cuda.empty_cache()

                # Next size
                if current < 100000:
                    current += 16384
                elif current < 200000:
                    current += 32768
                else:
                    current += 65536

            except torch.cuda.OutOfMemoryError:
                print(f"  {current:,} tokens: OOM")
                break
            except Exception as e:
                print(f"  {current:,} tokens: Error - {type(e).__name__}: {str(e)}")
                break

        print(f"  Maximum: {max_working:,} tokens")

    print("\n" + "=" * 60)

    # Test the claim of 250K tokens
    print("\nTesting 250K+ token claim with optimal config...")

    # Use configuration that should work based on the claim
    attention_config = DilatedAttentionConfig(
        segment_lengths=[4096, 8192, 16384, 32768],
        dilation_rates=[1, 2, 4, 8],
        dropout=0.0,
    )

    test_sequences = [131072, 196608, 262144, 327680]

    for seq_len in test_sequences:
        # Ensure divisibility
        if seq_len % 32768 != 0:
            seq_len = ((seq_len // 32768) + 1) * 32768

        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = ImprovedDilatedAttention(
                segment_lengths=attention_config.segment_lengths,
                dilation_rates=attention_config.dilation_rates,
                dropout=0.0,
                use_xformers=True,
                use_flex_attention=False,
            )

            # Minimal test
            q = torch.randn(1, seq_len, 8, 64, device=device)
            k = torch.randn(1, seq_len, 8, 64, device=device)
            v = torch.randn(1, seq_len, 8, 64, device=device)

            with torch.no_grad():
                output = model(q, k, v)

            peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
            kb_per_token = peak_mb * 1024 / seq_len

            print(
                f"{seq_len:,} tokens: Success - {peak_mb:.0f} MB ({kb_per_token:.1f} KB/token)"
            )

            del q, k, v, output, model
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:,} tokens: OOM")
        except Exception as e:
            print(f"{seq_len:,} tokens: Error - {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    test_improved_max_sequence()
