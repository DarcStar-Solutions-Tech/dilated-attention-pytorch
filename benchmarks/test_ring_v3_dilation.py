#!/usr/bin/env python3
"""
Test Ring V3 with dilation_rates > 1 which was mentioned as having dimension mismatch.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_v3_dilation.py
"""

import os
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_v3 import RingDilatedAttentionV3


def test_dilation():
    """Test with different dilation rates."""

    if "RANK" not in os.environ:
        # Single GPU test
        print("Testing Ring V3 with Dilation Rates (Single GPU)")
        print("=" * 50)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test configurations
        configs = [
            ([512], [1], "No dilation"),
            ([512], [2], "Dilation rate 2"),
            ([256, 256], [1, 2], "Mixed dilation"),
        ]

        for segment_lengths, dilation_rates, desc in configs:
            print(f"\n{desc}:")
            print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

            try:
                model = RingDilatedAttentionV3(
                    segment_lengths=segment_lengths,
                    dilation_rates=dilation_rates,
                    use_bucketed=False,
                    device=device,
                    dtype=torch.float32,
                    ring_size=1,
                )

                # Create inputs
                seq_len = 512
                q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
                k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
                v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

                output = model(q, k, v, is_causal=False)
                print(f"  ✅ Success! Shape: {output.shape}")

            except Exception as e:
                print(f"  ❌ Error: {e}")

        print("\nRun with torchrun for multi-GPU test")
        return

    # Multi-GPU test
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Ring V3 with Dilation Rates (Multi-GPU)")
        print("=" * 50)
        print(f"World size: {world_size}")

    # Test configurations
    configs = [
        ([256], [1], "No dilation"),
        ([256], [2], "Dilation rate 2"),
        ([128, 128], [1, 2], "Mixed dilation"),
    ]

    for segment_lengths, dilation_rates, desc in configs:
        if rank == 0:
            print(f"\n{desc}:")
            print(f"  Segments: {segment_lengths}, Dilation: {dilation_rates}")

        dist.barrier()

        try:
            model = RingDilatedAttentionV3(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                use_bucketed=False,
                device=device,
                dtype=torch.float32,
                ring_size=world_size,
            )

            # Create inputs
            seq_len = 256  # Small for quick test
            q = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            k = torch.randn(1, seq_len, 4, 32, device=device) * 0.1
            v = torch.randn(1, seq_len, 4, 32, device=device) * 0.1

            output = model(q, k, v, is_causal=False)

            has_nan = torch.isnan(output).any().item()

            # Gather results
            all_nan = [None] * world_size
            dist.all_gather_object(all_nan, has_nan)

            if rank == 0:
                any_nan = any(all_nan)
                if any_nan:
                    print("  ❌ Output has NaN")
                else:
                    print(f"  ✅ Success! Shape: {output.shape}")

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    dist.destroy_process_group()

    if rank == 0:
        print("\n✅ Testing completed")
        print("\nNote: Dilation > 1 is currently disabled in multi-GPU mode")
        print("See line 180-182 in ring_dilated_attention_v3.py")


if __name__ == "__main__":
    test_dilation()
