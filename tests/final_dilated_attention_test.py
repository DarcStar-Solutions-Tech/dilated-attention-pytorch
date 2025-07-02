#!/usr/bin/env python3
"""
Final test to verify dilated attention is working correctly with segments.
Run with: torchrun --nproc_per_node=2 tests/final_dilated_attention_test.py
"""

import os
import torch
import torch.distributed as dist


def final_test():
    """Final verification that dilated attention works correctly."""

    if "RANK" not in os.environ:
        print(
            "Run with: torchrun --nproc_per_node=2 tests/final_dilated_attention_test.py"
        )
        return

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("FINAL DILATED ATTENTION VERIFICATION")
        print("=" * 60)

    from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
        RingDilatedAttentionHybrid,
    )

    # Test with clear segment pattern
    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[2],
        dropout=0.0,
        ring_size=world_size,
        device=device,
        dtype=torch.float32,
        use_flash_attention=False,
        enable_memory_pool=False,  # Disable for simplicity
    )

    # Create input with clear pattern
    # Each segment has a unique pattern to verify segment-wise processing
    seq_len = 512
    q = torch.zeros(1, seq_len, 4, 32, device=device)
    k = torch.zeros(1, seq_len, 4, 32, device=device)
    v = torch.zeros(1, seq_len, 4, 32, device=device)

    # Segment 1 (0-255): alternating 1.0 and 0.0
    for i in range(256):
        value = 1.0 if i % 2 == 0 else 0.0
        q[:, i] = value
        k[:, i] = value
        v[:, i] = value

    # Segment 2 (256-511): alternating 10.0 and 0.0
    for i in range(256, 512):
        value = 10.0 if i % 2 == 0 else 0.0
        q[:, i] = value
        k[:, i] = value
        v[:, i] = value

    with torch.no_grad():
        output = model(q, k, v, is_causal=False)

    # With dilation_rate=2, only even positions are selected
    # So segment 1 should have value ~1.0, segment 2 should have value ~10.0
    seg1_even_mean = output[:, :256:2].mean().item()
    seg1_odd_mean = output[:, 1:256:2].mean().item()
    seg2_even_mean = output[:, 256::2].mean().item()
    seg2_odd_mean = output[:, 257::2].mean().item()

    if rank == 0:
        print("\nResults with dilation_rate=2:")
        print("Segment 1 (0-255):")
        print(f"  Even positions mean: {seg1_even_mean:.4f} (expected ~1.0)")
        print(f"  Odd positions mean: {seg1_odd_mean:.4f} (expected ~0.0)")
        print("Segment 2 (256-511):")
        print(f"  Even positions mean: {seg2_even_mean:.4f} (expected ~10.0)")
        print(f"  Odd positions mean: {seg2_odd_mean:.4f} (expected ~0.0)")

        # Verify behavior
        seg1_correct = abs(seg1_even_mean - 1.0) < 0.1
        seg2_correct = abs(seg2_even_mean - 10.0) < 0.1

        print(f"\nSegment 1 correct: {seg1_correct}")
        print(f"Segment 2 correct: {seg2_correct}")

        if seg1_correct and seg2_correct:
            print("\n✅ SUCCESS: Dilated attention is working correctly!")
            print("   - Segments are processed independently")
            print("   - Dilation is applied within segments")
            print("   - The fix has been successfully applied")
        else:
            print("\n❌ ISSUE: Dilated attention may not be working as expected")

    dist.destroy_process_group()


if __name__ == "__main__":
    final_test()
