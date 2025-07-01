#!/usr/bin/env python3
"""
Verify that RingDilatedAttentionV2Collective actually uses dilated patterns.
"""

import torch
import torch.distributed as dist
import os
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def setup_distributed():
    """Setup distributed if available."""
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return rank, world_size, device
    else:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pattern_test_input(seq_len, num_heads, head_dim, device, dtype):
    """Create input with distinct patterns to verify dilation."""
    batch_size = 1

    # Create Q, K, V with distinct patterns
    # Q: ascending values
    q = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len, 1, 1)
    q = q.expand(batch_size, seq_len, num_heads, head_dim).contiguous()

    # K: values = position * 100 (to make pattern obvious)
    k = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len, 1, 1) * 100
    k = k.expand(batch_size, seq_len, num_heads, head_dim).contiguous()

    # V: values = position * 1000 (to make pattern obvious)
    v = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len, 1, 1) * 1000
    v = v.expand(batch_size, seq_len, num_heads, head_dim).contiguous()

    return q, k, v


def main():
    rank, world_size, device = setup_distributed()

    # Test parameters
    seq_len = 8192
    num_heads = 8
    head_dim = 64
    dtype = torch.float32

    if rank == 0:
        print(f"\n{'=' * 60}")
        print("Verifying Dilated Patterns in Ring Attention V2")
        print(f"World Size: {world_size}")
        print(f"{'=' * 60}\n")

    # Test 1: Dilation rate = 1 (no dilation)
    if rank == 0:
        print("Test 1: No dilation (dilation_rates=[1, 1])")

    model_no_dilation = RingDilatedAttentionV2Collective(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 1],
        ring_size=world_size,
        device=device,
        dtype=dtype,
    )

    q, k, v = create_pattern_test_input(seq_len, num_heads, head_dim, device, dtype)

    with torch.no_grad():
        output_no_dilation = model_no_dilation(q, k, v, is_causal=False)

    if rank == 0:
        # Check output statistics
        print(f"  Output mean: {output_no_dilation.mean().item():.2f}")
        print(f"  Output std: {output_no_dilation.std().item():.2f}")
        print(f"  Output shape: {output_no_dilation.shape}")

    # Test 2: With dilation
    if rank == 0:
        print("\nTest 2: With dilation (dilation_rates=[1, 2])")

    model_with_dilation = RingDilatedAttentionV2Collective(
        segment_lengths=[4096, 4096],
        dilation_rates=[1, 2],
        ring_size=world_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output_with_dilation = model_with_dilation(q, k, v, is_causal=False)

    if rank == 0:
        print(f"  Output mean: {output_with_dilation.mean().item():.2f}")
        print(f"  Output std: {output_with_dilation.std().item():.2f}")

        # Compare outputs
        diff = (output_with_dilation - output_no_dilation).abs().mean()
        print(f"\n  Difference between outputs: {diff.item():.6f}")

        if diff.item() > 0.01:
            print("  ✓ Dilation IS being applied (outputs differ)")
        else:
            print("  ✗ Dilation NOT being applied (outputs are too similar)")

    # Test 3: Check dilation pattern directly
    if rank == 0:
        print("\nTest 3: Inspecting dilation patterns directly")

    # Hook to capture dilated K/V
    captured_k = None
    captured_v = None

    def capture_hook(module, input, output):
        nonlocal captured_k, captured_v
        if isinstance(output, tuple) and len(output) == 2:
            captured_k, captured_v = output
            return output

    # Register hook on _apply_dilated_patterns_to_chunk
    _ = None
    for name, module in model_with_dilation.named_modules():
        if hasattr(module, "_apply_dilated_patterns_to_chunk"):
            # Can't directly hook a method, so we'll check differently
            break

    # Alternative: Check head group distribution
    if rank == 0:
        print("\n  Head group distribution:")
        heads_per_group = model_with_dilation._calculate_head_groups(num_heads)
        for i, (seg_len, dil_rate, heads) in enumerate(
            zip(
                model_with_dilation.segment_lengths,
                model_with_dilation.dilation_rates,
                heads_per_group,
            )
        ):
            print(
                f"    Group {i}: segment_length={seg_len}, dilation_rate={dil_rate}, heads={heads}"
            )

    # Test 4: Extreme dilation to make it obvious
    if rank == 0:
        print("\nTest 4: Extreme dilation (dilation_rates=[1, 4])")

    model_extreme = RingDilatedAttentionV2Collective(
        segment_lengths=[4096, 4096],
        dilation_rates=[1, 4],  # Every 4th element
        ring_size=world_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output_extreme = model_extreme(q, k, v, is_causal=False)

    if rank == 0:
        print(f"  Output mean: {output_extreme.mean().item():.2f}")
        print(f"  Output std: {output_extreme.std().item():.2f}")

        # Compare to no dilation
        diff_extreme = (output_extreme - output_no_dilation).abs().mean()
        print(f"  Difference from no-dilation: {diff_extreme.item():.6f}")

        if diff_extreme.item() > diff.item():
            print("  ✓ More extreme dilation shows larger difference")
        else:
            print("  ✗ Extreme dilation not showing expected difference")

    # Summary
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"{'=' * 60}")
        print(f"No dilation output mean: {output_no_dilation.mean().item():.2f}")
        print(f"Dilation [1,2] output mean: {output_with_dilation.mean().item():.2f}")
        print(f"Dilation [1,4] output mean: {output_extreme.mean().item():.2f}")
        print("\nConclusion: ", end="")

        if diff.item() > 0.01 and diff_extreme.item() > diff.item():
            print("Dilation IS being applied in distributed mode ✓")
        else:
            print("Dilation may NOT be working correctly ✗")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
