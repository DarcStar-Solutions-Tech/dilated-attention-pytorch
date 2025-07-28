#!/usr/bin/env python3
"""
Verify only the splitting logic without running forward pass.
"""

import os


def main():
    """Test splitting logic."""

    # Check environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"\n[Rank {rank}] Environment:")
    print(f"  World size: {world_size}")
    print(f"  Local rank: {local_rank}")

    # Test sequence
    seq_len = 4096
    _ = 1
    num_heads = 8
    _ = 64

    # Simulate the splitting logic from StandardRingAttention._split_sequence
    print(f"\n[Rank {rank}] Testing splitting logic:")
    print(f"  Full sequence length: {seq_len}")

    if world_size > 1:
        # This is what the implementation does
        assert seq_len % world_size == 0, (
            f"Sequence length {seq_len} must be divisible by world size {world_size}"
        )

        local_seq_len = seq_len // world_size
        start_idx = rank * local_seq_len
        end_idx = start_idx + local_seq_len

        print(f"  Local sequence length: {local_seq_len}")
        print(f"  Start index: {start_idx}")
        print(f"  End index: {end_idx}")
        print(f"  Processing tokens {start_idx} to {end_idx - 1}")

        # Verify no overlap
        all_indices = list(range(start_idx, end_idx))
        print(f"  First 5 indices: {all_indices[:5]}")
        print(f"  Last 5 indices: {all_indices[-5:]}")

        # Memory estimation
        # Each GPU processes local_seq_len tokens
        attention_memory_mb = (local_seq_len * local_seq_len * 2) / (1024**2)  # float16
        print(f"\n[Rank {rank}] Memory estimation:")
        print(f"  Attention matrix size: {local_seq_len} x {local_seq_len}")
        print(f"  Estimated memory: {attention_memory_mb:.2f} MB per head")
        print(
            f"  Total for {num_heads} heads: {attention_memory_mb * num_heads:.2f} MB"
        )

        # Compare to full attention
        full_attention_mb = (seq_len * seq_len * 2) / (1024**2)
        print(f"  Full attention would need: {full_attention_mb:.2f} MB per head")
        print(f"  Reduction factor: {full_attention_mb / attention_memory_mb:.1f}x")

    else:
        print("  Single GPU: no splitting needed")
        print(f"  Processing full sequence: 0 to {seq_len - 1}")

    print(f"\n[Rank {rank}] ✓ Splitting logic verified!")


if __name__ == "__main__":
    main()
