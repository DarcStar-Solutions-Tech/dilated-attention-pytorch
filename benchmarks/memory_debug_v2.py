#!/usr/bin/env python3
"""
Debug memory usage in V2 implementation.
"""

import torch


def analyze_memory_usage():
    """Analyze where the memory is going in V2."""

    _ = torch.device("cuda")

    # Test configuration
    seq_len = 1024
    batch_size = 2
    num_heads = 8
    head_dim = 64
    world_size = 2

    # Calculate expected memory
    print("=== Memory Analysis ===")
    print(
        f"Configuration: seq={seq_len}, batch={batch_size}, heads={num_heads}, dim={head_dim}"
    )
    print(f"World size: {world_size} GPUs\n")

    # Input tensors (per GPU)
    input_size = 3 * batch_size * seq_len * num_heads * head_dim * 4  # float32
    print(f"Input tensors (Q,K,V): {input_size / 1024**2:.2f} MB")

    # Expected per-chunk processing
    chunk_size = seq_len // world_size
    kv_chunk_size = 2 * batch_size * chunk_size * num_heads * head_dim * 4
    print(f"K,V chunk size: {kv_chunk_size / 1024**2:.2f} MB")

    # The problem: V2 allocates full output for EACH chunk!
    output_full_size = batch_size * num_heads * seq_len * head_dim * 4  # (b, h, n, d)
    lse_full_size = batch_size * num_heads * seq_len * 4  # (b, h, n)

    print("\nPROBLEM in V2:")
    print(f"Output buffer (full seq): {output_full_size / 1024**2:.2f} MB")
    print(f"LSE buffer (full seq): {lse_full_size / 1024**2:.2f} MB")
    print(
        f"Total per chunk iteration: {(output_full_size + lse_full_size) / 1024**2:.2f} MB"
    )

    # This is allocated INSIDE the chunk loop AND inside segment processing!
    # If we have 2 chunks and process segments, this explodes

    print("\nWith segment processing:")
    segment_lengths = [512, 1024]
    num_segments = sum(seq_len // seg_len for seg_len in segment_lengths)
    print(f"Number of segments: {num_segments}")

    # Each segment processing ALSO allocates full buffers
    total_waste = num_segments * world_size * (output_full_size + lse_full_size)
    print(f"Total wasted memory: {total_waste / 1024**2:.2f} MB")

    # What it SHOULD be
    print("\nWhat it SHOULD use:")
    # Only need one accumulator for the full output
    accumulator_size = output_full_size + lse_full_size
    # Plus temporary buffers for chunk processing
    chunk_output_size = batch_size * num_heads * chunk_size * head_dim * 4
    chunk_lse_size = batch_size * num_heads * chunk_size * 4

    proper_total = (
        accumulator_size
        + chunk_output_size
        + chunk_lse_size
        + input_size
        + kv_chunk_size
    )
    print(f"Proper total: {proper_total / 1024**2:.2f} MB")

    # Compare to reported
    reported_mb = 213.6  # From benchmark
    print(f"\nReported usage: {reported_mb} MB")
    print(f"Excess: {reported_mb - (proper_total / 1024**2):.2f} MB")


if __name__ == "__main__":
    analyze_memory_usage()
