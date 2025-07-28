#!/usr/bin/env python3
"""Debug dilated attention dimension mismatch."""

import torch

# Test parameters
batch_size = 2
seq_len = 1024
num_heads = 8
head_dim = 64
segment_lengths = [256, 512]
dilation_rates = [1, 2]

# Create dummy inputs
q = torch.randn(batch_size, seq_len, num_heads, head_dim)
k = torch.randn_like(q)
v = torch.randn_like(q)

print(f"Input shape: {q.shape}")
print(f"Segment lengths: {segment_lengths}")
print(f"Dilation rates: {dilation_rates}")

# Test segment processing
for seg_idx, (seg_len, dilation_rate) in enumerate(
    zip(segment_lengths, dilation_rates)
):
    print(f"\n--- Segment {seg_idx}: length={seg_len}, dilation={dilation_rate} ---")

    if seq_len < seg_len:
        seg_len = seq_len
        print(f"Adjusted segment length to {seg_len}")

    # Number of segments
    num_segments = (seq_len + seg_len - 1) // seg_len
    print(f"Number of segments: {num_segments}")

    for segment_idx in range(num_segments):
        # Calculate segment boundaries
        start_idx = segment_idx * seg_len
        end_idx = min(start_idx + seg_len, seq_len)
        actual_seg_len = end_idx - start_idx

        print(
            f"\n  Segment {segment_idx}: [{start_idx}:{end_idx}], length={actual_seg_len}"
        )

        # Get segment queries
        q_seg = q[:, start_idx:end_idx]
        print(f"  Q segment shape: {q_seg.shape}")

        # For dilated attention
        if dilation_rate > 1:
            # Method 1: Original (problematic)
            dilated_indices_old = torch.arange(start_idx, seq_len, dilation_rate)[
                :actual_seg_len
            ]
            print(
                f"  Old dilated indices: {dilated_indices_old.shape}, values: {dilated_indices_old[:10].tolist()}..."
            )

            # Method 2: Fixed
            dilated_indices = []
            for i in range(actual_seg_len):
                idx = start_idx + i * dilation_rate
                if idx < seq_len:
                    dilated_indices.append(idx)

            if dilated_indices:
                dilated_indices = torch.tensor(dilated_indices, dtype=torch.long)
                print(
                    f"  New dilated indices: {dilated_indices.shape}, values: {dilated_indices[:10].tolist()}..."
                )

                k_seg = k[:, dilated_indices]
                v_seg = v[:, dilated_indices]
                print(f"  K/V segment shape: {k_seg.shape}")
            else:
                print("  No valid dilated indices!")
        else:
            k_seg = k[:, start_idx:end_idx]
            v_seg = v[:, start_idx:end_idx]
            print(f"  K/V segment shape: {k_seg.shape}")

        # Check if dimensions match
        if q_seg.shape[1] != k_seg.shape[1]:
            print(
                f"  ERROR: Dimension mismatch! Q: {q_seg.shape[1]} vs K: {k_seg.shape[1]}"
            )
