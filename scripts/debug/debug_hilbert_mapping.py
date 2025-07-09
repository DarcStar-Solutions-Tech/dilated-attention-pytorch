#!/usr/bin/env python3
"""Debug Hilbert mapping generation."""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dilated_attention_pytorch.kernels.hilbert_dilated_attention_triton_fixed import (
    create_hilbert_mapping_fixed,
    generate_hilbert_curve_2d,
)

# Test size 128
size = 128
mapping = create_hilbert_mapping_fixed(size)

print(f"Size: {size}")
print(f"Mapping shape: {mapping.shape}")
print(f"Mapping dtype: {mapping.dtype}")
print(f"Unique values: {torch.unique(mapping).shape[0]}")
print(f"Min value: {mapping.min().item()}")
print(f"Max value: {mapping.max().item()}")
print(f"First 10 values: {mapping[:10].tolist()}")

# Check for duplicates
unique_vals, counts = torch.unique(mapping, return_counts=True)
duplicates = unique_vals[counts > 1]
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} duplicate values")
    for dup in duplicates[:5]:  # Show first 5 duplicates
        positions = torch.where(mapping == dup)[0]
        print(f"  Value {dup.item()} appears at positions: {positions.tolist()[:5]}...")

# Check grid size calculation
grid_size = 1
while grid_size * grid_size < size:
    grid_size *= 2
print(
    f"\nGrid size for {size} elements: {grid_size} (covers {grid_size * grid_size} positions)"
)

# Generate coordinates and check
coords = generate_hilbert_curve_2d(grid_size)
print(f"Generated {len(coords)} coordinates")
print(f"First 10 coords: {coords[:10]}")
