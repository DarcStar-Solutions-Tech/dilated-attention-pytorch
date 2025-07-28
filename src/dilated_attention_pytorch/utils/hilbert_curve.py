"""
Hilbert Space-Filling Curve utilities for improved cache locality.
"""

import numpy as np
from typing import List


def generate_hilbert_indices(n: int) -> List[int]:
    """
    Generate indices for Hilbert curve traversal.

    Args:
        n: Number of levels (curve will have 2^n x 2^n points)

    Returns:
        List of indices in Hilbert curve order
    """
    # Validate input
    if n <= 0:
        raise ValueError(f"Number of levels must be positive, got {n}")
    if n > 16:  # Prevent overflow for very large curves
        raise ValueError(f"Number of levels too large (max 16), got {n}")

    def hilbert_index_to_xy(index: int, n: int) -> tuple:
        """Convert Hilbert index to (x, y) coordinates."""
        x = y = 0
        s = 1

        while s < 2**n:
            rx = 1 & (index // 2)
            ry = 1 & (index ^ rx)

            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x

            x += s * rx
            y += s * ry
            index //= 4
            s *= 2

        return x, y

    def xy_to_hilbert_index(x: int, y: int, n: int) -> int:
        """Convert (x, y) coordinates to Hilbert index."""
        index = 0
        s = 2 ** (n - 1)

        while s > 0:
            rx = int(x & s) > 0
            ry = int(y & s) > 0
            index += s * s * ((3 * rx) ^ ry)

            if ry == 0:
                if rx == 1:
                    x = 2**n - 1 - x
                    y = 2**n - 1 - y
                x, y = y, x

            s //= 2

        return index

    # Generate all indices in Hilbert order
    size = 2**n
    max_index = size * size

    # Validate we won't overflow
    if max_index > 2**31:  # Prevent integer overflow
        raise ValueError(f"Hilbert curve too large: {max_index} points")

    indices = []

    for i in range(max_index):
        x, y = hilbert_index_to_xy(i, n)
        # Validate coordinates
        if x >= size or y >= size or x < 0 or y < 0:
            raise RuntimeError(f"Invalid coordinates ({x}, {y}) for size {size}")
        linear_index = y * size + x
        indices.append(linear_index)

    return indices


def generate_hilbert_indices_rectangular(width: int, height: int) -> List[int]:
    """
    Generate Hilbert-like indices for rectangular grids.

    For non-square grids, we use a pseudo-Hilbert ordering that
    maintains good locality properties.

    Args:
        width: Width of the grid
        height: Height of the grid

    Returns:
        List of indices in Hilbert-like order
    """
    # Validate inputs
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got ({width}, {height})")

    total_size = width * height
    if total_size > 2**31:
        raise ValueError(f"Grid too large: {total_size} points")

    # For simplicity, we'll use a Z-order curve for rectangular grids
    # This still provides good cache locality
    indices = []

    # Find the maximum dimension
    max_dim = max(width, height)
    levels = int(np.ceil(np.log2(max_dim)))

    # Validate levels
    if levels > 16:
        raise ValueError(
            f"Grid dimensions too large, requiring {levels} levels (max 16)"
        )

    # Generate Z-order indices
    for i in range(total_size):
        y = i // width
        x = i % width

        # Interleave bits for Z-order
        z_index = 0
        width_bits = int(np.ceil(np.log2(width))) if width > 1 else 0
        height_bits = int(np.ceil(np.log2(height))) if height > 1 else 0

        for bit in range(levels):
            if bit < width_bits and x < width:
                z_index |= ((x >> bit) & 1) << (2 * bit)
            if bit < height_bits and y < height:
                z_index |= ((y >> bit) & 1) << (2 * bit + 1)

        indices.append((z_index, i))

    # Sort by Z-order index and extract original indices
    indices.sort(key=lambda x: x[0])
    return [idx for _, idx in indices]
