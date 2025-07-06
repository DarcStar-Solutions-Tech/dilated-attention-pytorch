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
    indices = []

    for i in range(size * size):
        x, y = hilbert_index_to_xy(i, n)
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
    # For simplicity, we'll use a Z-order curve for rectangular grids
    # This still provides good cache locality
    indices = []

    # Find the maximum dimension
    max_dim = max(width, height)
    levels = int(np.ceil(np.log2(max_dim)))

    # Generate Z-order indices
    for i in range(width * height):
        y = i // width
        x = i % width

        # Interleave bits for Z-order
        z_index = 0
        for bit in range(levels):
            if bit < int(np.log2(width)):
                z_index |= ((x >> bit) & 1) << (2 * bit)
            if bit < int(np.log2(height)):
                z_index |= ((y >> bit) & 1) << (2 * bit + 1)

        indices.append((z_index, i))

    # Sort by Z-order index and extract original indices
    indices.sort(key=lambda x: x[0])
    return [idx for _, idx in indices]
