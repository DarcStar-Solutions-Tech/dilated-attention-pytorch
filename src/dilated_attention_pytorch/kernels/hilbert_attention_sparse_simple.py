#!/usr/bin/env python3
"""
Simplified sparse Hilbert attention that demonstrates the key optimization:
applying Hilbert reordering only to the sparse positions actually accessed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict

from .cache_manager import BoundedCache


class HilbertAttentionSparseSimple(nn.Module):
    """
    Simplified implementation showing the core optimization:
    Hilbert reordering applied only to sparse positions.

    Key insight: Instead of creating a mapping for all 1024 positions and then
    filtering, we create a mapping only for the 256 positions we actually access
    (with dilation_rate=4).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        segment_size: int = 128,
        dilation_rate: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.segment_size = segment_size
        self.dilation_rate = dilation_rate
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Cache for sparse Hilbert patterns
        # Initialize bounded cache for sparse patterns
        self._sparse_pattern_cache = BoundedCache(
            max_size=32,
            max_memory_mb=50.0,  # Sparse patterns are smaller
            name=f"{self.__class__.__name__}_sparse_pattern",
        )

    def create_sparse_hilbert_pattern(self, num_positions: int) -> torch.Tensor:
        """
        Create a simple Hilbert-like pattern for sparse positions.
        This is much smaller than creating a pattern for the full sequence.
        """
        if num_positions <= 1:
            return torch.arange(num_positions)

        # Simple bit-reversal pattern that provides good cache locality
        _ = torch.arange(num_positions)

        # Bit reversal for powers of 2
        n = 1
        while n < num_positions:
            n *= 2

        result = torch.zeros(num_positions, dtype=torch.long)
        for i in range(num_positions):
            # Simple bit reversal
            rev = 0
            val = i
            for _ in range(n.bit_length() - 1):
                rev = (rev << 1) | (val & 1)
                val >>= 1
            if rev < num_positions:
                result[i] = rev
            else:
                result[i] = i

        return result

    def forward(self, x: torch.Tensor, use_hilbert: bool = True) -> torch.Tensor:
        """
        Forward pass demonstrating sparse Hilbert optimization.
        """
        B, M, _ = x.shape
        H = self.num_heads
        device = x.device

        # Pad if necessary
        if M % self.segment_size != 0:
            pad_len = self.segment_size - (M % self.segment_size)
            x = F.pad(x, (0, 0, 0, pad_len))
            M_padded = M + pad_len
        else:
            M_padded = M

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, M_padded, 3, H, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Allocate output
        out = torch.zeros_like(q)

        # Process each segment
        for seg_start in range(0, M_padded, self.segment_size):
            seg_end = min(seg_start + self.segment_size, M_padded)
            seg_len = seg_end - seg_start

            # Get queries for this segment
            q_seg = q[:, :, seg_start:seg_end, :]

            # Calculate number of sparse positions
            num_sparse = seg_len // self.dilation_rate
            if num_sparse == 0:
                continue

            # Key optimization: Create pattern only for sparse positions
            if use_hilbert and num_sparse > 1:
                # Try to get from cache
                sparse_pattern = self._sparse_pattern_cache.get(num_sparse)

                if sparse_pattern is None or sparse_pattern.device != device:
                    # Create new pattern
                    sparse_pattern = self.create_sparse_hilbert_pattern(num_sparse).to(
                        device
                    )
                    # Store in cache (will handle LRU eviction if needed)
                    self._sparse_pattern_cache.put(num_sparse, sparse_pattern)
            else:
                sparse_pattern = torch.arange(num_sparse, device=device)

            # Convert sparse indices to actual positions
            # This is the key: we're working with num_sparse positions, not seg_len
            actual_positions = seg_start + sparse_pattern * self.dilation_rate

            # Ensure positions are within segment bounds
            mask = actual_positions < seg_end
            actual_positions = actual_positions[mask]

            if len(actual_positions) == 0:
                continue

            # Get K and V at sparse positions (with Hilbert reordering if enabled)
            k_seg = k[:, :, actual_positions, :]
            v_seg = v[:, :, actual_positions, :]

            # Standard attention computation
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            out_seg = torch.matmul(attn_weights, v_seg)
            out[:, :, seg_start:seg_end, :] = out_seg

        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, M_padded, self.hidden_dim)
        out = self.out_proj(out)

        # Remove padding
        if M != M_padded:
            out = out[:, :M, :]

        return out

    def clear_cache(self) -> None:
        """Clear the sparse pattern cache to free memory."""
        self._sparse_pattern_cache.clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return self._sparse_pattern_cache.get_stats()

    def compare_memory_usage(self, seq_len: int) -> Dict[str, any]:
        """Compare memory usage with original approach."""
        # Original approach
        original_map_size = seq_len  # Full sequence Hilbert map

        # Optimized approach
        sparse_per_segment = self.segment_size // self.dilation_rate
        optimized_map_size = sparse_per_segment  # Only one pattern cached

        # Actual positions accessed
        num_segments = (seq_len + self.segment_size - 1) // self.segment_size
        total_accessed = num_segments * sparse_per_segment

        return {
            "original_approach": {
                "hilbert_map_size": original_map_size,
                "description": "Full sequence Hilbert mapping",
            },
            "optimized_approach": {
                "hilbert_map_size": optimized_map_size,
                "description": "Sparse positions only",
            },
            "improvement": {
                "memory_reduction": f"{original_map_size / optimized_map_size:.1f}x",
                "positions_accessed": total_accessed,
                "sparsity": f"{(1 - 1 / self.dilation_rate) * 100:.0f}%",
            },
        }


def demonstrate_optimization():
    """Demonstrate the sparse Hilbert optimization."""
    print("Sparse Hilbert Optimization Demonstration")
    print("=" * 70)

    # Configuration
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 1024
    segment_size = 256
    dilation_rate = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(
        f"Configuration: seq_len={seq_len}, segment_size={segment_size}, dilation_rate={dilation_rate}"
    )

    # Create modules for comparison
    from dilated_attention_pytorch.kernels import HilbertAttentionCore

    original = HilbertAttentionCore(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    optimized = HilbertAttentionSparseSimple(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        dilation_rate=dilation_rate,
    ).to(device)

    # Memory comparison
    print("\n" + "=" * 70)
    print("MEMORY USAGE COMPARISON")
    print("=" * 70)

    memory_comp = optimized.compare_memory_usage(seq_len)

    print("\nOriginal Implementation:")
    print(
        f"  - Creates Hilbert map for entire sequence: {memory_comp['original_approach']['hilbert_map_size']} entries"
    )
    print("  - Then filters to sparse positions during kernel execution")

    print("\nOptimized Implementation:")
    print(
        f"  - Creates Hilbert map only for sparse positions: {memory_comp['optimized_approach']['hilbert_map_size']} entries"
    )
    print("  - Direct access without filtering")

    print("\nImprovement:")
    print(f"  - Memory reduction: {memory_comp['improvement']['memory_reduction']}")
    print(
        f"  - Total positions accessed: {memory_comp['improvement']['positions_accessed']}"
    )
    print(f"  - Sparsity: {memory_comp['improvement']['sparsity']}")

    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Warmup
    for _ in range(3):
        _ = original(x, use_hilbert=True)
        _ = optimized(x, use_hilbert=True)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    num_iterations = 10

    # Original with Hilbert
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_original = original(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    time_original = (time.perf_counter() - start) / num_iterations * 1000

    # Optimized with sparse Hilbert
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_optimized = optimized(x, use_hilbert=True)
    if device == "cuda":
        torch.cuda.synchronize()
    time_optimized = (time.perf_counter() - start) / num_iterations * 1000

    # Without any Hilbert
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = original(x, use_hilbert=False)
    if device == "cuda":
        torch.cuda.synchronize()
    time_no_hilbert = (time.perf_counter() - start) / num_iterations * 1000

    print(f"\nOriginal (full Hilbert map): {time_original:.2f}ms")
    print(f"Optimized (sparse Hilbert):   {time_optimized:.2f}ms")
    print(f"No Hilbert (baseline):        {time_no_hilbert:.2f}ms")

    print(f"\nSpeedup of optimized vs original: {time_original / time_optimized:.2f}x")
    print(
        f"Overhead vs no Hilbert: {(time_optimized - time_no_hilbert) / time_no_hilbert * 100:.1f}%"
    )

    # Verify correctness
    with torch.no_grad():
        diff = (out_optimized - out_original).abs().max().item()
        print(f"\nMax difference between implementations: {diff:.6f}")

    # Visualize the difference
    print("\n" + "=" * 70)
    print("CONCEPTUAL DIFFERENCE")
    print("=" * 70)

    print("\nOriginal approach:")
    print("1. Create mapping: [0,1,2,3...1023] -> [hilbert reordered positions]")
    print("2. In kernel: Check if position is dilated (0,4,8,12...)")
    print("3. If yes, use Hilbert mapping to reorder")
    print("4. Result: 1024-entry map, but only use 256 entries")

    print("\nOptimized approach:")
    print("1. Identify sparse positions: [0,4,8,12...252] (64 positions per segment)")
    print(
        "2. Create mapping just for these: [0,1,2...63] -> [reordered sparse indices]"
    )
    print("3. Direct access without filtering")
    print("4. Result: 64-entry map, use all entries")

    print("\nKey insight: Why map positions we'll never access?")


if __name__ == "__main__":
    demonstrate_optimization()
