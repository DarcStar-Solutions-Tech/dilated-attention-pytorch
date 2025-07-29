#!/usr/bin/env python3
"""Debug Triton kernel overhead issues."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def analyze_triton_overhead():
    """Analyze where Triton overhead comes from."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Analyzing Triton Overhead")
    print("=" * 60)

    # Test different configurations
    configs = [
        # (seq_len, segment_size, dilation_rate)
        (1024, 128, 1),
        (2048, 128, 1),
        (4096, 128, 1),
        (4096, 256, 1),  # Larger segment
        (4096, 128, 2),  # With dilation
    ]

    for seq_len, segment_size, dilation_rate in configs:
        print(
            f"\nConfig: seq_len={seq_len}, segment_size={segment_size}, dilation_rate={dilation_rate}"
        )

        # Create module
        module = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            dropout=0.0,
            hilbert_threshold=100,  # Force Hilbert for testing
        ).to(device)

        x = torch.randn(1, seq_len, 768, device=device)

        # Get optimal block sizes
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            UnifiedHilbertAttention,
        )

        core = UnifiedHilbertAttention(
            hidden_dim=768,
            num_heads=12,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
        )
        BLOCK_M, BLOCK_N, BLOCK_D = core.get_optimal_block_sizes(
            seq_len, torch.device(device)
        )
        print(f"  Block sizes: M={BLOCK_M}, N={BLOCK_N}, D={BLOCK_D}")

        # Calculate grid size
        import triton

        grid_size = triton.cdiv(seq_len, BLOCK_M) * 1 * 12  # B=1, H=12
        print(f"  Grid size: {grid_size} blocks")

        # Time kernel launch overhead
        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=True)

            # Measure just kernel launch (small operation)
            dummy = torch.zeros(1, device=device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                dummy += 1
            torch.cuda.synchronize()
            launch_overhead = (time.perf_counter() - start) * 10  # ms per launch

            print(f"  Kernel launch overhead: {launch_overhead:.4f}ms")

            # Time actual Triton execution
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) * 1000

            # Force PyTorch backend
            module._triton_available = False
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=True)
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) * 1000

            print(f"  Triton time: {triton_time:.2f}ms")
            print(f"  PyTorch time: {pytorch_time:.2f}ms")
            print(f"  Overhead factor: {triton_time / pytorch_time:.2f}x")

        # Analyze memory access patterns
        if dilation_rate == 1:
            # For dense attention, every position is accessed
            memory_accesses = seq_len * seq_len
        else:
            # For dilated attention
            num_segments = seq_len // segment_size
            accesses_per_segment = segment_size * (segment_size // dilation_rate)
            memory_accesses = num_segments * accesses_per_segment

        print(f"  Memory accesses: {memory_accesses:,}")
        print(f"  Accesses per block: {memory_accesses // grid_size:,}")


if __name__ == "__main__":
    analyze_triton_overhead()
