#!/usr/bin/env python3
"""Debug Triton performance regression."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def debug_performance():
    """Debug why Triton performance regressed."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Debugging Triton Performance")
    print("=" * 60)

    # Test with 8192 sequence length where we saw regression
    seq_len = 8192

    # Create module
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,  # Normal threshold
    ).to(device)

    x = torch.randn(1, seq_len, 768, device=device)

    print(f"Sequence length: {seq_len}")
    print(f"Hilbert threshold: {module.hilbert_threshold}")
    print(f"Will use Hilbert: {seq_len > module.hilbert_threshold}")

    # Check if Triton is being used
    print(f"\nTriton available: {module._triton_available}")

    # Test different configurations
    configs = [
        ("PyTorch baseline", False, False),
        ("PyTorch + Hilbert", False, True),
        ("Triton baseline", True, False),
        ("Triton + Hilbert", True, True),
    ]

    print(f"\n{'Config':<20} {'Time (ms)':<15}")
    print("-" * 35)

    for name, use_triton, use_hilbert in configs:
        # Set backend
        original_triton = module._triton_available
        module._triton_available = use_triton

        with torch.no_grad():
            # Warmup
            _ = module(x, use_hilbert=use_hilbert)

            # Time
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = module(x, use_hilbert=use_hilbert)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000

        print(f"{name:<20} {elapsed:<15.2f}")

        # Restore
        module._triton_available = original_triton

    # Check block sizes being used
    from dilated_attention_pytorch.kernels.hilbert_attention_core import (
        HilbertAttentionCore,
    )

    core = HilbertAttentionCore(
        hidden_dim=768, num_heads=12, segment_size=128, dilation_rate=1
    )

    M, N, D = core.get_optimal_block_sizes(seq_len, torch.device(device))
    print(f"\nOptimal block sizes for seq={seq_len}: M={M}, N={N}, D={D}")

    # Check compute capability
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU compute capability: {capability}")


if __name__ == "__main__":
    debug_performance()
