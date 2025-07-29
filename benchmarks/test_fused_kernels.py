#!/usr/bin/env python3
"""Test fused kernels for improved performance at sequence length 4096."""

import torch
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def benchmark_sequence_length(seq_len: int, num_warmup: int = 3, num_runs: int = 10):
    """Benchmark a specific sequence length."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    # Check if fused kernels are available
    print(f"Fused kernels available: {module._fused_kernels_available}")
    print(f"Triton available: {module._triton_available}")

    # Create input
    x = torch.randn(1, seq_len, 768, device=device)

    # Test different configurations
    configs = [
        ("PyTorch Baseline", False, False),
        ("PyTorch + Hilbert", True, False),
        ("Triton + Hilbert", True, True),
    ]

    # If we're testing 4096, also test without forcing any backend
    if seq_len == 4096:
        configs.append(("Auto (should use fused)", True, None))

    results = {}

    for config_name, use_hilbert, force_triton in configs:
        # Skip Triton tests if not available
        if force_triton is True and not module._triton_available:
            continue

        # Temporarily modify Triton availability if needed
        original_triton = module._triton_available
        if force_triton is False:
            module._triton_available = False

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _ = module(x, use_hilbert=use_hilbert)

            # Time
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = module(x, use_hilbert=use_hilbert)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / num_runs * 1000

        # Restore
        module._triton_available = original_triton

        results[config_name] = elapsed

    return results


def main():
    print("Testing Fused Kernels for Sequence Length 4096")
    print("=" * 60)

    # Focus on problematic sequence length
    for seq_len in [2048, 4096, 8192]:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 40)

        results = benchmark_sequence_length(seq_len)

        # Find baseline
        baseline = results.get("PyTorch Baseline", None)

        for config_name, time_ms in results.items():
            if baseline is not None:
                speedup = baseline / time_ms
                print(f"{config_name:<30} {time_ms:>8.2f}ms ({speedup:.2f}x)")
            else:
                print(f"{config_name:<30} {time_ms:>8.2f}ms")
                baseline = time_ms

    # Detailed test for 4096
    print("\n\nDetailed Analysis for 4096 tokens:")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = UnifiedHilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=1,
        dropout=0.0,
        hilbert_threshold=1024,
    ).to(device)

    _ = torch.randn(1, 4096, 768, device=device)

    # Check which path is taken
    with torch.no_grad():
        # Inject some debug info
        M_padded = 4096  # Already aligned
        use_fused = (
            module._triton_available
            and device == "cuda"
            and 2048 <= M_padded <= 8192
            and hasattr(module, "_fused_kernels_available")
            and module._fused_kernels_available
        )

        print(f"Will use fused kernel: {use_fused}")
        print("Reason:")
        print(f"  - Triton available: {module._triton_available}")
        print(f"  - CUDA device: {device == 'cuda'}")
        print(f"  - Sequence in range [2048, 8192]: {2048 <= M_padded <= 8192}")
        print(
            f"  - Fused kernels available: {getattr(module, '_fused_kernels_available', False)}"
        )


if __name__ == "__main__":
    main()
