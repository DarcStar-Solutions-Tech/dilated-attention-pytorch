#!/usr/bin/env python3
"""Test if Hilbert ordering is applied correctly for sparse patterns."""

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import HilbertAttention


def test_hilbert_sparse_ordering():
    """Verify Hilbert ordering behavior with sparse patterns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create module with dilation
    module = HilbertAttention(
        hidden_dim=768,
        num_heads=12,
        segment_size=128,
        dilation_rate=4,  # Sparse pattern
        dropout=0.0,
    ).to(device)

    # Test input
    batch_size, seq_len = 2, 512
    x = torch.randn(batch_size, seq_len, 768, device=device)

    print("Testing Hilbert ordering with sparse patterns")
    print(
        f"Configuration: dilation_rate={module.dilation_rate}, segment_size={module.segment_size}"
    )
    print(f"Triton available: {module._triton_available}")
    print()

    # Test with PyTorch backend (force by disabling Triton temporarily)
    original_triton = module._triton_available
    module._triton_available = False

    with torch.no_grad():
        out_pytorch = module(x, use_hilbert=True)
        out_pytorch_no_hilbert = module(x, use_hilbert=False)

    pytorch_diff = torch.norm(out_pytorch - out_pytorch_no_hilbert).item()
    print(f"PyTorch backend difference (hilbert vs no hilbert): {pytorch_diff:.6f}")

    # Test with Triton backend if available
    if original_triton and device == "cuda":
        module._triton_available = True

        with torch.no_grad():
            out_triton = module(x, use_hilbert=True)
            out_triton_no_hilbert = module(x, use_hilbert=False)

        triton_diff = torch.norm(out_triton - out_triton_no_hilbert).item()
        print(f"Triton backend difference (hilbert vs no hilbert): {triton_diff:.6f}")

        # Compare PyTorch vs Triton
        backend_diff = torch.norm(out_pytorch - out_triton).item()
        print(f"\nPyTorch vs Triton output difference: {backend_diff:.6f}")

    # Test different dilation rates
    print("\nTesting impact of Hilbert ordering across dilation rates:")
    for dilation in [1, 2, 4, 8]:
        module.dilation_rate = dilation
        module._triton_available = original_triton

        with torch.no_grad():
            out_hilbert = module(x, use_hilbert=True)
            out_standard = module(x, use_hilbert=False)

        diff = torch.norm(out_hilbert - out_standard).item()
        backend = (
            "Triton" if module._triton_available and device == "cuda" else "PyTorch"
        )
        print(f"  Dilation={dilation}: difference={diff:.6f} (backend={backend})")


if __name__ == "__main__":
    test_hilbert_sparse_ordering()
