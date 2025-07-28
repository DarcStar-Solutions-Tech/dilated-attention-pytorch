#!/usr/bin/env python3
"""
Quick verification script for GPU utilities integration.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gpu_utils_import():
    """Test that GPU utilities can be imported."""
    print("Testing GPU utils import...")
    try:
        from dilated_attention_pytorch.utils import (
            get_gpu_info,
            select_gpu_attention_backend,
            benchmark_attention_backends,
            GPUInfo,
            GPUDetector,
        )

        print("✓ GPU utilities imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import GPU utilities: {e}")
        return False


def test_gpu_detection():
    """Test GPU detection functionality."""
    print("\nTesting GPU detection...")
    try:
        from dilated_attention_pytorch.utils import get_gpu_info

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        info = get_gpu_info(device)

        print("✓ GPU detection successful")
        print(f"  Device: {device}")
        print(f"  Name: {info.name}")
        print(f"  Architecture: {info.architecture}")
        print(f"  Recommended backend: {info.recommended_backend}")
        return True
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return False


def test_hilbert_gpu_import():
    """Test that GPU-optimized Hilbert attention can be imported."""
    print("\nTesting Hilbert GPU attention import...")
    try:
        from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized

        print("✓ RingDilatedAttentionHilbertGPUOptimized imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import RingDilatedAttentionHilbertGPUOptimized: {e}")
        return False


def test_hilbert_gpu_creation():
    """Test creating GPU-optimized Hilbert attention."""
    print("\nTesting Hilbert GPU attention creation...")
    try:
        from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized

        model = RingDilatedAttentionHilbertGPUOptimized(
            embed_dim=256,
            num_heads=8,
            segment_lengths=[128, 256, 128],
            dilation_rates=[1, 2, 4],
        )

        print("✓ Model created successfully")
        print(f"  Backend: {model.attention_backend}")
        print(f"  Dtype: {model.dtype}")
        print(f"  GPU: {model.gpu_info.name}")

        # Test forward pass
        x = torch.randn(2, 512, 256, dtype=model.dtype, device=model.device)
        output = model(x)

        print("✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ Model creation/forward failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_backend_selection():
    """Test backend selection logic."""
    print("\nTesting backend selection...")
    try:
        from dilated_attention_pytorch.utils import select_gpu_attention_backend

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test different scenarios
        scenarios = [
            {"seq_len": 512},
            {"seq_len": 512, "use_dilation": True},
            {"seq_len": 64},
            {"seq_len": 1024, "has_custom_mask": True},
        ]

        print("✓ Backend selection working")
        for scenario in scenarios:
            backend = select_gpu_attention_backend(device=device, **scenario)
            print(f"  {scenario} -> {backend}")

        return True
    except Exception as e:
        print(f"✗ Backend selection failed: {e}")
        return False


def main():
    print("GPU Integration Verification")
    print("=" * 50)

    tests = [
        test_gpu_utils_import,
        test_gpu_detection,
        test_hilbert_gpu_import,
        test_hilbert_gpu_creation,
        test_backend_selection,
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")

    if passed == len(tests):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
