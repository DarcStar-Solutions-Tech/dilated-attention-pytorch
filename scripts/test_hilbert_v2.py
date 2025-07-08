#!/usr/bin/env python3
"""Test script for Hilbert Attention V2 implementation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dilated_attention_pytorch.kernels.hilbert_attention_triton_v2 import (
    HilbertAttentionV2,
    create_z_order_mapping,
    create_gray_code_mapping,
    create_hilbert_mapping_simple,
)


def visualize_pattern(mapping: torch.Tensor, name: str, size: int = 16):
    """Visualize a pattern mapping as a 2D grid."""
    print(f"\n{name} Pattern (first {size}x{size} elements):")
    print("-" * (size * 4 + 1))

    grid = torch.zeros(size, size, dtype=torch.long)

    for i in range(min(len(mapping), size * size)):
        row = i // size
        col = i % size
        if row < size and col < size:
            grid[row, col] = mapping[i].item()

    # Print grid
    for row in range(size):
        print("|", end="")
        for col in range(size):
            print(f"{grid[row, col]:3d}|", end="")
        print()
    print("-" * (size * 4 + 1))


def test_pattern_mappings():
    """Test and visualize different pattern mappings."""
    print("=== Testing Pattern Mappings ===")

    seq_len = 256

    # Create different mappings
    z_order = create_z_order_mapping(seq_len)
    gray_code = create_gray_code_mapping(seq_len)
    bit_reversal = create_hilbert_mapping_simple(seq_len)

    # Visualize patterns
    visualize_pattern(z_order, "Z-Order (Morton)")
    visualize_pattern(gray_code, "Gray Code")
    visualize_pattern(bit_reversal, "Bit Reversal")

    # Check properties
    print("\nPattern Properties:")
    print("-" * 40)

    for name, mapping in [
        ("Z-Order", z_order),
        ("Gray Code", gray_code),
        ("Bit Reversal", bit_reversal),
    ]:
        # Check uniqueness
        unique_vals = torch.unique(mapping)
        is_permutation = len(unique_vals) == len(mapping)

        # Check range
        min_val = mapping.min().item()
        max_val = mapping.max().item()

        print(
            f"{name:12} - Permutation: {is_permutation}, Range: [{min_val}, {max_val}]"
        )


def test_attention_computation():
    """Test attention computation with different patterns."""
    print("\n=== Testing Attention Computation ===")

    # Configuration
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    num_heads = 8
    segment_size = 32
    dilation_rate = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create input
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    # Test each pattern type
    pattern_types = ["z_order", "gray_code", "bit_reversal"]
    models = {}
    outputs = {}

    print("\nRunning attention with different patterns...")

    for pattern_type in pattern_types:
        model = HilbertAttentionV2(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            segment_size=segment_size,
            dilation_rate=dilation_rate,
            pattern_type=pattern_type,
        ).to(device)

        models[pattern_type] = model

        # Forward pass with pattern
        with torch.no_grad():
            out_pattern = model(x, use_pattern=True)
            out_standard = model(x, use_pattern=False)

        outputs[pattern_type] = {
            "pattern": out_pattern,
            "standard": out_standard,
        }

        # Compare pattern vs standard
        diff = (out_pattern - out_standard).abs().max().item()
        print(f"{pattern_type:12} - Max diff (pattern vs standard): {diff:.2e}")

    # Cross-compare different patterns
    print("\nCross-comparison of pattern outputs:")
    print("-" * 50)

    patterns = list(outputs.keys())
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            diff = (
                (outputs[patterns[i]]["pattern"] - outputs[patterns[j]]["pattern"])
                .abs()
                .max()
                .item()
            )
            print(f"{patterns[i]:12} vs {patterns[j]:12}: {diff:.2e}")


def test_gradient_flow():
    """Test gradient flow through the attention mechanism."""
    print("\n=== Testing Gradient Flow ===")

    # Configuration
    batch_size = 1
    seq_len = 64
    hidden_dim = 128
    num_heads = 4
    segment_size = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and input
    model = HilbertAttentionV2(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        segment_size=segment_size,
        pattern_type="z_order",
    ).to(device)

    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, requires_grad=True)

    # Forward and backward
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check gradients
    print(f"Input gradient norm: {x.grad.norm().item():.4f}")
    print(
        f"QKV projection gradient norm: {model.qkv_proj.weight.grad.norm().item():.4f}"
    )
    print(
        f"Output projection gradient norm: {model.out_proj.weight.grad.norm().item():.4f}"
    )

    # Verify no NaN or inf
    has_nan = torch.isnan(x.grad).any().item()
    has_inf = torch.isinf(x.grad).any().item()
    print(f"Gradient has NaN: {has_nan}")
    print(f"Gradient has Inf: {has_inf}")


def benchmark_performance():
    """Benchmark performance of different patterns."""
    print("\n=== Performance Benchmark ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    # Configurations to test
    configs = [
        (1, 256, 256, 8, 64, 1),
        (2, 512, 512, 16, 128, 2),
        (4, 1024, 768, 12, 256, 4),
    ]

    pattern_types = ["z_order", "gray_code", "bit_reversal", None]

    print("\nConfig (B×L×D, H, seg, dil) | Pattern    | Time (ms) | Memory (MB)")
    print("-" * 70)

    for batch, seq_len, hidden_dim, heads, segment_size, dilation in configs:
        x = torch.randn(batch, seq_len, hidden_dim, device="cuda")

        for pattern_type in pattern_types:
            if pattern_type is None:
                # Test standard attention (no pattern)
                model = HilbertAttentionV2(
                    hidden_dim=hidden_dim,
                    num_heads=heads,
                    segment_size=segment_size,
                    dilation_rate=dilation,
                    pattern_type="z_order",  # Doesn't matter
                ).cuda()
                use_pattern = False
                pattern_name = "standard"
            else:
                model = HilbertAttentionV2(
                    hidden_dim=hidden_dim,
                    num_heads=heads,
                    segment_size=segment_size,
                    dilation_rate=dilation,
                    pattern_type=pattern_type,
                ).cuda()
                use_pattern = True
                pattern_name = pattern_type

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x, use_pattern=use_pattern)

            # Clear cache and measure memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Time forward pass
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                for _ in range(50):
                    _ = model(x, use_pattern=use_pattern)
            end.record()

            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / 50

            # Get peak memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

            config_str = (
                f"{batch}×{seq_len}×{hidden_dim}, {heads}, {segment_size}, {dilation}"
            )
            print(
                f"{config_str} | {pattern_name:10} | {time_ms:9.2f} | {peak_memory:10.2f}"
            )


if __name__ == "__main__":
    print("Hilbert Attention V2 Test Suite")
    print("=" * 60)

    test_pattern_mappings()
    test_attention_computation()
    test_gradient_flow()
    benchmark_performance()

    print("\n✓ All tests completed!")
