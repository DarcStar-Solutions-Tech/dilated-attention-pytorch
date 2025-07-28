#!/usr/bin/env python3
"""
Example demonstrating dynamic segment selection in Dilated Attention.

This example shows how segments are automatically selected based on:
- Available GPU memory
- Sequence length
- Batch size
"""

import torch
from dilated_attention_pytorch import (
    DynamicDilatedAttention,
    DynamicMultiheadDilatedAttention,
    SegmentSelectionConfig,
)


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        print(f"GPU Memory: {used / 1e9:.2f}GB used / {total / 1e9:.2f}GB total")
    else:
        print("No GPU available")


def demo_basic_dynamic_attention():
    """Demonstrate basic dynamic attention."""
    print("=== Basic Dynamic Attention Demo ===\n")

    # Create dynamic attention module
    attention = DynamicDilatedAttention(
        min_segment_size=512,
        max_segment_size=8192,
        improved=True,  # Use improved implementation
    )

    # Test with different sequence lengths
    sequence_lengths = [1024, 4096, 16384]
    batch_size = 2
    num_heads = 8
    head_dim = 64

    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        print_gpu_memory()

        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim)

        if torch.cuda.is_available():
            q, k, v = q.cuda(), k.cuda(), v.cuda()
            attention = attention.cuda()

        # Forward pass - segments selected automatically
        output = attention(q, k, v)

        # Get selected configuration
        segments, dilation_rates = attention.get_current_configuration()

        print(f"Selected segments: {segments}")
        print(f"Dilation rates: {dilation_rates}")
        print(f"Output shape: {output.shape}")

        # Clean up
        del q, k, v, output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def demo_memory_aware_selection():
    """Demonstrate memory-aware segment selection."""
    print("\n\n=== Memory-Aware Selection Demo ===\n")

    # Configure for aggressive memory management
    config = SegmentSelectionConfig(
        memory_safety_factor=0.6,  # Use only 60% of available memory
        min_free_memory_gb=1.0,  # Keep 1GB free
        prefer_power_of_2=True,
    )

    attention = DynamicDilatedAttention(selector_config=config, max_segment_size=32768)

    # Simulate high memory pressure
    print("Testing under different memory conditions...")

    # Test 1: Normal conditions
    seq_len = 8192
    batch_size = 4

    q = torch.randn(batch_size, seq_len, 8, 64)
    k = torch.randn(batch_size, seq_len, 8, 64)
    v = torch.randn(batch_size, seq_len, 8, 64)

    if torch.cuda.is_available():
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        attention = attention.cuda()

    output = attention(q, k, v)
    segments_normal, _ = attention.get_current_configuration()
    print(f"\nNormal conditions - segments: {segments_normal}")

    # Test 2: Large batch size (simulating memory pressure)
    large_batch = 16
    q_large = torch.randn(large_batch, seq_len, 8, 64)
    k_large = torch.randn(large_batch, seq_len, 8, 64)
    v_large = torch.randn(large_batch, seq_len, 8, 64)

    if torch.cuda.is_available():
        q_large, k_large, v_large = q_large.cuda(), k_large.cuda(), v_large.cuda()

    # Force segment update due to different batch size
    output_large = attention(q_large, k_large, v_large, force_segment_update=True)
    segments_pressure, _ = attention.get_current_configuration()
    print(f"High memory pressure - segments: {segments_pressure}")

    # Segments should be smaller under memory pressure
    if max(segments_pressure) <= max(segments_normal):
        print("✓ Successfully adapted to memory pressure")

    # Clean up
    del q, k, v, q_large, k_large, v_large, output, output_large
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def demo_multihead_compatibility():
    """Demonstrate drop-in replacement for MultiheadAttention."""
    print("\n\n=== Multihead Attention Compatibility Demo ===\n")

    # Create dynamic multihead attention
    embed_dim = 768
    num_heads = 12

    attention = DynamicMultiheadDilatedAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.1
    )

    # Test with typical transformer inputs
    seq_len = 512
    batch_size = 8

    # Input: [batch, seq_len, embed_dim]
    x = torch.randn(batch_size, seq_len, embed_dim)

    if torch.cuda.is_available():
        x = x.cuda()
        attention = attention.cuda()

    # Use like standard MultiheadAttention
    output, attn_weights = attention(x, x, x, need_weights=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if attn_weights is not None:
        print(f"Attention weights shape: {attn_weights.shape}")

    # Get configuration
    segments, rates = attention.get_current_configuration()
    print(f"Selected segments: {segments}")
    print(f"Dilation rates: {rates}")


def demo_content_aware_selection():
    """Demonstrate content-aware segment selection."""
    print("\n\n=== Content-Aware Selection Demo ===\n")

    # Simulate a document with natural boundaries
    # e.g., paragraphs at positions 256, 512, 1024, 1536
    content_boundaries = [256, 512, 1024, 1536]
    total_length = 2048

    attention = DynamicDilatedAttention()

    # Create input
    q = torch.randn(1, total_length, 8, 64)
    k = torch.randn(1, total_length, 8, 64)
    v = torch.randn(1, total_length, 8, 64)

    if torch.cuda.is_available():
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        attention = attention.cuda()

    # Note: Current implementation doesn't use content_boundaries directly,
    # but this demonstrates how it could be extended
    _ = attention(q, k, v)

    segments, rates = attention.get_current_configuration()
    print(f"Document length: {total_length}")
    print(f"Content boundaries: {content_boundaries}")
    print(f"Selected segments: {segments}")
    print(f"Dilation rates: {rates}")


def main():
    """Run all demonstrations."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Print system info
    print("System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print_gpu_memory()
    print("\n" + "=" * 60 + "\n")

    # Run demos
    demo_basic_dynamic_attention()
    demo_memory_aware_selection()
    demo_multihead_compatibility()
    demo_content_aware_selection()

    print("\n\nDemo completed!")


if __name__ == "__main__":
    main()
