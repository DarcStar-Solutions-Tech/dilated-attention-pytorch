#!/usr/bin/env python3
"""
Example of using the factory pattern to create ring attention implementations.

This demonstrates how to use create_dilated_attention() to instantiate
various ring attention types with proper O(n/k) memory scaling.
"""

from dilated_attention_pytorch import create_dilated_attention


def main():
    """Demonstrate factory usage for ring attention."""

    print("Ring Attention Factory Examples")
    print("=" * 60)

    # Example 1: Basic ring attention (StandardRingAttention)
    print("\n1. Basic Ring Attention (StandardRingAttention)")
    ring_basic = create_dilated_attention(
        "ring",
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        ring_size=4,  # Simulating 4 GPUs
    )
    print(f"   Created: {type(ring_basic).__name__}")
    print(f"   Memory scaling: O(n/{4}) per GPU")

    # Example 2: Distributed ring with enterprise features
    print("\n2. Distributed Ring Attention (enterprise features)")
    ring_dist = create_dilated_attention(
        "ring_distributed",
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        ring_size=8,
    )
    print(f"   Created: {type(ring_dist).__name__}")
    print("   Features: DeepSpeed, fault tolerance, monitoring")

    # Example 3: Hilbert-optimized ring attention
    print("\n3. Hilbert Ring Attention (cache-optimized)")
    ring_hilbert = create_dilated_attention(
        "ring_hilbert",
        segment_lengths=[4096],
        dilation_rates=[1],
        ring_size=2,
    )
    print(f"   Created: {type(ring_hilbert).__name__}")
    print("   Features: Hilbert curve reordering for cache efficiency")

    # Example 4: Legacy implementations (need embed_dim/num_heads)
    print("\n4. Legacy Ring Implementations")

    # Correct implementation
    ring_correct = create_dilated_attention(
        "ring_correct",
        segment_lengths=[2048],
        dilation_rates=[1],
        embed_dim=768,
        num_heads=12,
    )
    print(f"   RingCorrect: {type(ring_correct).__name__}")
    print("   Note: Splits sequences BEFORE projection")

    # SDPA implementation
    ring_sdpa = create_dilated_attention(
        "ring_sdpa",
        segment_lengths=[2048],
        dilation_rates=[1],
        embed_dim=768,
        num_heads=12,
    )
    print(f"   RingSDPA: {type(ring_sdpa).__name__}")
    print("   Note: Uses PyTorch's scaled_dot_product_attention")

    # GPU-optimized Hilbert
    ring_gpu = create_dilated_attention(
        "ring_hilbert_gpu",
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        embed_dim=768,
        num_heads=12,
        ring_size=4,
    )
    print(f"   RingHilbertGPU: {type(ring_gpu).__name__}")

    # Example 5: Auto-selection (future feature)
    print("\n5. Auto-Selection Example")
    print("   Note: 'auto' currently selects standard attention, not ring")
    print("   Use explicit ring types for O(n/k) memory scaling")

    # Example 6: Memory comparison
    print("\n6. Memory Scaling Comparison")
    print("   Standard attention: O(n) memory per GPU")
    print("   Ring attention: O(n/k) memory per GPU")
    print("   Example with 100K tokens:")
    print("     - 1 GPU: 1 GB (both)")
    print("     - 4 GPUs standard: 1 GB per GPU (4 GB total)")
    print("     - 4 GPUs ring: 250 MB per GPU (1 GB total)")

    # Example 7: Usage in training
    print("\n7. Training Usage Pattern")
    print("""
    # Multi-GPU training with ring attention
    model = MyTransformer(
        attention_layer=create_dilated_attention(
            "ring_distributed",
            segment_lengths=[2048, 4096],
            dilation_rates=[1, 2],
            ring_size=torch.distributed.get_world_size(),
        )
    )
    
    # Run with: torchrun --nproc_per_node=4 train.py
    """)

    print("\n" + "=" * 60)
    print("Remember: Ring attention provides O(n/k) memory scaling!")
    print("Use torchrun for multi-GPU execution.")


if __name__ == "__main__":
    main()
