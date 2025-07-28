#!/usr/bin/env python3
"""
Basic Dilated Attention Example

This example demonstrates how to use dilated attention for handling long sequences
on a single GPU. While not as memory-efficient as Ring Attention, dilated attention
still provides significant benefits for long sequence modeling.
"""

import torch
from dilated_attention_pytorch import (
    create_multihead_dilated_attention,
    create_block_sparse_attention,
)


def basic_usage():
    """Show basic usage of dilated attention."""
    print("=== Basic Dilated Attention Usage ===\n")

    # Configuration
    batch_size = 2
    seq_len = 8192  # 8K tokens
    embed_dim = 768
    num_heads = 12

    # Create dilated attention module
    attention = create_multihead_dilated_attention(
        "improved",  # Use improved implementation
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=[1024, 2048, 4096],
        dilation_rates=[1, 2, 4],
        dropout=0.1,
    )

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    # Create input
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Forward pass
    output = attention(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Device: {device}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")


def compare_implementations():
    """Compare different attention implementations."""
    print("\n=== Comparing Attention Implementations ===\n")

    # Configuration
    batch_size = 1
    seq_len = 4096
    embed_dim = 512
    num_heads = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    implementations = {
        "standard": create_multihead_dilated_attention(
            "standard",
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[2048],
            dilation_rates=[1],
        ),
        "improved": create_multihead_dilated_attention(
            "improved",
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[1024, 2048],
            dilation_rates=[1, 2],
        ),
        "block_sparse": create_block_sparse_attention(
            sparsity_ratio=0.9,  # 90% sparse
            embed_dim=embed_dim,
            num_heads=num_heads,
        ),
    }

    for name, module in implementations.items():
        module = module.to(device)

        # Measure memory and time
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        with torch.no_grad():
            output = module(x, x, x)
        end_time.record()

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
        elapsed_time = start_time.elapsed_time(end_time)

        print(f"{name.capitalize()} Attention:")
        print(f"  Time: {elapsed_time:.2f} ms")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Output shape: {output.shape}")


def causal_attention_example():
    """Show how to use causal (autoregressive) attention."""
    print("\n=== Causal Attention Example ===\n")

    # Create a model for autoregressive generation
    class SimpleGPT(torch.nn.Module):
        def __init__(self, vocab_size=50000, embed_dim=512, num_heads=8):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
            self.attention = create_multihead_dilated_attention(
                "improved",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[512, 1024, 2048],
                dilation_rates=[1, 2, 4],
                dropout=0.1,
            )
            self.ln = torch.nn.LayerNorm(embed_dim)
            self.output = torch.nn.Linear(embed_dim, vocab_size)

        def forward(self, input_ids):
            # Embed tokens
            x = self.embedding(input_ids)

            # Self-attention with causal mask
            attended = self.attention(x, x, x, is_causal=True)

            # Add & norm
            x = self.ln(x + attended)

            # Output logits
            logits = self.output(x)
            return logits

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGPT().to(device)

    # Generate some text
    input_ids = torch.randint(0, 50000, (1, 100), device=device)

    with torch.no_grad():
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Next token prediction: {next_token.item()}")


def adaptive_sparse_attention_example():
    """Show adaptive sparse attention that learns sparsity patterns."""
    print("\n=== Adaptive Sparse Attention Example ===\n")

    # Create adaptive sparse attention
    attention = create_multihead_dilated_attention(
        "block_sparse_ring",
        embed_dim=768,
        num_heads=12,
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        enable_adaptive=True,
        min_sparsity=0.8,
        max_sparsity=0.95,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attention = attention.to(device)

    # Training mode - learns sparsity patterns
    attention.train()
    x = torch.randn(2, 2048, 768, device=device)
    output = attention(x, x, x)

    print("Adaptive Sparse Attention:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Learns optimal sparsity patterns during training")
    print("  Sparsity range: 80% - 95%")


def main():
    """Run all examples."""
    print("Dilated Attention PyTorch Examples")
    print("=" * 50)

    # Check for GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slower)")

    # Run examples
    basic_usage()
    compare_implementations()
    causal_attention_example()
    adaptive_sparse_attention_example()

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
