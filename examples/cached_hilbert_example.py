"""
Example usage of the cached Hilbert implementation for block-sparse attention.
"""

import torch
import torch.nn as nn
from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert_cached import (
    create_cached_block_sparse_hilbert,
)


def simple_example():
    """Simple example of using cached Hilbert attention."""
    print("Simple Cached Hilbert Attention Example")
    print("=" * 50)

    # Create attention module with factory function
    attention = create_cached_block_sparse_hilbert(
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
        sparsity_ratio=0.1,  # 90% sparse
        pattern_type="dilated_sparse",
        block_size=64,
        use_hilbert=True,
        precompute_seq_lengths=[4096, 8192, 16384],  # Pre-compute these sizes
    )

    # Example input
    batch_size = 2
    seq_length = 8192
    num_heads = 8
    head_dim = 64

    # Create random tensors
    q = torch.randn(batch_size, seq_length, num_heads, head_dim)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim)
    v = torch.randn(batch_size, seq_length, num_heads, head_dim)

    # Forward pass
    output = attention(q, k, v)
    print(f"Input shape: {q.shape}")
    print(f"Output shape: {output.shape}")

    # Get statistics
    stats = attention.get_pattern_stats()
    print("\nHilbert cache statistics:")
    print(f"  Cached orderings: {stats['hilbert_optimization']['cached_orderings']}")
    print(
        f"  Memory usage: {stats['hilbert_optimization']['memory_usage_bytes'] / 1024:.2f} KB"
    )


def transformer_layer_example():
    """Example of using cached Hilbert attention in a transformer layer."""
    print("\n\nTransformer Layer with Cached Hilbert Attention")
    print("=" * 50)

    class HilbertTransformerLayer(nn.Module):
        def __init__(self, d_model, num_heads, segment_lengths, dilation_rates):
            super().__init__()

            # Cached Hilbert attention
            self.attention = create_cached_block_sparse_hilbert(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                sparsity_ratio=0.1,
                pattern_type="dilated_sparse",
                block_size=64,
                use_hilbert=True,
                # Pre-compute common sequence lengths
                precompute_seq_lengths=[1024, 2048, 4096, 8192, 16384],
            )

            # Standard transformer components
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

            # Projection layers for Q, K, V
            self.qkv_proj = nn.Linear(d_model, 3 * num_heads * (d_model // num_heads))
            self.out_proj = nn.Linear(num_heads * (d_model // num_heads), d_model)

            self.num_heads = num_heads
            self.head_dim = d_model // num_heads

        def forward(self, x):
            batch_size, seq_len, d_model = x.shape

            # Self-attention with residual
            residual = x
            x = self.norm1(x)

            # Project to Q, K, V
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)

            # Apply cached Hilbert attention
            attn_output = self.attention(q, k, v)

            # Reshape and project output
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
            attn_output = self.out_proj(attn_output)

            x = residual + attn_output

            # FFN with residual
            residual = x
            x = self.norm2(x)
            x = residual + self.ffn(x)

            return x

    # Create model
    model = HilbertTransformerLayer(
        d_model=512,
        num_heads=8,
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
    )

    # Example forward pass
    x = torch.randn(2, 4096, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")


def performance_comparison_example():
    """Compare performance with and without caching."""
    print("\n\nPerformance Comparison Example")
    print("=" * 50)

    import time

    # Configuration
    config = {
        "segment_lengths": [2048, 4096, 8192],
        "dilation_rates": [1, 2, 4],
        "sparsity_ratio": 0.1,
        "pattern_type": "dilated_sparse",
        "block_size": 64,
    }

    # Test configuration
    batch_size = 2
    seq_length = 16384
    num_heads = 8
    head_dim = 64
    num_iterations = 10

    # Create input tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)
    v = torch.randn(batch_size, seq_length, num_heads, head_dim, device=device)

    # Version 1: Without pre-computation (standard)
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention_hilbert import (
        create_block_sparse_hilbert as create_standard_hilbert,
    )

    attention_standard = create_standard_hilbert(**config, use_hilbert=True).to(device)

    # Warmup
    for _ in range(3):
        _ = attention_standard(q, k, v)

    # Time standard version
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = attention_standard(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_standard = (time.time() - start) / num_iterations

    # Version 2: With pre-computation (cached)
    attention_cached = create_cached_block_sparse_hilbert(
        **config,
        use_hilbert=True,
        precompute_seq_lengths=[seq_length],
    ).to(device)

    # Warmup
    for _ in range(3):
        _ = attention_cached(q, k, v)

    # Time cached version
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = attention_cached(q, k, v)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_cached = (time.time() - start) / num_iterations

    # Results
    print(f"Device: {device}")
    print(f"Sequence length: {seq_length}")
    print(f"Standard Hilbert time: {time_standard * 1000:.2f} ms")
    print(f"Cached Hilbert time: {time_cached * 1000:.2f} ms")
    print(f"Speedup: {time_standard / time_cached:.2f}x")

    # Cache statistics
    stats = attention_cached.get_pattern_stats()
    print(
        f"\nCache memory overhead: {stats['hilbert_optimization']['memory_usage_bytes'] / 1024:.2f} KB"
    )


if __name__ == "__main__":
    simple_example()
    transformer_layer_example()
    performance_comparison_example()
