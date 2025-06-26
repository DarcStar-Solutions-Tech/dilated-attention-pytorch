#!/usr/bin/env python3
"""
Example demonstrating the factory pattern for creating dilated attention modules.

This example shows how to use the new v0.2.0 factory pattern for easy creation
of dilated attention modules with automatic optimization.
"""

import torch
from torch import nn

# Import factory functions (v0.2.0+)
from dilated_attention_pytorch import (
    create_adaptive_sparse_attention,
    create_block_sparse_attention,
    create_dilated_attention,
    create_multihead_dilated_attention,
)

# For type hints
from dilated_attention_pytorch.core import DilatedAttentionConfig, MultiheadConfig


def example_1_auto_selection():
    """Example 1: Automatic selection of best implementation."""
    print("Example 1: Auto-selection based on hardware")
    print("-" * 50)

    # Create attention with auto-selection (recommended)
    attention = create_multihead_dilated_attention(
        "auto",  # Automatically selects best implementation
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        dropout=0.1,
    )

    print(f"Created: {type(attention).__name__}")
    print(f"Configuration: {attention.multihead_config}")

    # Use it like standard nn.MultiheadAttention
    batch_size = 2
    seq_len = 8192
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(batch_size, seq_len, 768, device=device)
    output = attention(x, x, x, is_causal=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_2_specific_implementation():
    """Example 2: Choose specific implementation."""
    print("Example 2: Specific implementation selection")
    print("-" * 50)

    implementations = ["standard", "improved", "ring"]

    for impl in implementations:
        try:
            attention = create_dilated_attention(
                impl,
                segment_lengths=[1024, 2048],
                dilation_rates=[1, 2],
                dropout=0.1,
            )
            print(f"{impl}: {type(attention).__name__}")
        except Exception as e:
            print(f"{impl}: Not available ({e})")

    print()


def example_3_block_sparse():
    """Example 3: Block-sparse attention for extreme efficiency."""
    print("Example 3: Block-sparse attention")
    print("-" * 50)

    # Create block-sparse attention with 95% sparsity
    attention = create_block_sparse_attention(
        sparsity_ratio=0.95,  # 95% sparse = 20x speedup
        pattern_type="dilated_sparse",
        embed_dim=512,
        num_heads=8,
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
    )

    print(f"Created: {type(attention).__name__}")
    print("Sparsity: 95% (20x theoretical speedup)")

    # Benchmark if available
    if torch.cuda.is_available():
        x = torch.randn(1, 4096, 512, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(3):
            _ = attention(x, x, x)

        # Time it
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            output = attention(x, x, x)
        end.record()

        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 10

        print(f"Average forward pass time: {elapsed:.2f} ms")

    print()


def example_4_adaptive_sparse():
    """Example 4: Adaptive sparse attention that learns sparsity patterns."""
    print("Example 4: Adaptive sparse attention")
    print("-" * 50)

    # Create adaptive sparse attention
    attention = create_adaptive_sparse_attention(
        embed_dim=768,
        num_heads=12,
        min_sparsity=0.1,  # 10% minimum sparsity
        max_sparsity=0.9,  # 90% maximum sparsity
        segment_lengths=[2048, 4096],
        dilation_rates=[1, 2],
    )

    print(f"Created: {type(attention).__name__}")
    print("Sparsity will be learned during training")
    print("Sparsity range: 10% - 90%")
    print()


def example_5_custom_transformer():
    """Example 5: Building a custom transformer with factory pattern."""
    print("Example 5: Custom transformer block")
    print("-" * 50)

    class CustomTransformerBlock(nn.Module):
        def __init__(
            self,
            embed_dim: int = 768,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            dropout: float = 0.1,
        ):
            super().__init__()

            # Use factory to create attention
            self.attention = create_multihead_dilated_attention(
                "auto",  # Auto-select best implementation
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[2048, 4096, 8192],
                dilation_rates=[1, 2, 4],
                dropout=dropout,
                layer_norm=True,  # MAGNETO-style layer norm
                gamma_init=1.0,  # MAGNETO initialization
            )

            # Standard transformer components
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

            mlp_dim = int(embed_dim * mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, embed_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
            # Pre-norm architecture
            attn_out = self.attention(self.norm1(x), is_causal=is_causal)
            x = x + attn_out

            x = x + self.mlp(self.norm2(x))
            return x

    # Create model
    model = CustomTransformerBlock(embed_dim=768, num_heads=12)
    print("Created custom transformer block")
    print(f"Attention type: {type(model.attention).__name__}")

    # Test it
    x = torch.randn(2, 8192, 768)
    output = model(x, is_causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_6_type_safe_config():
    """Example 6: Using type-safe configuration objects."""
    print("Example 6: Type-safe configuration")
    print("-" * 50)

    # Create configurations
    attention_config = DilatedAttentionConfig(
        segment_lengths=[2048, 4096, 8192, 16384],
        dilation_rates=[1, 2, 4, 8],
        dropout=0.1,
        use_tf32=True,  # Use TF32 on Ampere GPUs
    )

    multihead_config = MultiheadConfig(
        embed_dim=1024,
        num_heads=16,
        bias=False,
        layer_norm=True,
        layer_norm_eps=1e-6,
        gamma_init=1.0,
    )

    # Create attention with configs
    attention = create_multihead_dilated_attention(
        "improved",
        multihead_config=multihead_config,
        attention_config=attention_config,
    )

    print("Created attention with type-safe configs")
    print(f"Attention config: {attention_config}")
    print(f"Multihead config: {multihead_config}")
    print()


def main():
    """Run all examples."""
    print("Dilated Attention PyTorch - Factory Pattern Examples")
    print("=" * 60)
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Run examples
    example_1_auto_selection()
    example_2_specific_implementation()
    example_3_block_sparse()
    example_4_adaptive_sparse()
    example_5_custom_transformer()
    example_6_type_safe_config()

    print("All examples completed!")


if __name__ == "__main__":
    main()
