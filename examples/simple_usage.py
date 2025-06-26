#!/usr/bin/env python3
"""
Simple usage example for dilated attention modules.

This example shows the most common usage patterns for the dilated attention
implementation.
"""

import torch
from torch import nn

# Import the factory function (v0.2.0+ recommended approach)
# For backward compatibility, you can still use direct imports
from dilated_attention_pytorch import MultiheadDilatedAttention, create_multihead_dilated_attention


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print()

    # Example 1: Quick start with factory (recommended)
    print("Example 1: Factory Pattern (Recommended)")
    print("-" * 40)

    # Create attention module - auto-selects best implementation
    attention = create_multihead_dilated_attention(
        "auto",  # or "standard", "improved", "ring"
        embed_dim=512,
        num_heads=8,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        dropout=0.1,
        device=device,
        dtype=dtype,
    )

    # Create input
    batch_size = 2
    seq_len = 8192  # Must be divisible by largest segment length
    x = torch.randn(batch_size, seq_len, 512, device=device, dtype=dtype)

    # Forward pass
    output = attention(x, x, x, is_causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()

    # Example 2: Direct instantiation (backward compatible)
    print("Example 2: Direct Import (Backward Compatible)")
    print("-" * 40)

    # Create attention module directly
    attention_direct = MultiheadDilatedAttention(
        embed_dim=512,
        num_heads=8,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        dropout=0.1,
        device=device,
        dtype=dtype,
    )

    # Forward pass
    output_direct = attention_direct(x, x, x, is_causal=True)
    print(f"Output shape: {output_direct.shape}")
    print()

    # Example 3: In a transformer model
    print("Example 3: In a Transformer Model")
    print("-" * 40)

    class SimpleTransformer(nn.Module):
        def __init__(self, embed_dim=512, num_heads=8, num_layers=6):
            super().__init__()

            self.layers = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attention": create_multihead_dilated_attention(
                                "auto",
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                segment_lengths=[2048, 4096, 8192],
                                dilation_rates=[1, 2, 4],
                                dropout=0.1,
                            ),
                            "norm1": nn.LayerNorm(embed_dim),
                            "norm2": nn.LayerNorm(embed_dim),
                            "mlp": nn.Sequential(
                                nn.Linear(embed_dim, embed_dim * 4),
                                nn.GELU(),
                                nn.Linear(embed_dim * 4, embed_dim),
                            ),
                        }
                    )
                    for _ in range(num_layers)
                ]
            )

        def forward(self, x, is_causal=False):
            for layer in self.layers:
                # Attention block
                attn_out = layer["attention"](layer["norm1"](x), is_causal=is_causal)
                x = x + attn_out

                # MLP block
                x = x + layer["mlp"](layer["norm2"](x))

            return x

    # Create model
    model = SimpleTransformer(embed_dim=512, num_heads=8, num_layers=6)
    model = model.to(device).to(dtype)

    # Forward pass
    output = model(x, is_causal=True)
    print(f"Transformer output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Example 4: Different sequence lengths
    print("Example 4: Different Sequence Lengths")
    print("-" * 40)

    # The sequence length must be divisible by the largest segment length
    for seq_len in [8192, 16384, 32768]:
        x = torch.randn(1, seq_len, 512, device=device, dtype=dtype)
        output = attention(x, x, x, is_causal=True)
        print(f"Seq length {seq_len}: output shape {output.shape}")

    print()
    print("Examples completed!")


if __name__ == "__main__":
    main()
