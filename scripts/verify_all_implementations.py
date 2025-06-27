#!/usr/bin/env python3
"""
Quick verification script to ensure all implementations can be loaded and run.
"""

import torch

print("Verifying all dilated attention implementations...")
print("=" * 60)

# Test device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
print()

# Test parameters
batch_size = 1
seq_len = 2048
num_heads = 8
head_dim = 64
embed_dim = num_heads * head_dim
segment_lengths = [512, 1024, 2048]
dilation_rates = [1, 2, 4]

# Standard implementations
print("1. Testing standard implementations...")
try:
    from dilated_attention_pytorch import DilatedAttention, MultiheadDilatedAttention

    # Test DilatedAttention
    da = DilatedAttention(segment_lengths, dilation_rates).to(device)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    out = da(q, q, q)
    print(f"   ✓ DilatedAttention: output shape {out.shape}")

    # Test MultiheadDilatedAttention
    mha = MultiheadDilatedAttention(embed_dim, num_heads, segment_lengths, dilation_rates).to(
        device
    )
    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    out = mha(q, q, q)
    if isinstance(out, tuple):
        out = out[0]
    print(f"   ✓ MultiheadDilatedAttention: output shape {out.shape}")
except Exception as e:
    print(f"   ✗ Standard implementations failed: {e}")

# Improved implementations
print("\n2. Testing improved implementations...")
try:
    from dilated_attention_pytorch import (
        ImprovedDilatedAttention,
        ImprovedMultiheadDilatedAttention,
    )

    # Test ImprovedDilatedAttention
    ida = ImprovedDilatedAttention(segment_lengths, dilation_rates).to(device)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    out = ida(q, q, q)
    print(f"   ✓ ImprovedDilatedAttention: output shape {out.shape}")

    # Test ImprovedMultiheadDilatedAttention
    imha = ImprovedMultiheadDilatedAttention(
        embed_dim, num_heads, segment_lengths, dilation_rates
    ).to(device)
    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    out = imha(q, q, q)
    if isinstance(out, tuple):
        out = out[0]
    print(f"   ✓ ImprovedMultiheadDilatedAttention: output shape {out.shape}")
except Exception as e:
    print(f"   ✗ Improved implementations failed: {e}")

# Ring implementations
print("\n3. Testing ring implementations...")
try:
    from dilated_attention_pytorch.ring_dilated_attention import RingDilatedAttention
    from dilated_attention_pytorch.ring_multihead_dilated_attention import (
        RingMultiheadDilatedAttention,
    )

    # Choose ring_size that divides seq_len evenly
    ring_size = 4
    while seq_len % (ring_size * max(segment_lengths)) != 0 and ring_size > 1:
        ring_size -= 1

    # Test RingDilatedAttention
    rda = RingDilatedAttention(segment_lengths, dilation_rates, ring_size=ring_size).to(device)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    out = rda(q, q, q)
    print(f"   ✓ RingDilatedAttention: output shape {out.shape}, ring_size={ring_size}")

    # Test RingMultiheadDilatedAttention
    rmha = RingMultiheadDilatedAttention(
        embed_dim, num_heads, segment_lengths, dilation_rates, ring_size=ring_size
    ).to(device)
    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    out = rmha(q, q, q)
    if isinstance(out, tuple):
        out = out[0]
    print(f"   ✓ RingMultiheadDilatedAttention: output shape {out.shape}, ring_size={ring_size}")
except Exception as e:
    print(f"   ✗ Ring implementations failed: {e}")
    import traceback

    traceback.print_exc()

# Block sparse implementations
print("\n4. Testing block sparse implementations...")
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
        BlockSparseRingMultiheadDilatedAttention,
    )

    # Create proper SparsePatternConfig object
    sparsity_config = SparsePatternConfig(
        sparsity_ratio=0.9, block_size=64, local_window_size=256, pattern_type="local_window"
    )

    # Test BlockSparseRingDilatedAttention
    bsrda = BlockSparseRingDilatedAttention(
        segment_lengths, dilation_rates, sparse_config=sparsity_config
    ).to(device)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    out = bsrda(q, q, q)
    print(f"   ✓ BlockSparseRingDilatedAttention: output shape {out.shape}")

    # Test BlockSparseRingMultiheadDilatedAttention
    bsrmha = BlockSparseRingMultiheadDilatedAttention(
        embed_dim, num_heads, segment_lengths, dilation_rates, sparse_config=sparsity_config
    ).to(device)
    q = torch.randn(batch_size, seq_len, embed_dim, device=device)
    out = bsrmha(q, q, q)
    if isinstance(out, tuple):
        out = out[0]
    print(f"   ✓ BlockSparseRingMultiheadDilatedAttention: output shape {out.shape}")
except Exception as e:
    print(f"   ✗ Block sparse implementations failed: {e}")
    import traceback

    traceback.print_exc()

# Factory pattern
print("\n5. Testing factory pattern...")
try:
    from dilated_attention_pytorch import (
        create_dilated_attention,
        create_multihead_dilated_attention,
    )

    # Test dilated attention factory
    da_factory = create_dilated_attention(
        "auto", segment_lengths=segment_lengths, dilation_rates=dilation_rates
    )
    if da_factory:
        da_factory = da_factory.to(device)
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        out = da_factory(q, q, q)
        print(f"   ✓ create_dilated_attention: output shape {out.shape}")

    # Test multihead factory
    mha_factory = create_multihead_dilated_attention(
        "auto",
        embed_dim=embed_dim,
        num_heads=num_heads,
        segment_lengths=segment_lengths,
        dilation_rates=dilation_rates,
    )
    if mha_factory:
        mha_factory = mha_factory.to(device)
        q = torch.randn(batch_size, seq_len, embed_dim, device=device)
        out = mha_factory(q, q, q)
        if isinstance(out, tuple):
            out = out[0]
        print(f"   ✓ create_multihead_dilated_attention: output shape {out.shape}")
except Exception as e:
    print(f"   ✗ Factory pattern failed: {e}")

print("\n" + "=" * 60)
print("Verification complete!")
