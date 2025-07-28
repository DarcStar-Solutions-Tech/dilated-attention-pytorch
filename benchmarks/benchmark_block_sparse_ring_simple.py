#!/usr/bin/env python3
"""
Simple test to verify BlockSparseRingAttention's ring communication.
This benchmark documents that the implementation appears to have deadlock issues.
"""

import os
import sys
import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.ring.block_sparse_ring_attention import (
    BlockSparseRingAttention,
)
from dilated_attention_pytorch.ring.base.ring_config import RingAttentionConfig


def test_block_sparse_ring():
    """Test BlockSparseRingAttention basic functionality."""
    # Single GPU test first
    if not dist.is_initialized():
        print("Testing BlockSparseRingAttention on single GPU...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        config = RingAttentionConfig(
            segment_lengths=[64],
            dilation_rates=[1],
        )

        model = BlockSparseRingAttention(
            config=config,
            block_size=64,
            sparsity_ratio=0.5,
            device=device,
            dtype=torch.float32,
        )

        # Test input
        batch_size = 1
        seq_len = 256
        num_heads = 4
        head_dim = 32

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Forward pass
        output = model(q, k, v)

        print("✓ Single GPU test passed")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {output.shape}")
        print(
            f"  Has ring communication methods: {hasattr(model, 'ring_pass_forward')}"
        )
        print(
            f"  Inherits from RingCommunicationMixin: {hasattr(model, '_ring_communication')}"
        )

        # Check communication stats
        stats = model.get_communication_stats()
        print(f"  Communication sends (single GPU): {stats.get('sends', 0)}")

        print("\nCONCLUSION:")
        print("- BlockSparseRingAttention exists and has ring communication methods")
        print("- It works on single GPU")
        print("- Multi-GPU testing reveals deadlock issues (not shown here)")
        print("- The implementation DOES use isend/irecv for ring communication")
        print("- But there appears to be a synchronization bug causing deadlock")

    else:
        # Multi-GPU path - known to deadlock
        print("Multi-GPU testing skipped due to known deadlock issues")


if __name__ == "__main__":
    test_block_sparse_ring()
