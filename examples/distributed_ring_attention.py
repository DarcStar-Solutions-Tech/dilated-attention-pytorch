#!/usr/bin/env python3
"""
Distributed Ring Attention Example

This example demonstrates how to use Ring Attention in a truly distributed setting
across multiple GPUs. Ring Attention enables training with extremely long sequences
by distributing the key-value pairs across devices.

Requirements:
- Multiple GPUs (or run with torch.distributed.launch for simulation)
- PyTorch with distributed support

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=4 distributed_ring_attention.py

    # Multiple nodes
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT distributed_ring_attention.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dilated_attention_pytorch import (
    RingMultiheadDilatedAttention,
    create_multihead_dilated_attention,
)


def setup_distributed():
    """Initialize distributed training environment."""
    # Get rank and world size from torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def demonstrate_ring_attention_memory_savings():
    """Show memory savings of Ring Attention vs standard attention."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Configuration
    batch_size = 2
    seq_len = 32768  # 32K tokens - would OOM with standard attention
    num_heads = 16
    head_dim = 64
    embed_dim = num_heads * head_dim

    print(f"[Rank {rank}] Running Ring Attention demonstration...")
    print(f"[Rank {rank}] Sequence length: {seq_len:,} tokens")
    print(f"[Rank {rank}] World size: {world_size} GPUs")

    try:
        # Create Ring Attention module
        # Each GPU will only hold seq_len/world_size K/V tokens
        ring_attention = RingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4],
            dropout=0.1,
            ring_size=world_size,  # Automatically uses world_size
            device=device,
            dtype=torch.float16,  # Use fp16 for efficiency
        )

        # Wrap in DDP for distributed training
        ring_attention = DDP(ring_attention, device_ids=[local_rank])

        # Create inputs - each rank has the full sequence
        # In practice, sequences might be sharded differently
        inputs = torch.randn(
            batch_size, seq_len, embed_dim, device=device, dtype=torch.float16
        )

        # Measure memory before forward pass
        if rank == 0:
            allocated_before = torch.cuda.memory_allocated(device) / 1024**3
            print(
                f"\n[Rank 0] Memory allocated before forward: {allocated_before:.2f} GB"
            )

        # Forward pass - Ring Attention automatically handles K/V distribution
        with torch.cuda.amp.autocast():
            output = ring_attention(inputs, inputs, inputs)

        # Synchronize before measuring memory
        torch.cuda.synchronize()

        if rank == 0:
            allocated_after = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[Rank 0] Memory allocated after forward: {allocated_after:.2f} GB")
            print(
                f"[Rank 0] Memory used for attention: {allocated_after - allocated_before:.2f} GB"
            )

            # Estimate memory savings
            # Standard attention would need O(seq_len²) memory
            standard_memory_gb = (
                batch_size * num_heads * seq_len * seq_len * 2
            ) / 1024**3
            ring_memory_gb = allocated_after - allocated_before

            print("\n[Rank 0] Estimated memory usage:")
            print(f"  - Standard attention: {standard_memory_gb:.2f} GB (would OOM!)")
            print(f"  - Ring attention: {ring_memory_gb:.2f} GB")
            print(f"  - Memory reduction: {standard_memory_gb / ring_memory_gb:.1f}x")

        # Verify output shape
        assert output.shape == inputs.shape
        print(f"\n[Rank {rank}] ✓ Output shape correct: {output.shape}")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        raise
    finally:
        cleanup_distributed()


def demonstrate_training_loop():
    """Show how to use Ring Attention in a training loop."""
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Model configuration
    batch_size = 1
    seq_len = 16384  # 16K tokens
    embed_dim = 768
    num_heads = 12

    if rank == 0:
        print("\n[Rank 0] Ring Attention Training Loop Demo")
        print(f"[Rank 0] Sequence length: {seq_len:,} tokens")
        print(f"[Rank 0] Batch size: {batch_size}")

    # Create a simple model with Ring Attention
    class SimpleTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(50000, embed_dim)
            self.ring_attention = create_multihead_dilated_attention(
                "ring",
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=[1024, 2048, 4096],
                dilation_rates=[1, 2, 4],
                ring_size=world_size,
                device=device,
                dtype=torch.float16,
            )
            self.output_proj = torch.nn.Linear(embed_dim, 50000)

        def forward(self, input_ids):
            # Shape: [batch, seq_len]
            x = self.embedding(input_ids)  # [batch, seq_len, embed_dim]

            # Self-attention with Ring Attention
            x = self.ring_attention(x, x, x)  # [batch, seq_len, embed_dim]

            # Output projection
            logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
            return logits

    # Create model and wrap with DDP
    model = SimpleTransformer().to(device)
    model = DDP(model, device_ids=[local_rank])

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    for step in range(3):
        # Generate random data (in practice, load real data)
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
        labels = torch.randint(0, 50000, (batch_size, seq_len), device=device)

        # Forward pass
        with torch.cuda.amp.autocast():
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 50000), labels.reshape(-1)
            )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if rank == 0:
            print(f"[Rank 0] Step {step}: Loss = {loss.item():.4f}")

    if rank == 0:
        print("\n[Rank 0] ✓ Training loop completed successfully!")

    cleanup_distributed()


def main():
    """Run Ring Attention demonstrations."""
    # Check if distributed environment is set up
    if "RANK" not in os.environ:
        print(
            "ERROR: This script must be run with torchrun or torch.distributed.launch"
        )
        print("\nExample usage:")
        print("  torchrun --nproc_per_node=4 distributed_ring_attention.py")
        return

    # Run demonstrations
    print("=" * 60)
    print("Ring Attention Distributed Example")
    print("=" * 60)

    # Demo 1: Memory savings
    demonstrate_ring_attention_memory_savings()

    # Re-initialize for second demo
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n" + "=" * 60)
        print("Training Loop Demo")
        print("=" * 60)

    # Demo 2: Training loop
    demonstrate_training_loop()


if __name__ == "__main__":
    main()
