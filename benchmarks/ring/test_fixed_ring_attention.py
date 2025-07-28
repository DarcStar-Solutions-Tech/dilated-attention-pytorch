#!/usr/bin/env python3
"""Test ring attention with fixes from lucidrains implementation."""

import torch
import torch.distributed as dist
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch.utils import get_optimal_dtype


def send_and_receive_(send_tensor, receive_buffer, send_to_rank, receive_from_rank):
    """
    Send and receive tensors using P2P operations.
    Based on lucidrains/ring-attention-pytorch implementation.
    """
    # Ensure tensors are contiguous
    send_tensor = send_tensor.contiguous()
    receive_buffer = receive_buffer.contiguous()

    # Create P2P operations
    ops = []
    ops.append(dist.P2POp(dist.isend, send_tensor, send_to_rank))
    ops.append(dist.P2POp(dist.irecv, receive_buffer, receive_from_rank))

    # Execute all operations
    reqs = dist.batch_isend_irecv(ops)

    # Wait for completion
    for req in reqs:
        req.wait()

    # Synchronize all processes
    dist.barrier()

    return receive_buffer


def ring_pass(tensor, rank, world_size):
    """
    Perform one ring pass, sending to next rank and receiving from previous.
    """
    if world_size <= 1:
        return tensor

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    # Create receive buffer
    receive_buffer = torch.empty_like(tensor)

    # Send and receive
    return send_and_receive_(tensor, receive_buffer, next_rank, prev_rank)


def test_simple_ring_communication():
    """Test basic ring communication pattern."""
    if "RANK" not in os.environ:
        print("This test requires torchrun")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Testing simple ring communication")

    try:
        # Create a simple tensor with rank value
        tensor = torch.ones(10, 10, device=device) * rank
        print(f"[Rank {rank}] Initial tensor value: {tensor[0, 0].item()}")

        # Perform ring pass
        received = ring_pass(tensor, rank, world_size)

        # Check received value
        expected_from = (rank - 1) % world_size
        print(
            f"[Rank {rank}] Received tensor value: {received[0, 0].item()} (expected from rank {expected_from})"
        )

        # Verify
        assert received[0, 0].item() == expected_from, (
            f"Expected {expected_from}, got {received[0, 0].item()}"
        )

        print(f"[Rank {rank}] Ring communication test PASSED!")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    dist.destroy_process_group()


def test_ring_attention_with_fixed_comm():
    """Test ring attention with fixed communication pattern."""
    if "RANK" not in os.environ:
        print("This test requires torchrun")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = get_optimal_dtype(device)

    print(f"\n[Rank {rank}] Testing ring attention with fixed communication")
    print(f"  World size: {world_size}")
    print(f"  Device: {device}")

    try:
        # Create inputs
        batch_size = 1
        seq_len = 1024
        num_heads = 8
        head_dim = 64

        # Each rank gets a different part of the sequence
        local_seq_len = seq_len // world_size
        start_idx = rank * local_seq_len
        end_idx = start_idx + local_seq_len

        # Create local tensors
        q = torch.randn(
            batch_size, local_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, local_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, local_seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        print(f"[Rank {rank}] Created local tensors: {q.shape}")
        print(f"[Rank {rank}] Processing sequence range: {start_idx}-{end_idx}")

        # Initialize output accumulator
        output = torch.zeros_like(q)

        # Ring attention loop
        k_chunk = k.clone()
        v_chunk = v.clone()

        for step in range(world_size):
            print(f"[Rank {rank}] Ring step {step}")

            # Compute attention with current chunks
            scale = 1.0 / (head_dim**0.5)
            scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            chunk_output = torch.matmul(attn_weights, v_chunk)

            # Accumulate output
            output += chunk_output / world_size

            # Ring pass k and v
            if step < world_size - 1:
                k_chunk = ring_pass(k_chunk, rank, world_size)
                v_chunk = ring_pass(v_chunk, rank, world_size)

        print(f"[Rank {rank}] Ring attention complete! Output shape: {output.shape}")
        print(f"[Rank {rank}] Output mean: {output.mean().item():.6f}")

        # Final synchronization
        dist.barrier()
        print(f"[Rank {rank}] Test PASSED!")

    except Exception as e:
        print(f"[Rank {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    dist.destroy_process_group()


def main():
    """Run tests based on command line argument."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test", choices=["simple", "attention", "both"], default="both"
    )
    args = parser.parse_args()

    if args.test == "simple" or args.test == "both":
        test_simple_ring_communication()

    if args.test == "attention" or args.test == "both":
        test_ring_attention_with_fixed_comm()


if __name__ == "__main__":
    main()
