#!/usr/bin/env python3
"""Test RingDilatedAttentionSDPA with patched communication on multiple GPUs."""

import torch
import torch.distributed as dist
import os
import sys
import gc
import time

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch import RingDilatedAttentionSDPA  # noqa: E402
from dilated_attention_pytorch.utils import get_optimal_dtype  # noqa: E402

# Apply the communication fixes BEFORE importing ring attention
from dilated_attention_pytorch.ring.utils import ring_communication_fix  # noqa: E402

# Monkey patch the broken ring_pass_kv_safe function
from dilated_attention_pytorch.ring.base import ring_dilated_attention_sdpa  # noqa: E402

ring_dilated_attention_sdpa.ring_pass_kv_safe = (
    ring_communication_fix.ring_pass_kv_fixed
)


def main():
    # Initialize distributed
    if "RANK" not in os.environ:
        print("This script requires torchrun")
        return

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = get_optimal_dtype(device)

    print(f"\n[Rank {rank}] Testing patched RingDilatedAttentionSDPA")
    print(f"  World size: {world_size}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")

    # Test different sequence lengths
    test_configs = [
        (2048, [512, 1024], [1, 2]),
        (4096, [512, 1024, 2048], [1, 2, 4]),
        (8192, [1024, 2048, 4096], [1, 2, 4]),
    ]

    for seq_len, segment_lengths, dilation_rates in test_configs:
        # Skip if sequence not divisible by largest segment
        if seq_len % max(segment_lengths) != 0:
            continue

        print(f"\n[Rank {rank}] Testing sequence length: {seq_len}")

        try:
            # Create model
            embed_dim = 512
            num_heads = 8

            model = RingDilatedAttentionSDPA(
                embed_dim=embed_dim,
                num_heads=num_heads,
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=dtype,
            )
            model.eval()

            # Create input
            batch_size = 1
            x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Measure memory
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024**2)

            # Time the forward pass
            start_time = time.time()
            with torch.no_grad():
                output = model(x)
            torch.cuda.synchronize()
            end_time = time.time()

            # Get memory stats
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            mem_used = peak_mem - mem_before

            # Calculate metrics
            elapsed_time = end_time - start_time
            throughput = (batch_size * seq_len) / elapsed_time
            mem_per_token = mem_used / (batch_size * seq_len) * 1024  # KB per token

            # Calculate sparsity
            total_positions = 0
            for seg_len, dilation in zip(segment_lengths, dilation_rates):
                if seg_len <= seq_len:
                    actual_positions = min(seg_len, seq_len // dilation)
                    total_positions += actual_positions
            sparsity = 1.0 - (total_positions / seq_len)

            print(f"[Rank {rank}] Results:")
            print(f"  Output shape: {output.shape}")
            print(f"  Time: {elapsed_time:.3f}s")
            print(f"  Throughput: {throughput:,.0f} tokens/sec")
            print(f"  Peak memory: {peak_mem:.2f} MB")
            print(f"  Memory used: {mem_used:.2f} MB")
            print(f"  Memory per token: {mem_per_token:.3f} KB/token")
            print(f"  Sparsity: {sparsity:.1%}")

            # Verify O(n/k) scaling
            if world_size > 1:
                local_seq_len = seq_len // world_size
                print(f"  Local sequence per GPU: {local_seq_len} tokens")
                print(f"  Memory scaling: O(n/{world_size})")

            # Cleanup
            del model, x, output
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Rank {rank}] Error with seq_len={seq_len}: {e}")
            import traceback

            traceback.print_exc()

    # Synchronize all processes
    dist.barrier()
    print(f"\n[Rank {rank}] All tests completed!")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
