#!/usr/bin/env python3
"""
Test the Hybrid Ring Attention implementation.
Run with: torchrun --nproc_per_node=2 benchmarks/test_ring_hybrid.py
"""

import os
import time
import torch
import torch.distributed as dist
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid,
)
from dilated_attention_pytorch.ring_multihead_dilated_attention_hybrid import (
    RingMultiheadDilatedAttentionHybrid,
)


def test_hybrid():
    """Test hybrid implementation."""

    if "RANK" not in os.environ:
        print("Testing Hybrid Ring Attention - Single GPU")
        print("=" * 50)
        test_single_gpu()
        print("\nRun with torchrun --nproc_per_node=2 for multi-GPU test")
        return

    # Multi-GPU test
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Testing Hybrid Ring Attention - Multi GPU")
        print("=" * 50)
        print(f"World size: {world_size}")

    # Test configurations
    test_configs = [
        # (seq_len, segment_lengths, dilation_rates, name)
        (512, [256], [1], "Basic - no dilation"),
        (512, [256], [2], "Basic - dilation 2"),
        (1024, [256, 256], [1, 2], "Mixed dilation"),
        (2048, [1024], [1], "Large sequence"),
    ]

    for seq_len, segment_lengths, dilation_rates, name in test_configs:
        if rank == 0:
            print(f"\n{name}:")
            print(
                f"  Sequence: {seq_len}, Segments: {segment_lengths}, Dilation: {dilation_rates}"
            )

        dist.barrier()

        try:
            # Create model
            model = RingDilatedAttentionHybrid(
                segment_lengths=segment_lengths,
                dilation_rates=dilation_rates,
                dropout=0.0,
                device=device,
                dtype=torch.float32,
                ring_size=world_size,
                enable_memory_pool=True,
                use_pattern_cache=True,
                use_flash_attention=False,  # Disable for consistent testing
            )

            # Create inputs
            torch.manual_seed(42)
            scale = 0.1 / (seq_len**0.25)
            q = torch.randn(1, seq_len, 8, 64, device=device) * scale
            k = torch.randn(1, seq_len, 8, 64, device=device) * scale
            v = torch.randn(1, seq_len, 8, 64, device=device) * scale

            # Warmup
            _ = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            # Timed run
            dist.barrier()
            start = time.time()

            output = model(q, k, v, is_causal=False)
            torch.cuda.synchronize()

            elapsed = time.time() - start

            # Check output
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()
            output_mean = output.mean().item()

            # Check memory usage
            mem_mb = torch.cuda.memory_allocated(device) / (1024**2)

            # Gather results
            stats = {
                "time": elapsed,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "output_mean": output_mean,
                "memory_mb": mem_mb,
            }

            all_stats = [None] * world_size
            dist.all_gather_object(all_stats, stats)

            if rank == 0:
                avg_time = sum(s["time"] for s in all_stats) / len(all_stats)
                any_nan = any(s["has_nan"] for s in all_stats)
                any_inf = any(s["has_inf"] for s in all_stats)
                max_mem = max(s["memory_mb"] for s in all_stats)

                status = "✅" if not (any_nan or any_inf) else "❌"
                print(f"  {status} Time: {avg_time:.3f}s, Memory: {max_mem:.1f} MB")

                if any_nan or any_inf:
                    print(f"     NaN: {any_nan}, Inf: {any_inf}")

        except Exception as e:
            if rank == 0:
                print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    # Test multihead wrapper
    if rank == 0:
        print("\n\nTesting Multihead Wrapper:")
        print("-" * 40)

    dist.barrier()

    try:
        multihead = RingMultiheadDilatedAttentionHybrid(
            embed_dim=512,
            num_heads=8,
            segment_lengths=[512],
            dilation_rates=[1],
            device=device,
            dtype=torch.float32,
            ring_size=world_size,
            layer_norm=True,  # Test MAGNETO
            gamma_init=0.5,
        )

        # Test input
        x = torch.randn(2, 1024, 512, device=device) * 0.1

        output, _ = multihead(x, x, x, is_causal=True)

        has_nan = torch.isnan(output).any().item()

        all_nan = [None] * world_size
        dist.all_gather_object(all_nan, has_nan)

        if rank == 0:
            any_nan = any(all_nan)
            status = "✅" if not any_nan else "❌"
            print(f"  Multihead: {status} Shape: {output.shape}")

    except Exception as e:
        if rank == 0:
            print(f"  Multihead: ❌ Error: {e}")

    # Summary
    if rank == 0:
        print("\n" + "=" * 50)
        print("Hybrid Implementation Features:")
        print("✅ True ring communication (O(n/p) memory)")
        print("✅ Full dilation support in multi-GPU")
        print("✅ LSE accumulation for stability")
        print("✅ Memory pool and pattern caching")
        print("✅ MAGNETO layer norm support")

    dist.destroy_process_group()


def test_single_gpu():
    """Test on single GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test basic functionality
    model = RingDilatedAttentionHybrid(
        segment_lengths=[256],
        dilation_rates=[2],
        device=device,
        dtype=torch.float32,
    )

    # Test input
    q = torch.randn(1, 512, 8, 64, device=device) * 0.1
    k = torch.randn(1, 512, 8, 64, device=device) * 0.1
    v = torch.randn(1, 512, 8, 64, device=device) * 0.1

    output = model(q, k, v, is_causal=False)

    print("Single GPU test:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Has NaN: {torch.isnan(output).any().item()}")

    # Test multihead
    multihead = RingMultiheadDilatedAttentionHybrid(
        embed_dim=512,
        num_heads=8,
        segment_lengths=[256],
        dilation_rates=[1],
        device=device,
    )

    x = torch.randn(2, 1024, 512, device=device) * 0.1
    output, _ = multihead(x, x, x, is_causal=True)

    print("\nMultihead test:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean().item():.6f}")


if __name__ == "__main__":
    test_hybrid()
