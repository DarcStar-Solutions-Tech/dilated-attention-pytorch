#!/usr/bin/env python3
"""
Benchmark existing ring dilated attention implementations on multiple GPUs.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing implementations
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)
from dilated_attention_pytorch.ring_dilated_attention_production import (
    RingDilatedAttentionProduction,
)
from dilated_attention_pytorch.ring_multihead_dilated_attention import (
    RingMultiheadDilatedAttention,
)


def setup(rank, world_size):
    """Initialize distributed training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()


def benchmark_distributed(
    rank: int,
    world_size: int,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    segment_lengths: list,
    dilation_rates: list,
    implementation: str,
    warmup_iters: int,
    test_iters: int,
):
    """Benchmark on a single GPU in distributed setting."""
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create model based on implementation
    if implementation == "v2_collective":
        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            use_flash_attention=False,  # Pascal doesn't support Flash Attention
            device=device,
            dtype=torch.float32,
        )
        impl_name = "RingDilatedAttentionV2Collective"
    elif implementation == "production":
        model = RingDilatedAttentionProduction(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
        )
        impl_name = "RingDilatedAttentionProduction"
    elif implementation == "multihead":
        embed_dim = num_heads * head_dim
        model = RingMultiheadDilatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            batch_first=True,
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
        )
        impl_name = "RingMultiheadDilatedAttention"
    else:
        raise ValueError(f"Unknown implementation: {implementation}")

    # Create input tensors
    if implementation == "multihead":
        # Multihead expects (batch, seq, embed_dim)
        embed_dim = num_heads * head_dim
        q = torch.randn(
            batch_size, seq_len, embed_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
    else:
        # Others expect (batch, seq, heads, dim)
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

    # Synchronize before starting
    dist.barrier()

    # Warmup
    for _ in range(warmup_iters):
        with torch.no_grad():
            _ = model(q, k, v)

    # Synchronize after warmup
    dist.barrier()
    torch.cuda.synchronize()

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1024**3  # GB

    # Benchmark
    dist.barrier()
    start_event.record()

    with torch.no_grad():
        for _ in range(test_iters):
            output = model(q, k, v)

    end_event.record()
    torch.cuda.synchronize()

    # Get timing
    forward_time = start_event.elapsed_time(end_event) / test_iters  # ms

    # Get memory usage
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3  # GB
    mem_used = peak_mem - initial_mem

    # Calculate throughput
    total_tokens = batch_size * seq_len * world_size  # Total across all GPUs
    throughput = (total_tokens / forward_time) * 1000  # tokens/sec

    # Only rank 0 prints results
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Multi-GPU Benchmark Results: {impl_name}")
        print(f"{'=' * 80}")
        print(f"GPUs: {world_size}")
        print(f"Total Sequence Length: {seq_len * world_size}")
        print(f"Sequence per GPU: {seq_len}")
        print(f"Batch Size: {batch_size}")
        print(f"Forward Time: {forward_time:.2f} ms")
        print(f"Total Throughput: {throughput:,.0f} tokens/sec")
        print(f"Throughput per GPU: {throughput / world_size:,.0f} tokens/sec")
        print(f"Memory per GPU: {mem_used:.2f} GB")
        print(f"Output shape: {output.shape}")

    cleanup()


def run_single_gpu_baseline(args, implementation):
    """Run single GPU baseline for comparison."""
    device = torch.device("cuda:0")

    # Test with full sequence on single GPU
    total_seq = args.seq_len * args.world_size

    print(f"\nTesting {implementation} with sequence length: {total_seq} on single GPU")

    try:
        # Create model
        if implementation == "v2_collective":
            model = RingDilatedAttentionV2Collective(
                segment_lengths=args.segment_lengths,
                dilation_rates=args.dilation_rates,
                dropout=0.0,
                ring_size=1,
                use_flash_attention=False,
                device=device,
                dtype=torch.float32,
            )
        elif implementation == "production":
            model = RingDilatedAttentionProduction(
                segment_lengths=args.segment_lengths,
                dilation_rates=args.dilation_rates,
                dropout=0.0,
                ring_size=1,
                device=device,
                dtype=torch.float32,
            )
        elif implementation == "multihead":
            embed_dim = args.num_heads * args.head_dim
            model = RingMultiheadDilatedAttention(
                embed_dim=embed_dim,
                num_heads=args.num_heads,
                segment_lengths=args.segment_lengths,
                dilation_rates=args.dilation_rates,
                dropout=0.0,
                batch_first=True,
                ring_size=1,
                device=device,
                dtype=torch.float32,
            )

        # Create input
        if implementation == "multihead":
            embed_dim = args.num_heads * args.head_dim
            q = torch.randn(
                args.batch_size,
                total_seq,
                embed_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            q = torch.randn(
                args.batch_size,
                total_seq,
                args.num_heads,
                args.head_dim,
                device=device,
                dtype=torch.float32,
            )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(q, k, v)

        torch.cuda.synchronize()

        # Time
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(q, k, v)
        torch.cuda.synchronize()

        forward_time = (time.time() - start) * 100  # ms per iteration
        throughput = (args.batch_size * total_seq / forward_time) * 1000
        memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Forward Time: {forward_time:.2f} ms")
        print(f"Throughput: {throughput:,.0f} tokens/sec")
        print(f"Memory: {memory:.2f} GB")

        return forward_time, throughput

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark existing ring implementations"
    )
    parser.add_argument(
        "--world_size", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--seq_len", type=int, default=8192, help="Sequence length per GPU"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head_dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--segment_lengths",
        type=int,
        nargs="+",
        default=[2048, 4096],
        help="Segment lengths for dilated attention",
    )
    parser.add_argument(
        "--dilation_rates",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Dilation rates for dilated attention",
    )
    parser.add_argument(
        "--warmup_iters", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--test_iters", type=int, default=20, help="Number of test iterations"
    )
    parser.add_argument(
        "--implementation",
        type=str,
        default="all",
        choices=["all", "v2_collective", "production", "multihead"],
        help="Which implementation to test",
    )

    args = parser.parse_args()

    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    available_gpus = torch.cuda.device_count()
    if available_gpus < args.world_size:
        print(f"Requested {args.world_size} GPUs but only {available_gpus} available")
        args.world_size = available_gpus

    print(f"Testing with {args.world_size} GPUs")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    if args.world_size > 1:
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")

    print("\nUsing FP32 throughout (correct for Pascal architecture)")

    # Test implementations
    implementations = []
    if args.implementation == "all":
        implementations = ["v2_collective", "production", "multihead"]
    else:
        implementations = [args.implementation]

    print(f"\n{'=' * 80}")
    print("Single GPU Baseline")
    print(f"{'=' * 80}")

    baseline_results = {}
    for impl in implementations:
        single_time, single_throughput = run_single_gpu_baseline(args, impl)
        baseline_results[impl] = (single_time, single_throughput)

    # Multi-GPU tests
    for impl in implementations:
        try:
            print(f"\nTesting {impl} on {args.world_size} GPUs...")

            # Spawn processes for distributed training
            mp.spawn(
                benchmark_distributed,
                args=(
                    args.world_size,
                    args.seq_len,
                    args.batch_size,
                    args.num_heads,
                    args.head_dim,
                    args.segment_lengths,
                    args.dilation_rates,
                    impl,
                    args.warmup_iters,
                    args.test_iters,
                ),
                nprocs=args.world_size,
                join=True,
            )

        except Exception as e:
            print(f"Failed to test {impl}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print("Performance Summary")
    print(f"{'=' * 80}")

    for impl, (single_time, single_throughput) in baseline_results.items():
        if single_time is not None:
            print(f"\n{impl}:")
            print(
                f"  Single GPU ({args.seq_len * args.world_size} tokens): {single_time:.2f} ms, {single_throughput:,.0f} tokens/sec"
            )


if __name__ == "__main__":
    main()
