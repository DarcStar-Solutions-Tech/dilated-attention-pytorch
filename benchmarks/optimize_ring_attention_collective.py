#!/usr/bin/env python3
"""
Optimize the working RingDilatedAttentionV2Collective implementation.

This script tests various optimization strategies for the collective
operations version that we know works correctly.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add parent directory to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dilated_attention_pytorch import RingDilatedAttentionV2Collective


def setup(rank, world_size, backend="nccl"):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"

    # Initialize process group
    dist.init_process_group(
        backend=backend, init_method="env://", world_size=world_size, rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def run_optimized_test(rank, world_size, args, results_queue):
    """Run ring attention with various optimizations."""
    setup(rank, world_size, backend=args.backend)

    device = torch.device(f"cuda:{rank}")
    dtype = torch.float16 if args.use_fp16 else torch.float32

    # Create ring attention module
    segment_lengths = [1024, 2048, 4096]
    dilation_rates = [1, 2, 4]

    # Filter segment lengths
    segment_lengths = [s for s in segment_lengths if s <= args.seq_len]
    dilation_rates = dilation_rates[: len(segment_lengths)]

    try:
        # Enable various optimizations
        if args.use_torch_compile and hasattr(torch, "compile"):
            torch._dynamo.config.cache_size_limit = 64

        model = RingDilatedAttentionV2Collective(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            dropout=0.0,
            ring_size=world_size,
            device=device,
            dtype=dtype,
            enable_memory_pool=args.use_memory_pool,
            enable_profiling=args.enable_profiling,
            use_pattern_cache=args.use_pattern_cache,
        )

        # Compile model if requested
        if args.use_torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model, mode=args.compile_mode)
            if rank == 0:
                print(f"Model compiled with mode: {args.compile_mode}")

        # Create inputs
        batch_size = args.batch_size
        seq_len = args.seq_len
        num_heads = args.num_heads
        head_dim = args.head_dim

        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
        )

        # Set environment variables for NCCL optimizations
        if args.nccl_optimizations:
            os.environ["NCCL_TREE_THRESHOLD"] = "0"  # Always use tree algorithm
            os.environ["NCCL_LL_THRESHOLD"] = (
                "0"  # Disable LL algorithm for small messages
            )
            os.environ["NCCL_NET_GDR_LEVEL"] = "5"  # Enable GPU Direct RDMA

        # Warmup
        warmup_iters = 5
        for _ in range(warmup_iters):
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                _ = model(q, k, v)

        # Synchronize
        dist.barrier()
        torch.cuda.synchronize()

        # Profile if requested
        if args.profile and rank == 0:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            prof.start()

        # Benchmark
        start_time = time.perf_counter()

        num_iters = args.num_iters
        for i in range(num_iters):
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output = model(q, k, v)

            # Optional gradient computation
            if args.test_backward:
                loss = output.mean()
                loss.backward()

        torch.cuda.synchronize()
        dist.barrier()

        end_time = time.perf_counter()

        # Stop profiling
        if args.profile and rank == 0:
            prof.stop()
            prof.export_chrome_trace(f"ring_attention_trace_rank{rank}.json")
            print(f"Profile saved to ring_attention_trace_rank{rank}.json")

        avg_time = (end_time - start_time) / num_iters * 1000  # ms

        # Get memory stats
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        # Measure communication time separately
        comm_start = time.perf_counter()
        for _ in range(10):
            # Simulate the all_gather operation
            tensor = torch.randn(
                batch_size,
                seq_len // world_size,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            )
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(tensor_list, tensor)
        torch.cuda.synchronize()
        comm_end = time.perf_counter()
        comm_time = (comm_end - comm_start) / 10 * 1000  # ms

        # Send results from rank 0
        if rank == 0:
            results_queue.put(
                {
                    "success": True,
                    "avg_time": avg_time,
                    "memory_mb": memory_mb,
                    "comm_time": comm_time,
                    "dtype": str(dtype),
                    "backend": args.backend,
                }
            )

    except Exception as e:
        if rank == 0:
            results_queue.put({"success": False, "error": str(e)})
        import traceback

        traceback.print_exc()

    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Optimize Ring Attention Collective")
    # Basic settings
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Dimension per head")
    parser.add_argument(
        "--num-iters", type=int, default=20, help="Number of iterations"
    )

    # Optimization settings
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend",
    )
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument(
        "--use-amp", action="store_true", help="Use automatic mixed precision"
    )
    parser.add_argument(
        "--use-torch-compile", action="store_true", help="Use torch.compile"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Torch compile mode",
    )
    parser.add_argument(
        "--use-memory-pool", action="store_true", help="Enable memory pool"
    )
    parser.add_argument(
        "--use-pattern-cache",
        action="store_true",
        default=True,
        help="Enable pattern cache",
    )
    parser.add_argument(
        "--nccl-optimizations",
        action="store_true",
        help="Enable NCCL-specific optimizations",
    )
    parser.add_argument(
        "--test-backward", action="store_true", help="Test backward pass"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--enable-profiling", action="store_true", help="Enable internal profiling"
    )

    args = parser.parse_args()

    # Check GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus < args.world_size:
        print(f"Error: Requested {args.world_size} GPUs but only {n_gpus} available")
        return

    # Print configuration
    print(f"\n{'=' * 60}")
    print("Ring Attention Collective Optimization Test")
    print(f"{'=' * 60}")
    print("Configuration:")
    print(f"  World size: {args.world_size} GPUs")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num heads: {args.num_heads}")
    print(f"  Head dim: {args.head_dim}")
    print("\nOptimizations:")
    print(f"  Backend: {args.backend}")
    print(f"  FP16: {args.use_fp16}")
    print(f"  AMP: {args.use_amp}")
    print(f"  Torch compile: {args.use_torch_compile}")
    if args.use_torch_compile:
        print(f"    Compile mode: {args.compile_mode}")
    print(f"  Memory pool: {args.use_memory_pool}")
    print(f"  Pattern cache: {args.use_pattern_cache}")
    print(f"  NCCL optimizations: {args.nccl_optimizations}")
    print(f"  Test backward: {args.test_backward}")
    print(f"{'=' * 60}\n")

    # Create results queue
    results_queue = mp.Queue()

    # Spawn processes
    mp.spawn(
        run_optimized_test,
        args=(args.world_size, args, results_queue),
        nprocs=args.world_size,
        join=True,
    )

    # Get results
    results = results_queue.get()

    if results["success"]:
        print(f"\n{'=' * 60}")
        print("Results:")
        print(f"{'=' * 60}")
        print(f"Average time: {results['avg_time']:.2f} ms")
        print(f"Communication time: {results['comm_time']:.2f} ms")
        print(f"Computation time: {results['avg_time'] - results['comm_time']:.2f} ms")
        print(f"Memory per GPU: {results['memory_mb']:.2f} MB")
        print(f"Total memory: {results['memory_mb'] * args.world_size:.2f} MB")
        print(
            f"Throughput: {args.batch_size * args.seq_len / (results['avg_time'] / 1000):.0f} tokens/s"
        )
        print(f"Dtype: {results['dtype']}")
        print(f"Backend: {results['backend']}")

        # Calculate efficiency
        comm_ratio = results["comm_time"] / results["avg_time"] * 100
        print("\nEfficiency:")
        print(f"  Communication overhead: {comm_ratio:.1f}%")
        print(f"  Computation ratio: {100 - comm_ratio:.1f}%")
    else:
        print(f"\nError: {results['error']}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
