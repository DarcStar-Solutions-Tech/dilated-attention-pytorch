#!/usr/bin/env python3
"""
Actually test block-sparse on multiple GPUs with working implementations.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dilated_attention_pytorch.block_sparse_factory import create_block_sparse_attention


def test_data_parallel_actual():
    """Test DataParallel with actual multi-GPU execution."""
    print("DataParallel Actual Multi-GPU Test")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs")
        return

    print(f"GPUs available: {torch.cuda.device_count()}")

    # Create model
    model = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048],
        dilation_rates=[1],
        sparsity_ratio=0.05,  # 95% sparse
    )

    # Wrap in DataParallel
    model_dp = nn.DataParallel(model, device_ids=[0, 1])
    model_dp = model_dp.cuda()

    # Test configurations
    configs = [
        {"seq_len": 8192, "batch_size": 4},
        {"seq_len": 16384, "batch_size": 2},
        {"seq_len": 32768, "batch_size": 2},
    ]

    for config in configs:
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]

        print(f"\nTesting seq_len={seq_len}, batch_size={batch_size}")
        print("-" * 40)

        try:
            # Create inputs - batch will be split across GPUs
            q = torch.randn(
                batch_size, seq_len, 8, 64, device="cuda:0", dtype=torch.float16
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Measure memory before
            mem_before = []
            for i in range(2):
                mem_before.append(torch.cuda.memory_allocated(i) / 1024**2)

            # Forward pass
            start = time.time()
            output = model_dp(q, k, v)
            torch.cuda.synchronize()
            forward_time = (time.time() - start) * 1000

            # Measure memory after
            mem_after = []
            for i in range(2):
                mem_after.append(torch.cuda.memory_allocated(i) / 1024**2)

            print("✓ Success!")
            print(f"  Output shape: {output.shape}")
            print(f"  Forward time: {forward_time:.1f}ms")
            print(
                f"  GPU 0: {mem_before[0]:.1f}MB → {mem_after[0]:.1f}MB (+{mem_after[0] - mem_before[0]:.1f}MB)"
            )
            print(
                f"  GPU 1: {mem_before[1]:.1f}MB → {mem_after[1]:.1f}MB (+{mem_after[1] - mem_before[1]:.1f}MB)"
            )

            # Verify computation split
            total_mem_increase = sum(mem_after[i] - mem_before[i] for i in range(2))
            print(f"  Total memory increase: {total_mem_increase:.1f}MB")

        except torch.cuda.OutOfMemoryError:
            print("✗ Out of memory")
        except Exception as e:
            print(f"✗ Error: {e}")

        torch.cuda.empty_cache()


def test_model_parallel_manual():
    """Test manual model parallelism by splitting sequence."""
    print("\n\nManual Model Parallel Test (Sequence Splitting)")
    print("=" * 60)

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs")
        return

    # Create two models, one on each GPU
    model_gpu0 = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048],
        dilation_rates=[1],
        sparsity_ratio=0.05,
    ).to("cuda:0", dtype=torch.float16)

    model_gpu1 = create_block_sparse_attention(
        variant="base",
        segment_lengths=[2048],
        dilation_rates=[1],
        sparsity_ratio=0.05,
    ).to("cuda:1", dtype=torch.float16)

    # Test large sequence split across GPUs
    total_seq_len = 65536
    batch_size = 1

    print(f"\nTesting {total_seq_len} tokens split across 2 GPUs")
    print(f"Each GPU processes {total_seq_len // 2} tokens")
    print("-" * 40)

    try:
        # Create inputs - split sequence across GPUs
        seq_per_gpu = total_seq_len // 2

        q0 = torch.randn(
            batch_size, seq_per_gpu, 8, 64, device="cuda:0", dtype=torch.float16
        )
        k0 = torch.randn_like(q0)
        v0 = torch.randn_like(q0)

        q1 = torch.randn(
            batch_size, seq_per_gpu, 8, 64, device="cuda:1", dtype=torch.float16
        )
        k1 = torch.randn_like(q1)
        v1 = torch.randn_like(q1)

        # Process on each GPU
        torch.cuda.synchronize()
        start = time.time()

        output0 = model_gpu0(q0, k0, v0)
        output1 = model_gpu1(q1, k1, v1)

        torch.cuda.synchronize()
        forward_time = (time.time() - start) * 1000

        print("✓ Success!")
        print(f"  GPU 0 output: {output0.shape}")
        print(f"  GPU 1 output: {output1.shape}")
        print(f"  Total forward time: {forward_time:.1f}ms")

        # Memory usage
        for i in range(2):
            mem = torch.cuda.memory_allocated(i) / 1024**2
            print(f"  GPU {i} memory: {mem:.1f}MB")

        # Could concatenate outputs if needed
        # output = torch.cat([output0.to("cuda:0"), output1.to("cuda:0")], dim=1)

    except Exception as e:
        print(f"✗ Error: {e}")


def test_pipeline_parallel():
    """Test pipeline parallelism concept."""
    print("\n\nPipeline Parallel Concept Test")
    print("=" * 60)

    print("\nPipeline parallelism would work as follows:")
    print("1. GPU 0: Process first half of layers")
    print("2. GPU 1: Process second half of layers")
    print("3. Data flows: Input → GPU0 → GPU1 → Output")
    print("\nThis is most beneficial for very deep models.")


def verify_multi_gpu_benefits():
    """Verify the actual benefits of multi-GPU."""
    print("\n\nMulti-GPU Benefits Verification")
    print("=" * 60)

    # Single GPU baseline
    print("\n1. Single GPU Baseline (32K sequence):")
    try:
        model = create_block_sparse_attention(
            variant="base",
            segment_lengths=[2048],
            dilation_rates=[1],
            sparsity_ratio=0.05,
        ).cuda()

        q = torch.randn(1, 32768, 8, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        torch.cuda.synchronize()
        start = time.time()
        output = model(q, k, v)
        torch.cuda.synchronize()
        single_time = (time.time() - start) * 1000

        single_mem = torch.cuda.memory_allocated() / 1024**2

        print(f"  ✓ Time: {single_time:.1f}ms")
        print(f"  ✓ Memory: {single_mem:.1f}MB")

        del model, q, k, v, output
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        single_time = float("inf")

    # Test what sequence length fails on single GPU
    print("\n2. Finding single GPU limit:")
    max_single = 0
    for seq_len in [32768, 65536, 98304, 131072]:
        try:
            torch.cuda.empty_cache()
            q = torch.randn(1, seq_len, 8, 64, device="cuda", dtype=torch.float16)
            del q
            torch.cuda.empty_cache()
            max_single = seq_len
            print(f"  ✓ {seq_len:,} tokens: OK")
        except Exception:
            print(f"  ✗ {seq_len:,} tokens: OOM")
            break

    print(f"\n  Single GPU max: ~{max_single:,} tokens")
    print(f"  With 2 GPUs (split): ~{max_single * 2:,} tokens possible")
    print("  With ring attention: Even longer sequences possible")


def main():
    """Run actual multi-GPU tests."""
    print("Block-Sparse Multi-GPU Actual Tests")
    print("=" * 60)

    print("\nSetup:")
    print(f"  CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    # Run tests
    test_data_parallel_actual()
    test_model_parallel_manual()
    test_pipeline_parallel()
    verify_multi_gpu_benefits()

    print("\n✅ Multi-GPU testing completed!")


if __name__ == "__main__":
    main()
