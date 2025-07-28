#!/usr/bin/env python3
"""
Verify PyTorch multi-GPU setup and configuration.
"""

import torch
import torch.distributed as dist
import os
from datetime import datetime


def check_gpu_availability():
    """Check and report GPU availability."""
    print("=" * 60)
    print("PyTorch Multi-GPU Verification")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check PyTorch version
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # List all GPUs
        print("\nGPU Details:")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(
                f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            )

            # Check current memory usage
            if torch.cuda.memory_allocated(i) > 0:
                print(
                    f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB"
                )
                print(
                    f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
                )
    else:
        print("CUDA is not available!")
        return False

    return True


def test_multi_gpu_operations():
    """Test basic multi-GPU operations."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("\nMulti-GPU test skipped (requires at least 2 GPUs)")
        return

    print("\n" + "=" * 60)
    print("Testing Multi-GPU Operations")
    print("=" * 60)

    # Test 1: Create tensors on different GPUs
    print("\n1. Creating tensors on different GPUs:")
    try:
        tensor_gpu0 = torch.randn(1000, 1000, device="cuda:0")
        tensor_gpu1 = torch.randn(1000, 1000, device="cuda:1")
        print(f"   ✓ Created tensor on GPU 0: shape {tensor_gpu0.shape}")
        print(f"   ✓ Created tensor on GPU 1: shape {tensor_gpu1.shape}")
    except Exception as e:
        print(f"   ✗ Error creating tensors: {e}")
        return

    # Test 2: Move tensor between GPUs
    print("\n2. Moving tensors between GPUs:")
    try:
        tensor_moved = tensor_gpu0.to("cuda:1")
        print("   ✓ Moved tensor from GPU 0 to GPU 1")
        assert tensor_moved.device.index == 1
        print("   ✓ Verified tensor is on GPU 1")
    except Exception as e:
        print(f"   ✗ Error moving tensor: {e}")

    # Test 3: DataParallel wrapper
    print("\n3. Testing DataParallel:")
    try:
        model = torch.nn.Linear(100, 10)
        model_dp = torch.nn.DataParallel(model)
        model_dp = model_dp.cuda()

        # Test forward pass
        input_data = torch.randn(32, 100).cuda()
        output = model_dp(input_data)
        print("   ✓ DataParallel model created and tested")
        print(f"   ✓ Input shape: {input_data.shape}, Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error with DataParallel: {e}")

    # Test 4: Check distributed backend availability
    print("\n4. Checking distributed backends:")
    backends = ["nccl", "gloo", "mpi"]
    for backend in backends:
        available = dist.is_available() and (
            (backend == "nccl" and dist.is_nccl_available())
            or (backend == "gloo" and dist.is_gloo_available())
            or (backend == "mpi" and dist.is_mpi_available())
        )
        status = "✓" if available else "✗"
        print(f"   {status} {backend}: {'Available' if available else 'Not available'}")


def test_memory_info():
    """Display memory information for each GPU."""
    if not torch.cuda.is_available():
        return

    print("\n" + "=" * 60)
    print("GPU Memory Information")
    print("=" * 60)

    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"\nGPU {i} Memory:")

        # Get memory stats
        free_memory = torch.cuda.mem_get_info(i)[0] / 1024**3
        total_memory = torch.cuda.mem_get_info(i)[1] / 1024**3
        used_memory = total_memory - free_memory

        print(f"  Total: {total_memory:.2f} GB")
        print(
            f"  Used:  {used_memory:.2f} GB ({used_memory / total_memory * 100:.1f}%)"
        )
        print(
            f"  Free:  {free_memory:.2f} GB ({free_memory / total_memory * 100:.1f}%)"
        )


def check_environment_variables():
    """Check relevant environment variables for multi-GPU setup."""
    print("\n" + "=" * 60)
    print("Environment Variables")
    print("=" * 60)

    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NCCL_DEBUG",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]

    print("\nRelevant environment variables:")
    for var in env_vars:
        value = os.environ.get(var, "<not set>")
        print(f"  {var}: {value}")


if __name__ == "__main__":
    # Run all checks
    if check_gpu_availability():
        test_multi_gpu_operations()
        test_memory_info()

    check_environment_variables()

    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)
