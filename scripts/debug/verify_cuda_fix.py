#!/usr/bin/env python3
"""
Quick verification that the CUDA illegal memory access fix works.

Usage:
    python verify_cuda_fix.py                      # Single GPU
    torchrun --nproc_per_node=2 verify_cuda_fix.py  # Multi-GPU
"""

import os
import sys
import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dilated_attention_pytorch.ring_dilated_attention_v2_robust import (
    RingDilatedAttentionV2Robust,
)
from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
    RingDilatedAttentionV2Collective,
)


def setup_distributed():
    """Setup distributed if running with torchrun."""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

        return rank, world_size, device, True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, False


def test_implementation(name, model_class, rank, world_size, device):
    """Test a specific implementation."""
    print(f"\n{'=' * 60}")
    print(f"Testing {name} (Rank {rank}/{world_size})")
    print("=" * 60)

    try:
        # Create model
        model = model_class(
            segment_lengths=[2048, 4096, 8192],
            dilation_rates=[1, 2, 4],
            ring_size=world_size,
            device=device,
            dtype=torch.float16,
            use_flash_attention=False,  # Disable to isolate the issue
        )

        # Test multiple sequence lengths
        test_configs = [
            (4096, "Small"),
            (8192, "Medium"),
            (16384, "Large"),
        ]

        for seq_len, desc in test_configs:
            print(f"\n  Testing {desc} sequence (length={seq_len})...")

            # Create input tensors
            b, h, d = 2, 12, 64
            q = torch.randn(b, seq_len, h, d, device=device, dtype=torch.float16)
            k = v = q

            # Forward pass
            try:
                torch.cuda.synchronize()
                output = model(q, k, v, is_causal=False)
                torch.cuda.synchronize()

                # Verify output
                assert output.shape == q.shape, (
                    f"Shape mismatch: {output.shape} != {q.shape}"
                )
                assert not torch.isnan(output).any(), "Output contains NaN"
                assert not torch.isinf(output).any(), "Output contains Inf"

                print(f"    ✅ SUCCESS - Output shape: {output.shape}")

            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower():
                    print(f"    ❌ FAILED - OutOfMemoryError: {error_msg[:100]}...")
                else:
                    print(f"    ❌ FAILED - {type(e).__name__}: {error_msg}")
                    if "illegal memory access" in error_msg.lower():
                        print("    ⚠️  CUDA ILLEGAL MEMORY ACCESS DETECTED!")
                return False

        print(f"\n  ✅ All tests passed for {name}!")
        return True

    except Exception as e:
        print(f"  ❌ Failed to create model: {type(e).__name__}: {e}")
        return False


def main():
    """Main test function."""
    rank, world_size, device, is_distributed = setup_distributed()

    if rank == 0:
        print("\nCUDA Illegal Memory Access Fix Verification")
        print("=" * 80)
        print("\nEnvironment:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.cuda.is_available()}")
        print(f"  Distributed: {is_distributed}")
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Device: {device}")

    # Test implementations
    implementations = [
        ("Collective (baseline)", RingDilatedAttentionV2Collective),
        ("PyTorch Robust", RingDilatedAttentionV2Robust),
    ]

    results = {}
    for name, model_class in implementations:
        if is_distributed:
            dist.barrier()
        success = test_implementation(name, model_class, rank, world_size, device)
        results[name] = success

    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {name}: {status}")

        if all(results.values()):
            print("\n✅ All implementations passed. The CUDA fix is working correctly!")
        else:
            print("\n⚠️  Some implementations failed. The fix may need adjustment.")

    # Cleanup
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
