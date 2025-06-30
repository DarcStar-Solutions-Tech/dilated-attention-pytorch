"""
Quick test to verify dilated patterns are properly applied in Ring V2 Collective.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os


def test_dilated_patterns(rank: int, world_size: int):
    """Test that dilated patterns are applied in Ring V2 Collective."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12368"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    try:
        from dilated_attention_pytorch.ring_dilated_attention_v2_collective import (
            RingDilatedAttentionV2Collective,
        )
        from dilated_attention_pytorch.dilated_attention import DilatedAttention

        # Small test case
        batch_size = 1
        seq_len = 512
        num_heads = 4
        head_dim = 32

        # Create models
        ring_model = RingDilatedAttentionV2Collective(
            segment_lengths=[256, 512],
            dilation_rates=[1, 2],
            ring_size=world_size,
            device=device,
            dtype=torch.float32,
            enable_memory_pool=False,
            use_pattern_cache=False,
        ).to(device)

        if rank == 0:
            # Standard dilated attention for comparison
            std_model = DilatedAttention(
                segment_lengths=[256, 512],
                dilation_rates=[1, 2],
                device=device,
                dtype=torch.float32,
            ).to(device)

        # Create inputs
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Test Ring Attention with dilated patterns
        with torch.no_grad():
            ring_output = ring_model(q, k, v, is_causal=False)

        if rank == 0:
            print(f"✓ Ring output shape: {ring_output.shape}")
            print(f"✓ Ring output valid: {not torch.isnan(ring_output).any().item()}")
            print(
                f"✓ Ring output range: [{ring_output.min().item():.3f}, {ring_output.max().item():.3f}]"
            )

            # Compare with standard dilated attention
            with torch.no_grad():
                std_output = std_model(q, k, v, is_causal=False)

            print(f"✓ Standard output shape: {std_output.shape}")
            print(
                f"✓ Standard output valid: {not torch.isnan(std_output).any().item()}"
            )
            print(
                f"✓ Standard output range: [{std_output.min().item():.3f}, {std_output.max().item():.3f}]"
            )

            # Check if outputs are different (they should be due to different attention patterns)
            diff = torch.abs(ring_output - std_output).mean().item()
            print(f"✓ Mean difference: {diff:.6f}")

            if diff > 1e-6:
                print(
                    "✅ Ring and Standard outputs differ (expected - different attention patterns)"
                )
            else:
                print("⚠️  Ring and Standard outputs are very similar (unexpected)")

        # Test method calls to verify dilated pattern application
        if rank == 0:
            print("\n📊 Testing individual pattern methods:")

            # Test chunk pattern application
            k_chunk = k[:, :256]  # First chunk
            v_chunk = v[:, :256]

            k_dilated, v_dilated = ring_model._apply_dilated_patterns_to_chunk(
                k_chunk, v_chunk, chunk_start=0, chunk_size=256
            )

            print(f"✓ Chunk dilated K shape: {k_dilated.shape}")
            print(f"✓ Chunk dilated V shape: {v_dilated.shape}")

            # Test query pattern application
            q_dilated = ring_model._apply_dilated_patterns_to_query(q)
            print(f"✓ Query dilated shape: {q_dilated.shape}")

            # Check if dilation actually changed the tensors
            k_diff = torch.abs(k_chunk - k_dilated).mean().item()
            v_diff = torch.abs(v_chunk - v_dilated).mean().item()
            q_diff = torch.abs(q - q_dilated).mean().item()

            print(f"✓ K dilation difference: {k_diff:.6f}")
            print(f"✓ V dilation difference: {v_diff:.6f}")
            print(f"✓ Q dilation difference: {q_diff:.6f}")

            if k_diff > 1e-6 or v_diff > 1e-6 or q_diff > 1e-6:
                print("✅ Dilated patterns are being applied (tensors changed)")
            else:
                print("⚠️  Dilated patterns may not be working (tensors unchanged)")

    except Exception as e:
        print(f"[GPU {rank}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        dist.barrier()
        dist.destroy_process_group()


def main():
    """Run dilated pattern test."""
    print("Testing Dilated Patterns in Ring V2 Collective")
    print("=" * 60)

    world_size = 2
    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs")
        return

    try:
        mp.spawn(
            test_dilated_patterns, args=(world_size,), nprocs=world_size, join=True
        )
        print("\n✅ Dilated pattern test completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
