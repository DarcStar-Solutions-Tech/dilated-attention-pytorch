"""
Minimal Ring Attention test to identify the exact issue.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import gc


def minimal_ring_test(rank: int, world_size: int):
    """Minimal test combining attention + communication."""

    # Setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12363"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        print(f"\n[GPU {rank}] Starting minimal ring test")

        # Very small parameters to avoid memory issues
        batch = 1
        seq_len = 512  # Very small
        heads = 4
        dim = 32
        chunk_size = seq_len // world_size

        # Clear any existing memory
        torch.cuda.empty_cache()
        gc.collect()

        # Create tensors
        print(f"[GPU {rank}] Creating tensors...")
        q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Get local chunk
        start = rank * chunk_size
        end = (rank + 1) * chunk_size
        k_chunk = k[:, start:end].contiguous()
        v_chunk = v[:, start:end].contiguous()

        print(f"[GPU {rank}] Tensor shapes - Q: {q.shape}, K_chunk: {k_chunk.shape}")

        # Test 1: Local attention computation
        print(f"[GPU {rank}] Test 1: Local attention...")

        # Process own chunk
        q_local = q[:, start:end]

        # Important: Use correct reshaping
        # Don't do: q_local.reshape(batch * heads, chunk_size, dim)
        # Do: transpose first, then reshape
        q_t = q_local.transpose(1, 2)  # [batch, heads, chunk_size, dim]
        k_t = k_chunk.transpose(1, 2)  # [batch, heads, chunk_size, dim]
        v_t = v_chunk.transpose(1, 2)  # [batch, heads, chunk_size, dim]

        # Now we can safely compute attention
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        _ = torch.matmul(attn, v_t)

        print(f"[GPU {rank}] ✓ Local attention successful")

        # Test 2: Communication
        if world_size > 1:
            print(f"[GPU {rank}] Test 2: Ring exchange...")

            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            # Allocate receive buffers
            k_recv = torch.empty_like(k_chunk)
            v_recv = torch.empty_like(v_chunk)

            # Exchange
            dist.barrier()  # Sync before exchange

            req1 = dist.isend(k_chunk, dst=send_rank)
            req2 = dist.irecv(k_recv, src=recv_rank)
            req3 = dist.isend(v_chunk, dst=send_rank)
            req4 = dist.irecv(v_recv, src=recv_rank)

            req1.wait()
            req2.wait()
            req3.wait()
            req4.wait()

            print(f"[GPU {rank}] ✓ Ring exchange successful")

            # Test 3: Attention with exchanged chunks
            print(f"[GPU {rank}] Test 3: Attention with exchanged chunks...")

            k_t_new = k_recv.transpose(1, 2)
            v_t_new = v_recv.transpose(1, 2)

            scores_new = torch.matmul(q_t, k_t_new.transpose(-2, -1)) / (dim**0.5)
            attn_new = torch.softmax(scores_new, dim=-1)
            _ = torch.matmul(attn_new, v_t_new)

            print(f"[GPU {rank}] ✓ Attention with exchanged chunks successful")

        print(f"\n[GPU {rank}] ✅ ALL TESTS PASSED")

        # Cleanup
        dist.barrier()
        dist.destroy_process_group()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[GPU {rank}] ❌ ERROR: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        # Debug info
        print(f"\n[GPU {rank}] Debug info:")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Run minimal ring test."""
    print("Minimal Ring Attention Test")
    print("=" * 50)
    print("Testing with very small tensors to isolate the issue")

    world_size = 2

    if torch.cuda.device_count() < 2:
        print("Need 2 GPUs for this test")
        return

    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    gc.collect()

    try:
        mp.spawn(minimal_ring_test, args=(world_size,), nprocs=world_size, join=True)
        print("\n✅ Minimal ring test PASSED")
        print("\nConclusion: The issue appears when:")
        print("1. Using larger tensor sizes")
        print("2. OR incorrect tensor reshaping in attention computation")
        print("3. OR memory fragmentation from previous runs")
    except Exception as e:
        print(f"\n❌ Minimal ring test FAILED: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
