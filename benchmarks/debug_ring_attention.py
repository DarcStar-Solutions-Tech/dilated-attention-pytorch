"""
Debug Ring Attention to isolate the source of CUDA errors.

This script tests each component separately to identify the problem.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import traceback


def test_communication_only(rank: int, world_size: int):
    """Test only the distributed communication part."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12360"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        print(f"\n[GPU {rank}] Testing COMMUNICATION ONLY")

        # Test 1: Simple tensor exchange
        print(f"[GPU {rank}] Test 1: Simple send/recv")
        test_tensor = torch.ones(100, device=device) * rank
        recv_tensor = torch.zeros(100, device=device)

        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size

        if rank == 0:
            dist.send(test_tensor, dst=send_rank)
            dist.recv(recv_tensor, src=recv_rank)
        else:
            dist.recv(recv_tensor, src=recv_rank)
            dist.send(test_tensor, dst=send_rank)

        print(f"[GPU {rank}] ✓ Simple send/recv successful")

        # Test 2: isend/irecv with larger tensors
        print(f"\n[GPU {rank}] Test 2: isend/irecv with 1MB tensors")
        size = 256 * 1024  # 1MB in float32
        large_send = torch.ones(size, device=device, dtype=torch.float32) * rank
        large_recv = torch.zeros(size, device=device, dtype=torch.float32)

        req_send = dist.isend(large_send, dst=send_rank)
        req_recv = dist.irecv(large_recv, src=recv_rank)
        req_send.wait()
        req_recv.wait()

        print(f"[GPU {rank}] ✓ isend/irecv successful")

        # Test 3: Multiple exchanges (simulating ring pattern)
        print(f"\n[GPU {rank}] Test 3: Multiple ring exchanges")
        chunk = torch.randn(1024, 1024, device=device, dtype=torch.float16)  # 2MB

        for step in range(min(3, world_size)):
            print(f"[GPU {rank}] Ring step {step}")

            if step < world_size - 1:
                chunk_recv = torch.empty_like(chunk)

                req_send = dist.isend(chunk, dst=send_rank)
                req_recv = dist.irecv(chunk_recv, src=recv_rank)
                req_send.wait()
                req_recv.wait()

                chunk = chunk_recv
                print(f"[GPU {rank}] ✓ Exchange {step} completed")

        print(f"\n[GPU {rank}] ✅ ALL COMMUNICATION TESTS PASSED")

        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        print(f"\n[GPU {rank}] ❌ COMMUNICATION ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_attention_only():
    """Test only the attention mechanism without distribution."""

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("\nTesting ATTENTION MECHANISM ONLY (no distribution)")

    try:
        # Test 1: Small attention
        print("\nTest 1: Small attention (seq_len=512)")
        batch, seq_len, heads, dim = 1, 512, 8, 64

        q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Simple attention computation
        q_t = q.transpose(1, 2)  # [b, h, n, d]
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v_t).transpose(1, 2)

        print(f"✓ Small attention successful. Output shape: {output.shape}")

        # Test 2: Medium attention
        print("\nTest 2: Medium attention (seq_len=2048)")
        seq_len = 2048
        q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Chunked attention (simulating ring pattern)
        chunk_size = seq_len // 2
        output = torch.zeros_like(q)

        for i in range(2):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            q_chunk = q[:, start:end]
            # Use full K,V for now to test attention math

            q_t = q_chunk.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (dim**0.5)
            attn = torch.softmax(scores, dim=-1)
            chunk_out = torch.matmul(attn, v_t).transpose(1, 2)

            output[:, start:end] = chunk_out

        print(f"✓ Chunked attention successful. Output shape: {output.shape}")

        # Test 3: Ring-style computation
        print("\nTest 3: Ring-style attention computation")
        # Each chunk of K,V
        k_chunk = k[:, :chunk_size]
        v_chunk = v[:, :chunk_size]

        # Process each Q chunk against current K,V chunk
        for i in range(2):
            q_start = i * chunk_size
            q_end = (i + 1) * chunk_size
            q_slice = q[:, q_start:q_end]

            # Reshape for computation
            b, n_q, h, d = q_slice.shape
            _, n_kv, _, _ = k_chunk.shape

            # This is where the error might be happening
            q_flat = q_slice.reshape(b * h, n_q, d)
            k_flat = k_chunk.reshape(b * h, n_kv, d)
            v_flat = v_chunk.reshape(b * h, n_kv, d)

            scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / (d**0.5)
            attn = torch.softmax(scores, dim=-1)
            out_flat = torch.matmul(attn, v_flat)

            # Reshape back
            _ = out_flat.reshape(b, n_q, h, d)
            print(f"  ✓ Processed Q chunk {i} against K,V chunk")

        print("\n✅ ALL ATTENTION TESTS PASSED")

    except Exception as e:
        print(f"\n❌ ATTENTION ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()

        # Try to identify the exact operation
        print("\nDebugging info:")
        print(f"Device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")


def test_combined(rank: int, world_size: int):
    """Test attention + communication together."""

    # Setup distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12361"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

        print(f"\n[GPU {rank}] Testing COMBINED (attention + communication)")

        # Small test case
        batch, seq_len, heads, dim = 1, 1024, 4, 64
        chunk_size = seq_len // world_size

        # Create full tensors
        q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Get local chunk
        local_start = rank * chunk_size
        local_end = (rank + 1) * chunk_size
        k_chunk = k[:, local_start:local_end].contiguous()
        v_chunk = v[:, local_start:local_end].contiguous()

        print(f"[GPU {rank}] Processing with chunk size {chunk_size}")

        # Test one iteration
        chunk_idx = rank
        q_start = chunk_idx * chunk_size
        q_end = (chunk_idx + 1) * chunk_size
        q_slice = q[:, q_start:q_end]

        # Attention computation
        b, n_q, h, d = q_slice.shape
        _, n_kv, _, _ = k_chunk.shape

        print(f"[GPU {rank}] Shapes - Q: {q_slice.shape}, K: {k_chunk.shape}")

        # Compute attention
        q_flat = q_slice.reshape(b * h, n_q, d)
        k_flat = k_chunk.reshape(b * h, n_kv, d)
        v_flat = v_chunk.reshape(b * h, n_kv, d)

        scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / (d**0.5)
        attn = torch.softmax(scores, dim=-1)
        out_flat = torch.matmul(attn, v_flat)
        _ = out_flat.reshape(b, n_q, h, d)

        print(f"[GPU {rank}] ✓ Attention computation successful")

        # Test communication
        if world_size > 1:
            send_rank = (rank + 1) % world_size
            recv_rank = (rank - 1 + world_size) % world_size

            k_recv = torch.empty_like(k_chunk)
            _ = torch.empty_like(v_chunk)

            req1 = dist.isend(k_chunk, dst=send_rank)
            req2 = dist.irecv(k_recv, src=recv_rank)
            req1.wait()
            req2.wait()

            print(f"[GPU {rank}] ✓ Communication successful")

        print(f"\n[GPU {rank}] ✅ COMBINED TEST PASSED")

        dist.barrier()
        dist.destroy_process_group()

    except Exception as e:
        print(f"\n[GPU {rank}] ❌ COMBINED ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Run isolation tests to identify the problem."""
    print("Ring Attention Debug - Isolating the Problem")
    print("=" * 70)

    # First test attention only (no distribution)
    print("\n=== PHASE 1: Testing Attention Mechanism ===")
    test_attention_only()

    # Clear GPU memory
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    # Test communication only if we have multiple GPUs
    world_size = min(2, torch.cuda.device_count())
    if world_size >= 2:
        print("\n\n=== PHASE 2: Testing Communication ===")
        try:
            mp.spawn(
                test_communication_only,
                args=(world_size,),
                nprocs=world_size,
                join=True,
            )
        except Exception as e:
            print(f"Communication test failed: {e}")

        # Clear GPU memory again
        torch.cuda.empty_cache()
        gc.collect()

        print("\n\n=== PHASE 3: Testing Combined ===")
        try:
            mp.spawn(test_combined, args=(world_size,), nprocs=world_size, join=True)
        except Exception as e:
            print(f"Combined test failed: {e}")

    print("\n" + "=" * 70)
    print("Debug complete. Check results above to identify the issue.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
