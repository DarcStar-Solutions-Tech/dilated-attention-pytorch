"""
Simple memory monitoring test to show Ring V2 behavior.
"""

import torch
import nvidia_ml_py3 as nvml
import time

from dilated_attention_pytorch import ImprovedDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2


def get_gpu_memory_mb(gpu_id=0):
    """Get current GPU memory usage in MB."""
    nvml.nvmlInit()
    handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024


def test_memory_usage():
    print("Ring Attention V2 - Memory Usage Comparison")
    print("=" * 60)

    # Test configuration
    seq_len = 16384
    batch_size = 1
    num_heads = 8
    head_dim = 64

    print(
        f"Configuration: seq_len={seq_len}, batch={batch_size}, "
        f"heads={num_heads}, dim={head_dim}"
    )
    print("=" * 60)

    # Test 1: Improved (baseline)
    print("\n1. Testing Improved Dilated Attention (baseline)...")

    torch.cuda.empty_cache()
    time.sleep(1)

    mem_before = get_gpu_memory_mb(0)
    print(f"   Memory before: {mem_before:.0f}MB")

    model = ImprovedDilatedAttention(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
    ).cuda()

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    mem_after_create = get_gpu_memory_mb(0)
    print(
        f"   Memory after creating model+data: {mem_after_create:.0f}MB "
        f"(+{mem_after_create - mem_before:.0f}MB)"
    )

    # Forward pass
    output = model(q, k, v)
    torch.cuda.synchronize()

    mem_after_forward = get_gpu_memory_mb(0)
    print(
        f"   Memory after forward pass: {mem_after_forward:.0f}MB "
        f"(+{mem_after_forward - mem_after_create:.0f}MB)"
    )

    improved_total = mem_after_forward - mem_before

    # Cleanup
    del model, q, k, v, output
    torch.cuda.empty_cache()

    # Test 2: Ring V2 with ring_size=1 (no distribution)
    print("\n2. Testing Ring V2 (ring_size=1, no distribution)...")

    time.sleep(1)
    mem_before = get_gpu_memory_mb(0)
    print(f"   Memory before: {mem_before:.0f}MB")

    model = RingDilatedAttentionV2(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        ring_size=1,
        device="cuda",
        dtype=torch.float16,
    )
    print(f"   Mode: {model.mode}")

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    mem_after_create = get_gpu_memory_mb(0)
    print(
        f"   Memory after creating model+data: {mem_after_create:.0f}MB "
        f"(+{mem_after_create - mem_before:.0f}MB)"
    )

    output = model(q, k, v)
    torch.cuda.synchronize()

    mem_after_forward = get_gpu_memory_mb(0)
    print(
        f"   Memory after forward pass: {mem_after_forward:.0f}MB "
        f"(+{mem_after_forward - mem_after_create:.0f}MB)"
    )

    ring1_total = mem_after_forward - mem_before

    # Cleanup
    del model, q, k, v, output
    torch.cuda.empty_cache()

    # Test 3: Ring V2 with ring_size=2 (simulated distribution)
    print("\n3. Testing Ring V2 (ring_size=2, simulated distribution)...")

    time.sleep(1)
    mem_before = get_gpu_memory_mb(0)
    print(f"   Memory before: {mem_before:.0f}MB")

    model = RingDilatedAttentionV2(
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4],
        ring_size=2,
        device="cuda",
        dtype=torch.float16,
    )
    print(f"   Mode: {model.mode}")

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.float16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    mem_after_create = get_gpu_memory_mb(0)
    print(
        f"   Memory after creating model+data: {mem_after_create:.0f}MB "
        f"(+{mem_after_create - mem_before:.0f}MB)"
    )

    _ = model(q, k, v)
    torch.cuda.synchronize()

    mem_after_forward = get_gpu_memory_mb(0)
    print(
        f"   Memory after forward pass: {mem_after_forward:.0f}MB "
        f"(+{mem_after_forward - mem_after_create:.0f}MB)"
    )

    ring2_total = mem_after_forward - mem_before

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Improved (baseline):     {improved_total:.0f}MB")
    print(
        f"Ring V2 (ring_size=1):   {ring1_total:.0f}MB "
        f"({ring1_total / improved_total:.1f}x baseline)"
    )
    print(
        f"Ring V2 (ring_size=2):   {ring2_total:.0f}MB "
        f"({ring2_total / improved_total:.1f}x baseline)"
    )

    print("\nKey insights:")
    print("- Ring V2 with ring_size=1 uses more memory due to overhead")
    print("- Ring V2 with ring_size=2 (simulated) shows some reduction")
    print("- True benefits require actual distributed execution")


if __name__ == "__main__":
    test_memory_usage()
