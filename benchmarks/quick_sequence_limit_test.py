#!/usr/bin/env python3
"""Quick test of sequence limits."""

import torch
import gc
import sys

sys.path.append("..")

from dilated_attention_pytorch.kernels import (
    UnifiedHilbertAttention,
    UnifiedHilbertAttentionOptimizedEnhanced,
)

# Disable CUDA synchronization for speed
torch.cuda.set_sync_debug_mode(0)


def quick_test(impl_name, impl_class, seq_len, dilation_rate=1):
    """Quick test without timing."""
    try:
        gc.collect()
        torch.cuda.empty_cache()

        model = (
            impl_class(
                hidden_dim=512,
                num_heads=8,
                segment_size=128,
                dilation_rate=dilation_rate,
            )
            .cuda()
            .eval()
        )

        x = torch.randn(1, seq_len, 512, device="cuda", dtype=torch.float16)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                out = model(x)

        mem_gb = torch.cuda.max_memory_allocated() / 1024**3

        del out, x, model
        gc.collect()
        torch.cuda.empty_cache()

        return True, mem_gb

    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return False, None
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {str(e)}")
        gc.collect()
        torch.cuda.empty_cache()
        return False, None


print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Test sequence lengths
test_lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

print("=== Dense Patterns (d=1) ===")
print(f"{'Seq Length':<12} | {'Unified':<20} | {'Enhanced':<20}")
print("-" * 55)

for seq_len in test_lengths:
    # Test Unified
    success_u, mem_u = quick_test("Unified", UnifiedHilbertAttention, seq_len, 1)

    # Test Enhanced
    success_e, mem_e = quick_test(
        "Enhanced", UnifiedHilbertAttentionOptimizedEnhanced, seq_len, 1
    )

    u_str = f"✓ {mem_u:.2f} GB" if success_u else "✗ Failed"
    e_str = f"✓ {mem_e:.2f} GB" if success_e else "✗ Failed"

    print(f"{seq_len:<12,} | {u_str:<20} | {e_str:<20}")

    # Stop testing larger sizes after both fail
    if not success_u and not success_e:
        break

print("\n=== Sparse Patterns (d=2) ===")
print(f"{'Seq Length':<12} | {'Unified':<20} | {'Enhanced':<20}")
print("-" * 55)

for seq_len in test_lengths:
    # Test Unified
    success_u, mem_u = quick_test("Unified", UnifiedHilbertAttention, seq_len, 2)

    # Test Enhanced
    success_e, mem_e = quick_test(
        "Enhanced", UnifiedHilbertAttentionOptimizedEnhanced, seq_len, 2
    )

    u_str = f"✓ {mem_u:.2f} GB" if success_u else "✗ Failed"
    e_str = f"✓ {mem_e:.2f} GB" if success_e else "✗ Failed"

    print(f"{seq_len:<12,} | {u_str:<20} | {e_str:<20}")

    # Stop testing larger sizes after both fail
    if not success_u and not success_e:
        break

print("\n=== Sparse Patterns (d=4) ===")
print(f"{'Seq Length':<12} | {'Unified':<20} | {'Enhanced':<20}")
print("-" * 55)

for seq_len in [1024, 2048, 4096, 8192, 16384, 32768]:
    # Test Unified
    success_u, mem_u = quick_test("Unified", UnifiedHilbertAttention, seq_len, 4)

    # Test Enhanced
    success_e, mem_e = quick_test(
        "Enhanced", UnifiedHilbertAttentionOptimizedEnhanced, seq_len, 4
    )

    u_str = f"✓ {mem_u:.2f} GB" if success_u else "✗ Failed"
    e_str = f"✓ {mem_e:.2f} GB" if success_e else "✗ Failed"

    print(f"{seq_len:<12,} | {u_str:<20} | {e_str:<20}")

    # Stop testing larger sizes after both fail
    if not success_u and not success_e:
        break
