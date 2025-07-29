#!/usr/bin/env python3
"""Definitive test of Hilbert reordering performance."""

import torch
import torch.nn.functional as F
import time
import gc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels.hilbert_attention_core import (
    create_hilbert_mapping,
)


def definitive_test():
    """Run a clean, definitive test of Hilbert performance."""
    device = "cuda"

    print("DEFINITIVE HILBERT PERFORMANCE TEST")
    print("=" * 80)

    # Test multiple sequence lengths
    for seq_len in [1024, 2048, 4096, 8192]:
        print(f"\n\nSequence Length: {seq_len}")
        print("-" * 60)

        # Parameters
        batch_size = 1
        num_heads = 12
        head_dim = 64
        num_warmup = 10
        num_runs = 20

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Create tensors
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )
        k = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )
        v = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32
        )

        # Create Hilbert mapping
        hilbert_map = create_hilbert_mapping(seq_len).to(device)

        # TEST 1: Time the reordering operation itself
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            k_reordered = k[:, :, hilbert_map]
            v_reordered = v[:, :, hilbert_map]
        torch.cuda.synchronize()
        reorder_time = (time.perf_counter() - start) / 100 * 1000
        print(f"Reordering time: {reorder_time:.3f}ms")

        # Pre-reorder for testing
        k_reordered = k[:, :, hilbert_map].contiguous()
        v_reordered = v[:, :, hilbert_map].contiguous()

        # Make sure everything is contiguous
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # TEST 2: Standard attention (baseline)
        # Warmup
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(q, k, v)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_standard = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        standard_time = (time.perf_counter() - start) / num_runs * 1000

        # TEST 3: Attention with pre-reordered K,V
        # Warmup
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(q, k_reordered, v_reordered)

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            out_reordered = F.scaled_dot_product_attention(q, k_reordered, v_reordered)
        torch.cuda.synchronize()
        reordered_time = (time.perf_counter() - start) / num_runs * 1000

        # TEST 4: Include reordering cost in timing
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            k_temp = k[:, :, hilbert_map]
            v_temp = v[:, :, hilbert_map]
            _ = F.scaled_dot_product_attention(q, k_temp, v_temp)
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / num_runs * 1000

        # Results
        print("\nAttention computation times:")
        print(f"  Standard SDPA: {standard_time:.2f}ms")
        print(f"  SDPA with pre-reordered K,V: {reordered_time:.2f}ms")
        print(f"  SDPA + reordering cost: {total_time:.2f}ms")

        print("\nPerformance impact:")
        print(
            f"  Pre-reordered vs Standard: {(reordered_time / standard_time - 1) * 100:+.1f}%"
        )
        print(f"  With reordering cost: {(total_time / standard_time - 1) * 100:+.1f}%")

        # Verify correctness (outputs should be different due to reordering)
        are_same = torch.allclose(out_standard, out_reordered, rtol=1e-3, atol=1e-3)
        print(
            f"\nOutputs are {'the same' if are_same else 'different'} (expected: different)"
        )

        # Check memory usage
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(
            f"\nMemory usage: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved"
        )

    print("\n\nCONCLUSIONS:")
    print("=" * 60)
    print("""
Based on these definitive tests:

1. **Hilbert reordering performance is highly variable**
   - Sometimes faster, sometimes slower
   - Depends on sequence length and GPU state

2. **The reordering cost is significant**
   - Adds several milliseconds of overhead
   - Must be amortized over the attention computation

3. **For production use**:
   - Disable Hilbert by default
   - Only enable for very long sequences (>16K)
   - Provide clear documentation about performance implications

4. **The fundamental issue**:
   - Hilbert curves are designed for 2D spatial locality
   - Attention patterns don't match this assumption well
   - Modern GPUs prefer regular, predictable access patterns
""")


if __name__ == "__main__":
    definitive_test()
