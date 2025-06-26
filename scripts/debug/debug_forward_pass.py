"""
Debug forward pass of block sparse implementation
"""

import sys
import time

import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention, SparsePatternConfig)


def debug_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    batch_size = 1
    seq_len = 2048
    num_heads = 8
    head_dim = 64

    # Create module
    sparse_config = SparsePatternConfig(
        pattern_type="dilated_sparse",
        sparsity_ratio=0.1,  # 90% sparse
        block_size=32,
    )

    print("Creating module...")
    module = BlockSparseRingDilatedAttention(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        dropout=0.0,
        ring_size=1,
    )

    print("Moving to device...")
    module = module.to(device, dtype)

    # Create inputs
    print("Creating inputs...")
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    print("Starting forward pass...")
    sys.stdout.flush()

    # Add instrumentation to forward method
    original_forward = module.forward

    def instrumented_forward(q, k, v, **kwargs):
        print("  - Enter forward")
        sys.stdout.flush()

        # Call _get_sparse_pattern
        print("  - Getting sparse pattern...")
        sys.stdout.flush()
        start = time.time()
        pattern = module._get_sparse_pattern(q, k)
        print(f"    Pattern generation took: {(time.time() - start) * 1000:.2f}ms")
        sys.stdout.flush()

        # Call _block_sparse_ring_attention
        print("  - Calling block sparse ring attention...")
        sys.stdout.flush()
        start = time.time()
        output, _ = module._block_sparse_ring_attention(q, k, v, pattern, False, False)
        print(f"    Block sparse attention took: {(time.time() - start) * 1000:.2f}ms")
        sys.stdout.flush()

        return output

    # Monkey-patch for debugging
    module.forward = instrumented_forward

    try:
        start = time.time()
        with torch.no_grad():
            output = module(query, key, value)
        torch.cuda.synchronize()
        total_time = (time.time() - start) * 1000
        print(f"Total forward pass took: {total_time:.2f}ms")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_forward()
