"""
Quick profiling script to identify bottlenecks in block sparse implementation
"""

import time

import torch

from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
    BlockSparseRingDilatedAttention,
    SparsePatternConfig,
)


def profile_block_sparse():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

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

    module = BlockSparseRingDilatedAttention(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        dropout=0.0,
    ).to(device, dtype)

    # Create inputs
    query = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Profile different parts
    print("Profiling Block Sparse Ring Dilated Attention")
    print("=" * 60)

    # Test pattern generation
    print("\n1. Testing pattern generation...")
    start = time.time()
    with torch.no_grad():
        pattern = module._get_sparse_pattern(query, key)
    torch.cuda.synchronize()
    print(f"   Pattern generation took: {(time.time() - start) * 1000:.2f}ms")
    print(f"   Pattern shape: {pattern.shape}")
    print(
        f"   Pattern sparsity: {(pattern.sum().item() / pattern.numel() * 100):.1f}% non-zero"
    )

    # Test forward pass with torch profiler
    print("\n2. Profiling forward pass...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            start = time.time()
            _ = module(query, key, value)
            torch.cuda.synchronize()
            total_time = (time.time() - start) * 1000

    print(f"   Total forward pass: {total_time:.2f}ms")

    # Print top operations
    print("\n3. Top time-consuming operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Test individual components
    print("\n4. Testing individual components...")

    # Test memory pool
    start = time.time()
    _ = module.memory_pool.get_buffer(query.shape, query.dtype, query.device)
    torch.cuda.synchronize()
    print(f"   Memory pool buffer allocation: {(time.time() - start) * 1000:.2f}ms")

    # Test pattern generator create_pattern
    start = time.time()
    with torch.no_grad():
        _ = module.pattern_generator.create_pattern(seq_len, num_heads, device)
    torch.cuda.synchronize()
    print(f"   Pattern generator create_pattern: {(time.time() - start) * 1000:.2f}ms")


if __name__ == "__main__":
    profile_block_sparse()
