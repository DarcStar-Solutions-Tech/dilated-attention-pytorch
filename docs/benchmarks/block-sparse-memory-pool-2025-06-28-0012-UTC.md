# Block-Sparse Ring Dilated Attention Memory Pool Benchmark

Generated: 2025-06-28T00:12:18.310852Z

## Configuration

- Device: cuda
- Batch Size: 1
- Sequence Length: 16384
- Num Heads: 8
- Head Dim: 64
- Iterations: 10
- Segment Lengths: [2048, 4096, 8192]
- Dilation Rates: [1, 2, 4]
- Sparse Pattern: local_window
- Sparsity Ratio: 0.05 (95% sparse)
- Block Size: 128
- PyTorch Version: 2.7.1+cu126

## Results

| Configuration | Time/Iter | Peak Memory | With Weights Time |
|---------------|-----------|-------------|-------------------|
| Without Pool | 0.2179s | 170.4MB | 0.0000s |
| With Pool | 0.1598s | 938.4MB | 0.0000s |
| **Improvement** | **+26.7%** | **-450.7%** | - |

## Memory Pool Integration Details

### Optimizations Applied:
1. **Inherited Memory Pool**: Leverages parent RingDilatedAttentionV2 memory pool
2. **1MB Threshold**: Only uses pool for tensors ≥ 1MB
3. **Causal Mask Caching**: Reuses causal masks across blocks
4. **Lightweight Pool Mode**: Uses bucketed allocation without fragmentation tracking
5. **Sparse-Aware Allocation**: Only allocates memory for active blocks

### Performance Analysis:
- ✅ **Time**: 26.7% faster with memory pool
- ⚠️ **Memory**: 450.7% increase with memory pool

### Block-Sparse Specific Benefits:
- Memory pools are particularly beneficial for sparse patterns with many active blocks
- Causal mask caching reduces redundant allocations in diagonal blocks
- Sparse computation already minimizes memory usage, so pool benefits may be limited
- Best suited for long sequences where communication buffers dominate memory usage
