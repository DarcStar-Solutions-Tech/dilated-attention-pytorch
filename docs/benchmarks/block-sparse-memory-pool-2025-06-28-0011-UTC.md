# Block-Sparse Ring Dilated Attention Memory Pool Benchmark

Generated: 2025-06-28T00:11:41.444898Z

## Configuration

- Device: cuda
- Batch Size: 2
- Sequence Length: 4096
- Num Heads: 8
- Head Dim: 64
- Iterations: 20
- Segment Lengths: [2048, 4096, 8192]
- Dilation Rates: [1, 2, 4]
- Sparse Pattern: dilated_sparse
- Sparsity Ratio: 0.1 (90% sparse)
- Block Size: 128
- PyTorch Version: 2.7.1+cu126

## Results

| Configuration | Time/Iter | Peak Memory | With Weights Time |
|---------------|-----------|-------------|-------------------|
| Without Pool | 0.0557s | 92.6MB | 0.2786s |
| With Pool | 0.0641s | 380.6MB | 0.1707s |
| **Improvement** | **-15.0%** | **-310.9%** | - |

## Memory Pool Integration Details

### Optimizations Applied:
1. **Inherited Memory Pool**: Leverages parent RingDilatedAttentionV2 memory pool
2. **1MB Threshold**: Only uses pool for tensors ≥ 1MB
3. **Causal Mask Caching**: Reuses causal masks across blocks
4. **Lightweight Pool Mode**: Uses bucketed allocation without fragmentation tracking
5. **Sparse-Aware Allocation**: Only allocates memory for active blocks

### Performance Analysis:
- ⚠️ **Time**: 15.0% slower with memory pool
- ⚠️ **Memory**: 310.9% increase with memory pool

### Block-Sparse Specific Benefits:
- Memory pools are particularly beneficial for sparse patterns with many active blocks
- Causal mask caching reduces redundant allocations in diagonal blocks
- Sparse computation already minimizes memory usage, so pool benefits may be limited
- Best suited for long sequences where communication buffers dominate memory usage
