# Block-Sparse Optimization Benchmark

Generated: 2025-06-27T22:59:14.137104Z

## Configuration

- Device: cuda
- Sequence length: 4096
- Batch size: 1
- Num heads: 8
- Head dim: 64
- Sparsity: 0.9
- Segment lengths: [1024, 2048, 4096]
- Dilation rates: [1, 2, 4]

## Results

| Implementation | Mean Time (ms) | Std Dev | Memory (MB) | Speedup vs Original |
|---------------|----------------|---------|-------------|--------------------|
| ImprovedDilatedAttention | 27.73 | ±1.11 | 28.27 | 2.18x |
| BlockSparse_Original | 60.32 | ±2.19 | 28.44 | 1.00x |
| BlockSparse_Optimized | 19.95 | ±1.57 | 148.13 | 3.02x |

## Analysis

- Optimization improvement: 66.9%
- Optimized vs baseline: 1.39x
- Original vs baseline: 0.46x

## Cache Statistics

- Final cache hit rate: 96.97%
- Total accesses: 32
- Cached patterns: 1
