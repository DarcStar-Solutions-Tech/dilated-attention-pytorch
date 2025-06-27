# Hierarchical Attention Patterns Benchmark

Generated: 2025-06-27T23:21:57.477419Z

## Configuration

- Device: cuda
- Data type: torch.float32
- Sequence length: 4096
- Batch size: 1
- Num heads: 8
- Head dim: 64

## Results

| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |
|----------------|-----------|-------------|--------------------|
| Dense Baseline | 3.17 | 58.52 | 1.00x |
| Original Block-Sparse | 59.47 | 48.75 | 0.05x |
| Optimized Block-Sparse | 20.07 | 288.13 | 0.16x |
| Hierarchical (default) | 629.11 | 3120.19 | 0.01x |
| Hierarchical-standard | 449.37 | 3120.19 | 0.01x |
| Hierarchical-fine_grained | 629.59 | 3120.19 | 0.01x |
| Hierarchical-long_range | 450.94 | 3120.19 | 0.01x |
| Hierarchical-ultra_sparse | 210.74 | 1392.15 | 0.02x |

## Key Findings

- Fastest: Dense Baseline (3.17 ms)
- Most memory efficient: Original Block-Sparse (48.75 MB)
- Hierarchical is 3034.0% slower than local window
