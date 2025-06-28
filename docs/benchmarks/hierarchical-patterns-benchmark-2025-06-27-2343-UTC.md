# Hierarchical Attention Patterns Benchmark

Generated: 2025-06-27T23:43:01.153072Z

## Configuration

- Device: cuda
- Data type: torch.float16
- Sequence length: 2048
- Batch size: 1
- Num heads: 8
- Head dim: 64

## Results

| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |
|----------------|-----------|-------------|--------------------|
| Dense Baseline | 16.40 | 12.13 | 1.00x |
| Original Block-Sparse | 28.01 | 16.44 | 0.59x |
| Optimized Block-Sparse | 9.73 | 76.13 | 1.69x |
| Hierarchical (default) | 17.08 | 124.13 | 0.96x |
| Hierarchical-standard | 16.53 | 124.13 | 0.99x |
| Hierarchical-fine_grained | 66.78 | 400.14 | 0.25x |
| Hierarchical-long_range | 64.94 | 400.14 | 0.25x |
| Hierarchical-ultra_sparse | 42.39 | 244.88 | 0.39x |

## Key Findings

- Fastest: Optimized Block-Sparse (9.73 ms)
- Most memory efficient: Dense Baseline (12.13 MB)
- Hierarchical is 75.7% slower than local window
