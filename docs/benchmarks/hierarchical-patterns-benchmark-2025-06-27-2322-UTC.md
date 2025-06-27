# Hierarchical Attention Patterns Benchmark

Generated: 2025-06-27T23:22:55.415293Z

## Configuration

- Device: cuda
- Data type: torch.float16
- Sequence length: 4096
- Batch size: 1
- Num heads: 8
- Head dim: 64

## Results

| Implementation | Time (ms) | Memory (MB) | Speedup vs Baseline |
|----------------|-----------|-------------|--------------------|
| Dense Baseline | 29.53 | 28.27 | 1.00x |
| Original Block-Sparse | 58.65 | 28.44 | 0.50x |
| Optimized Block-Sparse | 22.57 | 148.13 | 1.31x |
| Hierarchical (default) | 638.05 | 1564.19 | 0.05x |
| Hierarchical-standard | 670.35 | 1564.19 | 0.04x |
| Hierarchical-fine_grained | 487.51 | 1564.19 | 0.06x |
| Hierarchical-long_range | 601.06 | 1564.19 | 0.05x |
| Hierarchical-ultra_sparse | 340.39 | 700.15 | 0.09x |

## Key Findings

- Fastest: Optimized Block-Sparse (22.57 ms)
- Most memory efficient: Dense Baseline (28.27 MB)
- Hierarchical is 2727.5% slower than local window
