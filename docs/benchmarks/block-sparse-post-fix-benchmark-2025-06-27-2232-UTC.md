# Block-Sparse Ring Attention Benchmark (Post Ring Attention Fix)

Generated: 2025-06-27T22:32:11.279888Z

## Configuration

- Device: cuda
- Sequence length: 4096
- Batch size: 1
- Num heads: 8
- Head dim: 64
- Segment lengths: [1024, 2048, 4096]
- Dilation rates: [1, 2, 4]

## Results

| Implementation | Mean Time (ms) | Std Dev | Memory (MB) | Speedup vs Baseline |
|---------------|----------------|---------|-------------|--------------------|
| ImprovedDilatedAttention | 30.03 | ±1.70 | 28.27 | 1.00x |
| RingDilatedAttentionV2 | 15.15 | ±0.12 | 226.17 | 1.98x |
| BlockSparse_LocalWindow_0.9 | 153.24 | ±100.90 | 28.44 | 0.20x |
| BlockSparse_Dilated_0.9 | 84.52 | ±2.37 | 28.44 | 0.36x |
| BlockSparse_LocalWindow_0.95 | 60.10 | ±1.56 | 28.44 | 0.50x |
| BlockSparse_Dilated_0.95 | 97.39 | ±26.42 | 28.44 | 0.31x |
| BlockSparse_LocalWindow_0.98 | 59.77 | ±2.24 | 28.44 | 0.50x |
| BlockSparse_Dilated_0.98 | 85.28 | ±3.67 | 28.44 | 0.35x |

## Analysis

- Best Block-Sparse: BlockSparse_LocalWindow_0.98 (59.77 ms)
- Worst Block-Sparse: BlockSparse_LocalWindow_0.9 (153.24 ms)
- Best speedup vs baseline: 0.50x

## Notes

This benchmark was run after fixing Ring Attention normalization issues.
The goal is to verify if Block-Sparse implementations benefit from the fixes.
