# Comprehensive Dilated Attention Benchmark

Generated: 2025-06-28T00:02:38.192379Z

## Configuration
- Device: cuda
- Data type: torch.float16

## Performance Results

### Small Configuration
- Sequence length: 2048
- Batch size: 2
- Num heads: 8
- Head dim: 64

| Implementation | Time (ms) | Memory (MB) |
|----------------|-----------|-------------|
| BlockSparseRing | 35.10 ± 2.15 | 28.88 |
| BlockSparseOptimized | 12.56 ± 1.75 | 148.13 |
| BlockSparseHierarchical | 21.52 ± 1.95 | 232.13 |
| RingDilatedAttentionV2 | 3.53 ± 0.32 | 38.88 |

### Medium Configuration
- Sequence length: 8192
- Batch size: 1
- Num heads: 8
- Head dim: 64

| Implementation | Time (ms) | Memory (MB) |
|----------------|-----------|-------------|
| DilatedAttention | 57.05 ± 3.09 | 56.62 |
| ImprovedDilatedAttention | 64.02 ± 7.66 | 67.53 |
| BlockSparseRing | 120.55 ± 3.16 | 48.45 |
| BlockSparseOptimized | 39.70 ± 1.56 | 288.13 |
| BlockSparseHierarchical | 197.81 ± 176.04 | 759.90 |
| RingDilatedAttentionV2 | 36.63 ± 4.66 | 103.16 |

### Large Configuration
- Sequence length: 16384
- Batch size: 1
- Num heads: 12
- Head dim: 64

| Implementation | Time (ms) | Memory (MB) |
|----------------|-----------|-------------|
| DilatedAttention | 131.18 ± 2.72 | 154.12 |
| ImprovedDilatedAttention | 135.19 ± 3.40 | 172.16 |
| BlockSparseRing | 233.75 ± 3.50 | 128.61 |
| BlockSparseOptimized | 213.24 ± 209.15 | 848.14 |
| RingDilatedAttentionV2 | 465.16 ± 296.81 | 404.19 |

## Summary

- **Fastest**: RingDilatedAttentionV2 (3.53 ms)
- **Most Memory Efficient**: BlockSparseRing (28.88 MB)
