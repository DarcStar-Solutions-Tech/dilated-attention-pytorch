# Sequence Length Range Benchmark Report

Generated: 2025-06-30 04:37:08 UTC

Device: cuda

## Production (1K-16K) Range

### Performance Comparison

| Implementation | Seq Length | Time (ms) | Memory (MB) | Throughput (tok/s) | Status |
|----------------|------------|-----------|-------------|-------------------|---------|
| dilated | 1,024 | 30.6 | 6.0 | 66,860 | ✓ |
| dilated | 2,048 | 47.2 | 19.0 | 86,834 | ✓ |
| dilated | 4,096 | 86.2 | 32.0 | 95,043 | ✓ |
| dilated | 8,192 | 176.8 | 60.0 | 92,688 | ✓ |
| dilated | 8,192 | 394.4 | 38.0 | 20,769 | ✓ |
| dilated | 16,384 | 850.9 | 76.0 | 19,255 | ✓ |
| improved | 1,024 | 8.3 | 26.0 | 245,601 | ✓ |
| improved | 2,048 | 55.9 | 47.0 | 73,308 | ✓ |
| improved | 4,096 | 74.0 | 178.0 | 110,766 | ✓ |
| improved | 8,192 | 141.3 | 170.0 | 115,939 | ✓ |
| improved | 8,192 | 60.0 | 166.0 | 136,513 | ✓ |
| improved | 16,384 | 133.7 | 140.0 | 122,560 | ✓ |

### Best Implementation by Sequence Length
- 1,024 tokens: improved (245,601 tok/s)
- 2,048 tokens: dilated (86,834 tok/s)
- 4,096 tokens: improved (110,766 tok/s)
- 8,192 tokens: improved (136,513 tok/s)
- 16,384 tokens: improved (122,560 tok/s)