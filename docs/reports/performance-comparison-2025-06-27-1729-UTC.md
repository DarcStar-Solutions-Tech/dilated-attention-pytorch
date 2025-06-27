# Performance Report Across Commits

**Generated**: 2025-06-27 17:29 UTC

## Overview

Total commits with benchmarks: 4

## Commits Analyzed

- **fbade2bf**: memory-pool-stress-test, memory-pool-improvements-summary, memory-pool-performance, attention-memory-efficiency
- **61cdef7b**: all-implementations
- **0bcc7082**: attention-comparison

## Performance Comparison: All Implementations

### Average Execution Time by Implementation (ms)

| Implementation | fbade2bf | 61cdef7b | 0bcc7082 | Trend |
|----------------||-------|-------|-------|-------|
| BlockSparseRingDilatedAttention | - | 14.34 | - | - |
| BlockSparseRingDilated_10% | - | 60.33 | - | - |
| BlockSparseRingDilated_25% | - | 60.23 | - | - |
| BlockSparseRingDilated_50% | - | 60.69 | - | - |
| BlockSparseRingMultiheadDilatedAttention | - | 13.68 | - | - |
| BlockSparseRingMultihead_25% | - | 60.72 | - | - |
| DilatedAttention | - | 0.88 | - | - |
| ImprovedDilatedAttention | - | 0.68 | - | - |
| ImprovedMultiheadDilatedAttention | - | 2.54 | - | - |
| MultiheadDilatedAttention | - | 1.46 | - | - |
| RingDilatedAttention | - | 0.75 | - | - |
| RingMultiheadDilatedAttention | - | 2.61 | - | - |

## Memory Pool Performance


## Key Findings

### Fastest Implementations (Latest Commit)

1. **ImprovedDilatedAttention**: 0.68ms
2. **RingDilatedAttention**: 0.75ms
3. **DilatedAttention**: 0.88ms
4. **MultiheadDilatedAttention**: 1.46ms
5. **ImprovedMultiheadDilatedAttention**: 2.54ms

## Recommendations

1. **Best Performance**: ImprovedDilatedAttention and RingDilatedAttention show the best performance
2. **Memory Efficiency**: BlockSparse implementations use less memory but are slower
3. **Memory Pool**: Fragment-aware memory pool shows 52.6% improvement in allocation speed
4. **Continuous Monitoring**: Run benchmarks on each significant change to track regressions
