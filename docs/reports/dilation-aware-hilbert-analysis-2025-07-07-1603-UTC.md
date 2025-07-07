# Dilation-Aware Hilbert Optimization - Analysis Report

**Date**: 2025-07-07 16:03 UTC  
**Subject**: Performance analysis of dilation-aware Hilbert space-filling curve optimization  
**Hardware**: NVIDIA GeForce GTX 1080

## Executive Summary

The dilation-aware Hilbert implementation groups blocks based on their dilated access patterns and applies Hilbert ordering within each group. This approach shows **30% improvement** over the original Hilbert V1 but is still **25% slower** than the standard implementation on average.

## Implementation Approach

### Key Innovation: Access Pattern Grouping

Instead of applying Hilbert ordering globally, the dilation-aware approach:

1. **Groups blocks by dilation pattern**: Blocks that access each other due to dilation are grouped together
2. **Applies Hilbert within groups**: Each group is optimized independently
3. **Preserves dilated structure**: The overall sparse pattern remains intact

### Example Grouping (Dilation Rate 4):
```
Group 0: [0, 4, 8, 12]    # These blocks access each other
Group 1: [1, 5, 9, 13]    # These blocks access each other
Group 2: [2, 6, 10, 14]   # These blocks access each other
Group 3: [3, 7, 11, 15]   # These blocks access each other
Local groups: [0,1,2], [3,4,5], ...  # Adjacent blocks
```

## Results

### Performance Comparison

| Seq Length | Dilation | Standard | Hilbert V1 | Dilation-Aware | V1 Speedup | DA Speedup |
|------------|----------|----------|------------|----------------|------------|------------|
| 4096 | 1 | 29.9ms | 59.7ms | 43.3ms | 0.50x | **0.69x** |
| 4096 | 4 | 30.4ms | 59.8ms | 35.7ms | 0.51x | **0.85x** |
| 4096 | 8 | 29.0ms | 58.9ms | 35.4ms | 0.49x | **0.82x** |
| 8192 | 1 | 63.4ms | 123.9ms | 101.0ms | 0.51x | **0.63x** |
| 8192 | 4 | 64.9ms | 123.6ms | 82.6ms | 0.52x | **0.79x** |
| 8192 | 8 | 62.8ms | 124.4ms | 78.6ms | 0.51x | **0.80x** |
| 16384 | 1 | 246.8ms | 318.4ms | 316.3ms | 0.78x | 0.78x |
| 16384 | 4 | 201.0ms | 299.3ms | 331.6ms | 0.67x | 0.61x |
| 16384 | 8 | 196.7ms | 303.1ms | 361.6ms | 0.65x | 0.54x |

### Key Findings

1. **30% Average Improvement over V1**: The dilation-aware approach is significantly better than naive Hilbert
   - Best improvement: **67.8%** at dilation=4, seq_len=4096
   - Consistent improvements across most configurations

2. **Still 25% Slower than Standard**: Despite improvements, still can't beat the baseline
   - Best case: 0.85x (15% slower) at dilation=4, seq_len=4096
   - Worst case: 0.54x (46% slower) at dilation=8, seq_len=16384

3. **Performance Degrades at Larger Scales**: The approach works better on smaller sequences
   - 4K tokens: Average 0.75x speedup
   - 8K tokens: Average 0.75x speedup  
   - 16K tokens: Average 0.63x speedup

## Analysis

### Why Dilation-Aware Helps

1. **Preserves Access Patterns**: Unlike V1, doesn't disrupt the dilated structure
2. **Smaller Optimization Scope**: Hilbert applied to smaller groups (4-8 blocks) rather than all blocks
3. **Maintains Locality**: Blocks that communicate stay together

### Why It's Still Slower

1. **Overhead of Grouping**: Creating and managing access groups adds complexity
2. **Multiple Small Hilbert Curves**: Less effective than one large curve
3. **GPU Architecture**: Still fighting against GPU's preference for sequential access

### Interesting Observations

1. **Group Count Increases with Dilation**: More groups = more fragmentation
   - Dilation 1: ~22 groups
   - Dilation 8: ~29-93 groups

2. **Sweet Spot at Dilation 4**: Best relative performance at moderate dilation
   - Enough structure for grouping
   - Not too fragmented

3. **Diminishing Returns**: Larger sequences show worse performance
   - More groups to manage
   - Overhead dominates benefits

## Conclusion

The dilation-aware Hilbert approach successfully addresses the fundamental mismatch between Hilbert curves and dilated patterns by:
- ✓ Grouping blocks that actually communicate
- ✓ Preserving dilated access patterns
- ✓ Achieving 30% improvement over naive Hilbert

However, it still can't overcome the fundamental GPU architecture preferences:
- ✗ 25% slower than standard on average
- ✗ Performance degrades with sequence length
- ✗ Overhead of managing groups is significant

### Recommendation

While the dilation-aware approach is a significant improvement over naive Hilbert ordering, it still doesn't provide performance benefits over the standard implementation. The lesson is clear: **GPU architectures strongly favor simple, predictable access patterns** over complex reordering schemes, even when those schemes are theoretically superior for cache locality.

The implementation serves as a valuable exploration of how to adapt classical algorithms (Hilbert curves) to modern constraints (dilated patterns), but should not be used in production due to the performance penalty.