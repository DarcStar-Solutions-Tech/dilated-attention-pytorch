# Hilbert Block-Sparse Analysis Report

**Date**: 2025-07-07 14:55 UTC  
**Subject**: Performance analysis of Hilbert space-filling curve optimization for block-sparse attention  
**Hardware**: NVIDIA GeForce GTX 1080 (7.9GB)

## Executive Summary

Contrary to expectations, the Hilbert space-filling curve optimization for block-sparse attention showed **performance degradation** rather than improvement. The Hilbert variant was on average **44% slower** (0.56x speedup) than the standard implementation.

## Key Findings

### 1. Performance Results

| Sequence Length | Standard (ms) | Hilbert (ms) | Relative Performance |
|-----------------|---------------|--------------|---------------------|
| 2,048 | 15.11 | 28.16 | 0.54x |
| 4,096 | 29.96 | 62.20 | 0.48x |
| 8,192 | 55.77 | 121.96 | 0.46x |
| 16,384 | 134.32 | 285.65 | 0.47x |
| 32,768 | 449.43 | 632.80 | 0.71x |
| 65,536 | 825.45 | 1222.12 | 0.68x |

**Key Observation**: Performance gap narrows at larger sequences (0.46x → 0.71x), suggesting the overhead diminishes with scale.

### 2. Configuration Comparison

Testing different Hilbert configurations at 16K tokens:

| Configuration | Time (ms) | vs Standard |
|---------------|-----------|-------------|
| Standard | 208.09 | 1.00x |
| Hilbert Block-Level | 333.26 | 0.62x |
| Hilbert Element-Level | 251.64 | 0.83x |
| Hilbert Full | 371.53 | 0.56x |

**Best Hilbert**: Element-level only (0.83x) - still slower than standard

### 3. Why Hilbert is Slower

#### **Overhead Sources**:

1. **Index Computation**:
   - Computing Hilbert indices requires bit manipulation
   - Mapping linear indices to Hilbert positions adds overhead
   - Device transfers for index tensors

2. **Memory Access Pattern**:
   - Modern GPUs have sophisticated caching
   - Block-sparse already has good locality (sequential blocks)
   - Hilbert reordering may actually hurt GPU's prefetching

3. **Small Block Sizes**:
   - With 64x64 blocks, overhead dominates any cache benefits
   - Hilbert curve benefits emerge at much larger granularities

4. **GPU Architecture**:
   - GTX 1080 (Pascal) has different cache hierarchy than expected
   - Newer GPUs (H100) might show different results

### 4. Block Size Impact

| Block Size | Implementation | 8K tokens | 16K tokens |
|------------|----------------|-----------|------------|
| 64 | Standard | 104.99ms | 188.85ms |
| 64 | Hilbert | 239.54ms | 417.05ms |
| 128 | Standard | 42.81ms | 228.28ms |
| 128 | Hilbert | 71.08ms | 269.42ms |

**Observation**: Larger blocks (128) show better relative performance for Hilbert (0.60x vs 0.44x)

## Analysis

### Why Theory Didn't Match Practice

1. **GPU vs CPU Assumptions**:
   - Hilbert curves optimize for CPU cache lines
   - GPUs have different memory hierarchies and access patterns
   - Warp-level coalescing matters more than cache locality

2. **Block-Sparse Already Optimal**:
   - Sequential block processing already provides good locality
   - Sparse patterns are designed to minimize memory jumps
   - Additional reordering adds complexity without benefit

3. **Overhead vs Benefit Trade-off**:
   - Index computation overhead is significant
   - Benefits only emerge at very large scales
   - Current implementation may not be optimized enough

### When Hilbert Might Help

1. **Much Larger Sequences**: 1M+ tokens where cache misses dominate
2. **Different Hardware**: CPUs or GPUs with different cache architectures
3. **Different Patterns**: Random sparse patterns (not structured block-sparse)
4. **Batch Processing**: When processing multiple sequences together

## Recommendations

### 1. **Don't Use Hilbert for Block-Sparse**
The current implementation adds overhead without performance benefits on tested hardware.

### 2. **Stick with Standard Block-Sparse**
```python
# Recommended
model = create_block_sparse_attention(
    variant="base",  # Not "hilbert"
    sparsity_ratio=0.01,
    block_size=128  # Larger blocks are more efficient
)
```

### 3. **Future Optimizations**
If pursuing Hilbert optimization:
- Pre-compute indices and cache aggressively
- Use native CUDA kernels for index mapping
- Test on different GPU architectures (H100, A100)
- Consider Hilbert only for sequences > 100K tokens

### 4. **Alternative Optimizations**
Instead of Hilbert ordering, consider:
- Larger block sizes (256, 512)
- Better sparse patterns (learned or adaptive)
- Kernel fusion optimizations
- Mixed precision (FP8 on newer GPUs)

## Conclusion

While Hilbert space-filling curves are elegant and theoretically sound for improving cache locality, the practical implementation for block-sparse attention on GPUs shows performance degradation. The overhead of computing and applying Hilbert orderings outweighs any cache locality benefits, at least on the tested GTX 1080 GPU with sequences up to 65K tokens.

This exploration was valuable as it:
1. Demonstrated that theoretical optimizations don't always translate to practice
2. Highlighted the importance of benchmarking on target hardware
3. Showed that block-sparse attention is already well-optimized
4. Provided a framework for testing future ordering optimizations

The implementation remains available in the codebase for future experimentation on different hardware or at larger scales where the benefits might emerge.