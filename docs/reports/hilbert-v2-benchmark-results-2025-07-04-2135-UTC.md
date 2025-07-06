# Hilbert V2 Benchmark Results

**Date**: July 4, 2025  
**Hardware**: NVIDIA GTX 1080 (Pascal, 8GB)  
**Implementation**: RingDilatedAttentionHilbertV2

## Executive Summary

Initial benchmark results show that the Hilbert V2 implementation has significant performance overhead compared to the baseline. This indicates implementation inefficiencies that need to be addressed.

## Benchmark Results

### 16K tokens, dilation=4:

| Implementation | Time (ms) | Throughput (tokens/sec) | Relative Speed |
|----------------|-----------|-------------------------|----------------|
| Baseline (no Hilbert) | 5.10 | 3,211,155 | 1.00x |
| Original Hilbert | 5.78 | 2,834,316 | 0.88x |
| Hilbert V2 (dilated) | 518.74 | 31,584 | 0.01x |
| Hilbert V2 (segment) | 537.30 | 30,493 | 0.01x |

## Performance Analysis

### Issues Identified:

1. **Overhead in Pattern Generation**: The V2 implementation recalculates Hilbert patterns frequently
2. **Inefficient Indexing**: Multiple tensor indexing operations in hot path
3. **Cache Miss**: Pattern cache not being effectively utilized

### Original Hilbert Results:

From the full benchmark, the original Hilbert implementation shows:
- **Performance degradation** across all configurations
- **Worse with higher dilation** (0.16x for dilation=8 at 32K)
- **No benefit** from the current placement strategy

## Key Findings

1. **Current Hilbert Placement is Ineffective**:
   - Applying Hilbert before splitting provides no benefit
   - Actually degrades performance in most cases
   - Especially poor with high dilation rates

2. **V2 Implementation Needs Optimization**:
   - Current implementation has too much overhead
   - Pattern generation should be cached more aggressively
   - Need to minimize tensor operations in critical path

3. **Baseline Performs Best**:
   - No Hilbert ordering is currently the fastest
   - Dilation alone provides massive speedups (up to 8x)
   - Adding Hilbert in current form only adds overhead

## Recommendations

1. **Short Term**: Use baseline implementation without Hilbert
2. **Medium Term**: Optimize V2 implementation:
   - Pre-compute all Hilbert patterns
   - Use more efficient indexing
   - Minimize overhead in critical path
3. **Long Term**: Explore hardware-specific optimizations:
   - CUDA kernels for Hilbert reordering
   - Fused operations to reduce memory traffic
   - Cache-aware tiling strategies

## Conclusion

While the theoretical benefits of applying Hilbert SFC to dilated patterns are sound, the current implementation overhead negates these benefits. The baseline implementation without Hilbert ordering remains the best option for production use.

Future work should focus on:
1. Reducing V2 implementation overhead
2. Hardware-specific optimizations
3. Exploring alternative space-filling curves that may be more efficient