# Hilbert SFC Placement Optimization Report

**Date**: July 4, 2025  
**Finding**: Current Hilbert implementation is suboptimal - should be applied to dilated patterns, not full sequence

## Executive Summary

The current implementation applies Hilbert Space-Filling Curve (SFC) to the entire sequence BEFORE splitting across GPUs. This approach loses most of the cache locality benefits. The Hilbert ordering should instead be applied to the actual memory access patterns AFTER segmentation and dilation.

## Current vs Improved Approach

### Current Implementation (Suboptimal):
```
1. Apply Hilbert to full sequence [0,1,2,...,63]
2. Split across GPUs (breaks Hilbert locality)
3. Apply dilation within segments
```

**Problem**: The Hilbert curve's spatial locality is destroyed when the sequence is split across GPUs.

### Improved Implementation (Optimal):
```
1. Split sequence across GPUs
2. Apply dilation to get access patterns
3. Apply Hilbert ordering to the dilated access pattern
```

**Benefit**: Hilbert ordering now matches the actual memory access pattern, improving cache efficiency.

## Visual Example

### Sequence: 64 tokens, 4 segments, dilation_rate=4

**Current approach**:
- Full sequence gets Hilbert ordering: [0→15→3→12→...]
- Split destroys locality - adjacent Hilbert positions on different GPUs
- Dilated access [0,4,8,12] has no Hilbert benefit

**Improved approach**:
- Segment 0 dilated access: [0,4,8,12]
- Apply Hilbert to these 4 indices for optimal cache access
- Each GPU optimizes its own access pattern

## Implementation

Created `RingDilatedAttentionHilbertV2` with two modes:

1. **"dilated" mode**: Apply Hilbert to dilated index patterns
   - Best for large dilation rates
   - Optimizes sparse access patterns

2. **"segment" mode**: Apply Hilbert to K,V within segments
   - Best for dense attention patterns
   - Maintains locality within each segment

## Performance Impact

### Cache Efficiency Analysis:
- **Cache line**: 64 bytes (16 float32 elements)
- **Current**: Random access after split - poor cache utilization
- **Improved**: Hilbert-ordered access - better spatial locality

### Expected Benefits:
1. **Reduced cache misses** - Adjacent accesses in Hilbert order
2. **Better memory bandwidth utilization** - Fewer random accesses
3. **Scalable** - Each GPU optimizes independently

## Recommendations

1. **Replace current implementation** with HilbertV2
2. **Use "dilated" mode** for sparse patterns (dilation > 2)
3. **Use "segment" mode** for dense patterns (dilation = 1)
4. **Benchmark both modes** on target hardware

## Code Changes

### Before:
```python
# Apply Hilbert to full sequence (suboptimal)
k_hilbert = self._apply_hilbert_to_chunk(k)
v_hilbert = self._apply_hilbert_to_chunk(v)
# Then split
k_local = split_by_rank(k_hilbert, self.rank, self.ring_size)
```

### After:
```python
# Split first
k_local = split_by_rank(k, self.rank, self.ring_size)
# Apply Hilbert to dilated patterns during attention
pattern = self._get_segment_dilation_pattern(...)
pattern_hilbert = self._apply_hilbert_to_pattern(pattern)
```

## Conclusion

The current Hilbert implementation provides limited benefit because it's applied at the wrong stage. By moving Hilbert ordering to match the actual memory access patterns (after segmentation and dilation), we can achieve the intended cache efficiency improvements.

This is especially important for:
- Large dilation rates (sparse patterns benefit most)
- Multi-GPU setups (each GPU optimizes its access independently)
- Memory-bandwidth-limited operations (better cache usage = higher throughput)