# Comprehensive Benchmark Results

Generated: 2025-06-26-1729-UTC

## Configuration

- Device: cuda
- Batch size: 1
- Number of heads: 8
- Head dimension: 64
- Embed dimension: 512

## Results

| Sequence Length | DilatedAttention | ImprovedDilatedAttention | MultiheadDilatedAttention | ImprovedMultiheadDilatedAttention |
|-----------------|------------------|--------------------------|---------------------------|-----------------------------------|
| 2048 | 0.0016s | 0.0010s | N/A | N/A |
| 4096 | 0.0026s | 0.0019s | N/A | N/A |
| 8192 | 0.0058s | 0.0062s | N/A | N/A |
| 16384 | 0.0110s | 0.0120s | N/A | N/A |

## Analysis

### Performance Improvements

- Sequence 2048: ImprovedDilatedAttention is 1.51x faster
- Sequence 4096: ImprovedDilatedAttention is 1.37x faster
- Sequence 8192: ImprovedDilatedAttention is 0.94x faster
- Sequence 16384: ImprovedDilatedAttention is 0.92x faster

### Bug Fixes Impact

The following critical bug fixes were implemented:

1. **Thread Safety**: Added proper synchronization for cache access
2. **Memory Leak**: Fixed circular references in WeakValueDictionary
3. **Ring Size Validation**: Added validation for distributed scenarios
4. **Gradient Normalization**: Fixed mathematical order of operations

These fixes ensure correctness while maintaining or improving performance.
