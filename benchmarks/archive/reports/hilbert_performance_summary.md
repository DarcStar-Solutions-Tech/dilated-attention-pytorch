# Hilbert Performance Issues - Summary and Recommendations

## The Core Problem

After extensive benchmarking and analysis, we've identified that **Hilbert reordering is fundamentally hurting performance** in our implementation:

### Performance Data
- **Sequence 1024**: Triton+Hilbert is 41% slower than Triton baseline
- **Sequence 2048**: Triton+Hilbert is 14% slower  
- **Sequence 4096**: Triton+Hilbert is 51% slower
- **Sequence 8192**: Triton+Hilbert is 176% slower!

## Root Causes

1. **Hardware Mismatch**
   - GTX 1080 has only 2MB L2 cache - too small to benefit from Hilbert locality
   - High memory bandwidth (320 GB/s) makes cache optimization less critical
   - GPU prefers coalesced, predictable memory access patterns

2. **Implementation Overhead**
   - Hilbert mapping generation takes ~21ms
   - Tensor reordering adds 15-20ms per forward pass
   - This overhead is 20-30% of total computation time

3. **Algorithmic Mismatch**
   - Hilbert curves optimize for 2D spatial locality
   - Attention has different access patterns (all queries access all keys)
   - The scattered memory access breaks GPU optimizations

4. **Indirect Memory Access**
   - Each K,V access requires loading Hilbert indices first
   - Prevents memory coalescing within warps
   - Disables hardware prefetching

## Recommendations

### 1. **Change Default Behavior**
```python
# Set hilbert_threshold very high to effectively disable it
hilbert_threshold = 65536  # Only use for 64K+ sequences
```

### 2. **Make Hilbert Opt-In Only**
```python
# Default use_hilbert to False in forward()
def forward(self, x, use_hilbert=False, is_causal=False):
    ...
```

### 3. **Add Performance Warning**
```python
if use_hilbert and seq_len < 32768:
    warnings.warn(
        "Hilbert reordering may reduce performance for sequences < 32K. "
        "Consider disabling it with use_hilbert=False."
    )
```

### 4. **Focus on Proven Optimizations**
- **Fused kernels**: Already providing 3.77x speedup at 4K sequences
- **Flash Attention**: Better tiling strategy for GPU architecture
- **Ring Attention**: For very long sequences
- **Block-sparse patterns**: More GPU-friendly than Hilbert

## The Fundamental Issue

Hilbert reordering was designed for different hardware (CPUs with large caches) and different access patterns (2D image processing). For transformer attention on GPUs:

1. **Regular patterns win**: GPUs excel at predictable, coalesced memory access
2. **Cache is less critical**: High bandwidth can hide cache misses
3. **Overhead matters**: Any reordering must provide massive benefits to overcome its cost

## Conclusion

**We should effectively disable Hilbert reordering in the current implementation.** It's making performance worse, not better. The fused kernels we implemented are providing real speedups, while Hilbert is a net negative.

The theoretical benefits of Hilbert curves for cache locality are overwhelmed by:
- Implementation overhead (mapping generation, reordering)
- Hardware limitations (small cache, preference for regular patterns)
- Algorithmic mismatch (attention patterns ≠ 2D spatial locality)

For users who want to experiment, they can explicitly enable it, but it should not be the default behavior.