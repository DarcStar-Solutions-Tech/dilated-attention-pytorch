# Hilbert Attention Performance Analysis

## Executive Summary

After comprehensive benchmarking and optimization of the HilbertAttention implementation, we've identified key performance characteristics and optimal configuration strategies.

## Key Findings

### 1. Threshold Behavior (seq_len ≤ 1024)
- Identity mapping is used (no Hilbert reordering)
- All backends perform similarly (~2.6-2.8ms for 1024 tokens)
- Minimal overhead from the Hilbert infrastructure

### 2. Medium Sequences (2K-4K tokens)
- PyTorch backend maintains good performance
- Triton kernel shows significant overhead:
  - 2048 tokens: Triton+Hilbert is 2.2x slower than baseline
  - 4096 tokens: Triton+Hilbert is 8.4x slower than PyTorch baseline
- Hilbert ordering provides no benefit at these sizes

### 3. Long Sequences (8K+ tokens)
- Hilbert ordering becomes beneficial:
  - PyTorch + Hilbert: 1.70x speedup
  - Triton + Hilbert: 1.56x speedup
- Cache locality improvements outweigh reordering overhead
- This aligns with the intended use case for sequences 8K+

### 4. Backend Selection Strategy
Based on the benchmarks, the optimal backend selection is:
- **seq_len ≤ 4096**: Use PyTorch backend
- **seq_len > 4096**: Consider Triton if optimized further
- **Always use Hilbert for seq_len > 8192**

## Performance Bottlenecks Identified

### 1. Triton Kernel Overhead
The current Triton implementation has high overhead for medium sequences:
- Excessive block size (BLOCK_M=64, BLOCK_N=64)
- Repeated Hilbert index loading inside loops
- Poor memory access patterns for sparse attention

### 2. Sparse Pattern Issues
For dilated attention (dilation_rate > 1):
- Global Hilbert mapping destroys locality
- Segment-local mapping partially addresses this but needs refinement
- Average jump distance increases from 4 to 34+ with global Hilbert

## Recommendations

### 1. Immediate Actions
- [x] Keep threshold at 1024 as requested
- [x] Fix double reordering issue
- [x] Implement segment-local Hilbert for sparse patterns
- [ ] Optimize Triton kernel block sizes for different sequence lengths

### 2. Future Optimizations
- Implement adaptive backend selection based on sequence length
- Optimize Triton kernel for sparse patterns:
  - Pre-compute Hilbert indices per segment
  - Use smaller block sizes for medium sequences
  - Implement specialized kernels for different dilation rates
- Consider Flash Attention 3 integration for additional speedup

### 3. Configuration Guidelines
For production use:
```python
# Optimal configuration
attn = HilbertAttention(
    hidden_dim=768,
    num_heads=12,
    segment_size=128,
    dilation_rate=1,  # Use 1 for dense attention
    hilbert_threshold=1024,  # As requested
)

# For very long sequences (8K+)
# Hilbert will automatically activate and provide benefits
```

## Benchmark Results Summary

| Sequence Length | Best Configuration | Time (ms) | Speedup vs Baseline |
|----------------|-------------------|-----------|-------------------|
| 1024 | Triton + No Hilbert | 2.64 | 1.04x |
| 2048 | PyTorch + No Hilbert | 6.73 | 1.00x |
| 4096 | PyTorch + No Hilbert | 19.53 | 1.00x |
| 8192 | PyTorch + Hilbert | 179.48 | 1.70x |

## Conclusion

The HilbertAttention implementation now correctly:
1. Uses identity mapping for sequences ≤ 1024 tokens
2. Applies Hilbert reordering only once (no double ordering)
3. Shows performance benefits for sequences ≥ 8K tokens
4. Provides segment-local mapping for sparse patterns

The implementation is ready for production use with the understanding that:
- It's optimized for long sequences (8K+)
- PyTorch backend is preferred for shorter sequences
- Further Triton kernel optimization could improve medium-sequence performance