# Fused Kernel Recommendations for Longer Sequences

## Benchmark Results Summary

Based on our analysis and benchmarks, here's what we found:

### Actual Performance Results:
- **2048 tokens**: Fused 16.83x faster (9.8ms vs 165.5ms)
- **4096 tokens**: Fused 1.22x faster (21.8ms vs 26.6ms) 
- **8192 tokens**: Fused 0.56x SLOWER (453.3ms vs 253.5ms)

### Why Performance Degrades at 8192+

1. **Shared Memory Pressure**: 
   - GTX 1080 has only 48KB shared memory
   - At 8192 tokens, even small tiles create memory pressure
   - Forces smaller block sizes, reducing efficiency

2. **Occupancy Issues**:
   - Larger sequences need more registers per thread
   - Reduces the number of concurrent thread blocks
   - Diminishes the benefit of kernel fusion

3. **Cache Thrashing**:
   - 8192 sequence doesn't fit in L2 cache
   - Fused kernel's access pattern may be less cache-friendly
   - Standard kernel benefits from Triton's optimizations

## Recommendations

### 1. **Use Fused Kernels Selectively**

```python
# Optimal range for fused kernels
use_fused_kernel = (
    self._triton_available
    and device.type == "cuda" 
    and 2048 <= M_padded <= 4096  # Narrow the range
    and self._fused_kernels_available
)
```

### 2. **Extend to 8K Only on Modern GPUs**

```python
# For Ampere+ GPUs with more shared memory
if compute_capability >= 8:  # A100, RTX 3090, etc.
    max_fused_seq_len = 8192
else:  # Pascal, Volta
    max_fused_seq_len = 4096
```

### 3. **Different Strategies by Sequence Length**

| Sequence Length | Recommended Approach | Reason |
|----------------|---------------------|---------|
| 1K - 2K | Standard PyTorch | Launch overhead minimal |
| 2K - 4K | **Fused Kernels** | Sweet spot - massive speedup |
| 4K - 8K | Fused (GPU-dependent) | Benefits on modern GPUs |
| 8K - 32K | Standard Triton | Better cache utilization |
| 32K+ | Ring Attention | Memory efficiency critical |

### 4. **Implementation Updates Needed**

1. **Adjust the sequence range check**:
```python
# In hilbert_attention.py
use_fused_kernel = (
    self._triton_available
    and device.type == "cuda" 
    and 2048 <= M_padded <= 4096  # Reduced from 8192
    and self._fused_kernels_available
)
```

2. **Add GPU-aware configuration**:
```python
# In hilbert_attention_fused_v2.py
def get_max_fused_seq_len(device):
    compute_capability = torch.cuda.get_device_capability(device)[0]
    if compute_capability >= 8:  # Ampere+
        return 8192
    elif compute_capability >= 7:  # Volta/Turing  
        return 6144
    else:  # Pascal
        return 4096
```

3. **Consider Flash Attention style tiling for 8K+**:
- Instead of simple fusion, implement Flash Attention's tiling strategy
- Better memory access patterns for long sequences
- Requires more complex kernel design

## Conclusion

**Yes, we should use fused kernels, but only for sequences up to 4096 tokens** (or 8192 on modern GPUs). The dramatic 16.83x speedup at 2048 tokens and 1.22x at 4096 tokens makes them worthwhile, but performance degrades beyond that due to hardware limitations.

For longer sequences (8K+), the standard Triton kernels with Hilbert optimization perform better, and for very long sequences (32K+), specialized algorithms like Ring Attention are necessary.