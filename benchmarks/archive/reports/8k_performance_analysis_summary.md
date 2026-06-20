# 8K Performance Anomaly Analysis

## The Problem

The fused kernels showed an unusual performance pattern:
- 4K: 4.71x speedup
- **8K: 1.53x speedup** (anomaly)
- 12K: 2.25x speedup
- 16K: 2.01x speedup

## Root Cause Analysis

### 1. Grid Alignment Issue
- 8K with 64x64 blocks creates 16,384 total blocks
- This is 819.2 blocks per SM on GTX 1080 (20 SMs)
- Poor load balancing causes underutilization

### 2. Cache Threshold
- 8K sequences hit a critical point where working set exceeds L2 cache
- Too large for optimal cache reuse (unlike 4K)
- Too small for streaming benefits (unlike 12K+)

### 3. Shared Memory Constraints (Pascal GPUs)
- GTX 1080 has only 48KB shared memory
- Original configs with BLOCK_D=64 required 56.5KB+
- This caused fallback to less efficient configurations

## The Solution

### Memory-Aware Configuration
```python
# Pascal GPUs (compute capability < 7)
if M == 8192:
    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
BLOCK_D = min(32, D)  # Critical: Reduced from 64 to fit in 48KB
```

### Why This Works
1. **Shared Memory**: 64x64x32 uses only 36.5KB (fits in 48KB limit)
2. **Grid Alignment**: Still not perfect, but avoids memory errors
3. **Consistent Performance**: Removes the configuration thrashing

## Expected Results

With the corrected configuration:
- 8K should achieve ~2.0x speedup (up from 1.53x)
- Smooth performance curve from 4K to 16K
- No shared memory errors

## Key Insights

1. **Hardware Constraints Matter**: Pascal's 48KB shared memory is a hard limit
2. **Grid Alignment Affects Performance**: Poor SM utilization can cause 30%+ slowdown
3. **One Size Doesn't Fit All**: Different sequence lengths need different optimizations

## Future Improvements

1. **Dynamic Block Selection**: Choose blocks based on:
   - Sequence length
   - GPU architecture
   - Available shared memory

2. **Alternative Algorithms for 8K+**:
   - Consider Flash Attention style tiling
   - Use Ring Attention for very long sequences

3. **Hardware-Specific Tuning**:
   - Volta+ (96KB shared memory): Can use larger blocks
   - Ampere+ (164KB shared memory): Even more flexibility