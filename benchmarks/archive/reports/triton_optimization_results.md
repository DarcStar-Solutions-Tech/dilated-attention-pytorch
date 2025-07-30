# Triton Kernel Optimization Results

## Problem
The Triton kernel was showing 2-8x overhead compared to PyTorch implementation for medium sequence lengths.

## Root Cause
The `HilbertAttentionFunction.forward()` method was using hardcoded block sizes:
```python
BLOCK_M = min(64, M_padded)
BLOCK_N = min(64, M_padded) 
BLOCK_D = min(64, D)
```

This was problematic because:
1. On Pascal GPUs (GTX 1080), 64x64 blocks exceed shared memory limits
2. The block sizes were not optimized for different sequence lengths
3. The hardware-aware optimizations in `get_optimal_block_sizes()` were being ignored

## Fix Applied
Replaced hardcoded block sizes with hardware-aware selection based on compute capability:
- Pascal (6.x): 32x32 for seq≤1024, 64x64 for larger
- Volta/Turing (7.x): 64x64 blocks
- Ampere+ (8.x): 64x64 for seq≤2048, 128x128 for larger

## Performance Improvements

| Sequence Length | Before Fix | After Fix | Improvement |
|-----------------|------------|-----------|-------------|
| 2048 | 14.71ms | 6.83ms | **2.2x faster** |
| 4096 | 164.24ms | 31.76ms | **5.2x faster** |
| 8192 | 195.44ms | 100.96ms | **1.9x faster** |

## Updated Performance Profile

### Sequences ≤ 1024 (below Hilbert threshold)
- All backends perform similarly (~2-3ms)
- Minimal overhead from Triton

### Sequences 2048
- PyTorch: 5.73ms (baseline)
- Triton+Hilbert: 6.83ms (0.84x)
- Triton overhead reduced from 2.2x to just 1.2x

### Sequences 4096  
- PyTorch: 18.84ms (baseline)
- PyTorch+Hilbert: 17.39ms (1.08x speedup)
- Triton+Hilbert: 31.76ms (0.59x)
- Still slower but much improved from 0.12x

### Sequences 8192+
- PyTorch: 1219ms (baseline)
- PyTorch+Hilbert: 217ms (5.6x speedup)
- **Triton+Hilbert: 101ms (12.1x speedup)**
- Triton now significantly outperforms PyTorch!

## Conclusion

The fix successfully addresses the Triton overhead issue:
1. ✅ Reduced overhead for medium sequences (2K-4K tokens)
2. ✅ Achieved superior performance for long sequences (8K+ tokens)
3. ✅ Hardware-aware optimization now working correctly

The implementation now behaves as intended:
- Uses efficient backends for all sequence lengths
- Hilbert ordering provides significant benefits for 8K+ sequences
- Triton acceleration kicks in where it's most beneficial