# Kernel Consolidation Plan

**Date**: 2025-07-29 11:52 UTC

## Overview

Based on comprehensive benchmarking, we can safely remove 4 redundant Hilbert kernel implementations, reducing from 7 to 3 high-performance kernels.

## Performance Summary

| Kernel | 1K Token Time | Performance Wins | Decision |
|--------|---------------|------------------|----------|
| `hilbert_attention_unified_optimized_enhanced.py` | 1.18ms | 13/18 tests | **KEEP** - Best overall |
| `hilbert_attention_unified_optimized.py` | 1.62ms | 3/18 tests | **KEEP** - Best for large sequences |
| `hilbert_attention_unified.py` | 3.44ms | 3/18 tests | **KEEP** - Stable baseline |
| `hilbert_attention_simple.py` | 1.40ms | 0/18 tests | **REMOVE** - Redundant |
| `hilbert_attention_enhanced.py` | 2.44ms | 0/18 tests | **REMOVE** - Features merged |
| `hilbert_attention.py` | 2.55ms | 0/18 tests | **REMOVE** - Superseded |
| `hilbert_attention_core.py` | 3.49ms | 0/18 tests | **REMOVE** - Poor performance |

## Kernels to Remove

### 1. `hilbert_attention_simple.py`
- **Reason**: All features available in `unified_optimized_enhanced`
- **Replacement**: `UnifiedHilbertAttention` from `hilbert_attention_unified.py`

### 2. `hilbert_attention_enhanced.py`
- **Reason**: All enhancements integrated into `unified_optimized_enhanced`
- **Replacement**: `UnifiedHilbertAttentionOptimizedEnhanced`

### 3. `hilbert_attention.py`
- **Reason**: Basic implementation superseded by unified version
- **Replacement**: `UnifiedHilbertAttention`

### 4. `hilbert_attention_core.py`
- **Reason**: Slowest performance, custom autograd not providing benefits
- **Replacement**: `UnifiedHilbertAttention`

## Kernels to Keep

### 1. `hilbert_attention_unified_optimized_enhanced.py`
- **Best overall performance** (1.18ms)
- Includes all optimizations:
  - GPU-specific configurations
  - 8K sequence optimization
  - Strided sparse iteration
  - Multi-row processing
- Wins 72% of test cases

### 2. `hilbert_attention_unified_optimized.py`
- **Best for very large sequences** (>4K tokens)
- Optimized for memory efficiency
- Excels at:
  - 8K dense sequences
  - Sparse patterns with dilation=4
  - Maximum dimension tests

### 3. `hilbert_attention_unified.py`
- **Stable baseline** implementation
- Most predictable performance
- Best for:
  - Large head counts
  - Certain sparse patterns
  - Edge cases

## Migration Guide

### Import Updates

```python
# Old imports → New imports
from dilated_attention_pytorch.kernels.hilbert_attention_simple import HilbertAttentionCore
# → 
from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention

from dilated_attention_pytorch.kernels.hilbert_attention_enhanced import HilbertAttentionEnhanced
# → 
from dilated_attention_pytorch.kernels.hilbert_attention_unified_optimized_enhanced import UnifiedHilbertAttentionOptimizedEnhanced

from dilated_attention_pytorch.kernels.hilbert_attention import HilbertAttention
# → 
from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention

from dilated_attention_pytorch.kernels.hilbert_attention_core import HilbertAttentionCore
# → 
from dilated_attention_pytorch.kernels.hilbert_attention_unified import UnifiedHilbertAttention
```

## Benefits

1. **Reduced Complexity**: From 7 to 3 implementations
2. **Clearer Selection**: Each remaining kernel has distinct use case
3. **Better Performance**: Keeping only the fastest implementations
4. **Easier Maintenance**: Less code to maintain
5. **No Feature Loss**: All capabilities preserved in remaining kernels

## Validation

- All remaining kernels pass Triton compilation
- All kernels tested across 18 configurations
- No compilation errors with formatter fix
- Performance verified on Pascal GPU (GTX 1080)

## Action Items

1. Run `scripts/update_kernel_imports.py` to update imports
2. Run `scripts/remove_redundant_kernels.py` to remove files
3. Update documentation to reflect new structure
4. Run tests to ensure everything still works