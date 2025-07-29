# Kernel Cleanup Plan

## Overview

After implementing the unified adaptive kernel and reviewing all kernel implementations, we can now consolidate and remove redundant files. The main `HilbertAttention` class already provides automatic optimization and handles all use cases.

## Files to Remove

### 1. **Redundant Core Implementations**
- `hilbert_attention_core_fixed.py` - Functionality already in main implementation
- `hilbert_attention_core_v2.py` - Sparse Hilbert ordering can be added as option to main

### 2. **Fused Kernel Variants**
- `hilbert_attention_fused_simple.py` - Superseded by v2
- `hilbert_attention_fused.py` - Keep temporarily for fused optimizations
- `hilbert_attention_fused_v2.py` - Keep temporarily for GPU-specific configs

### 3. **Optimization Variants**
- `hilbert_attention_optimized.py` - Segment optimization already in main
- `hilbert_attention_sparse_optimized.py` - Strided iteration can be added to main
- `hilbert_attention_triton_wrapper.py` - Interface already handled differently

### 4. **Unified Kernels (Our Recent Work)**
- Keep `hilbert_attention_unified.py` - Good reference implementation
- Keep `hilbert_attention_unified_optimized.py` - Contains valuable optimizations

## Files to Keep

### Core Architecture (4 files)
1. `hilbert_attention.py` - Main implementation with auto-optimization
2. `hilbert_attention_core.py` - Triton kernels (used by main)
3. `hilbert_attention_simple.py` - PyTorch fallback (used by main)
4. `cache_manager.py` - Memory management utility

### Valuable References (4 files)
5. `hilbert_attention_unified.py` - Clean unified kernel design
6. `hilbert_attention_unified_optimized.py` - Performance optimizations
7. `hilbert_attention_fused.py` - Fused operations to integrate
8. `hilbert_attention_fused_v2.py` - GPU-specific configurations

## Integration Plan

### Phase 1: Immediate Cleanup (Remove 5 files)
- Remove definitely redundant files:
  - `hilbert_attention_core_fixed.py`
  - `hilbert_attention_core_v2.py`
  - `hilbert_attention_fused_simple.py`
  - `hilbert_attention_optimized.py`
  - `hilbert_attention_triton_wrapper.py`

### Phase 2: Feature Integration
Before removing the remaining files, integrate their key features:

1. **From `hilbert_attention_fused_v2.py`:**
   - GPU-specific block configurations
   - Special optimizations for 8K sequences

2. **From `hilbert_attention_sparse_optimized.py`:**
   - Strided iteration for sparse patterns
   - Direct sparse position calculation

3. **From unified kernels:**
   - Adaptive configuration logic
   - Segment boundary pre-computation

### Phase 3: Final Cleanup
After integration, remove:
- `hilbert_attention_sparse_optimized.py`
- `hilbert_attention_fused.py`
- `hilbert_attention_fused_v2.py`

## Test Updates Required

1. Update `test_kernel_imports.py` to remove tests for deleted modules
2. Update `test_hilbert_attention_triton_wrapper.py` or remove if no longer needed
3. Update benchmark scripts that reference old implementations

## Benefits

- **Reduced Maintenance**: From 16 kernel files to 4-6 core files
- **Clearer Architecture**: Single entry point with `HilbertAttention`
- **Better Performance**: Automatic optimization selection
- **Easier Testing**: Fewer implementations to test
- **Simpler Documentation**: One API to document

## Migration Path

For any code using old implementations:
```python
# Old
from kernels.hilbert_attention_optimized import HilbertAttentionOptimized
from kernels.hilbert_attention_sparse_optimized import SparseHilbertAttention

# New
from kernels import HilbertAttention
# All optimizations are applied automatically
```

## Timeline

1. **Immediate**: Remove 5 redundant files (Phase 1)
2. **Next Sprint**: Integrate key features from remaining files
3. **Following Sprint**: Final cleanup after integration