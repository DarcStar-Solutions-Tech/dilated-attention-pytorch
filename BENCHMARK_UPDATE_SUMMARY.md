# Benchmark Update Summary

## Changes Made to benchmark_all.py

### 1. Added Block Sparse Imports
```python
# Import block sparse implementations
try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    from dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention import (
        BlockSparseRingMultiheadDilatedAttention,
    )
    HAS_BLOCK_SPARSE = True
except ImportError:
    HAS_BLOCK_SPARSE = False
    print("Note: Block sparse implementations not available")
```

### 2. Added Block Sparse Core Implementations
The benchmark now tests BlockSparseRingDilatedAttention with three sparsity ratios:
- **10% dense (90% sparse)** - Maximum sparsity for extreme speedup
- **25% dense (75% sparse)** - Balanced sparsity/quality tradeoff  
- **50% dense (50% sparse)** - Conservative sparsity

### 3. Added Block Sparse Multihead Implementation
Added BlockSparseRingMultiheadDilatedAttention with 25% density (75% sparse) as the default configuration.

### 4. Benchmark Output Format
The implementations appear in results as:
- `BlockSparseRingDilated_10%`
- `BlockSparseRingDilated_25%`
- `BlockSparseRingDilated_50%`
- `BlockSparseRingMultihead_25%`

## Usage

Run the updated benchmark with:
```bash
# Full benchmark
python benchmark_all.py

# Quick test with smaller sequences
python benchmark_all.py --batch-sizes 1 2 --seq-lens 1024 2048

# Use float32 to avoid dtype issues with multihead
python benchmark_all.py --dtype float32
```

## Expected Performance

Block sparse implementations provide:
- **Memory Reduction**: Proportional to sparsity (e.g., 90% sparse = 90% less memory)
- **Theoretical Speedup**: 1/density (e.g., 25% dense = 4x theoretical speedup)
- **Actual Speedup**: Depends on hardware and sequence length

## Notes

1. Block sparse implementations may have higher initialization overhead
2. Best performance gains are seen with longer sequences (4K+)
3. Use float32 dtype for multihead implementations to avoid dtype conflicts
4. The 'dilated_sparse' pattern is used by default for all block sparse tests