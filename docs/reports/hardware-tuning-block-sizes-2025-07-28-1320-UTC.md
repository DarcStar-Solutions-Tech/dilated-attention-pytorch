# Hardware-Specific Block Size Optimization

**Date**: 2025-07-28 13:20 UTC  
**Author**: Claude Code Assistant  
**Status**: Implemented (sequence limits reverted)

## Overview

Implemented hardware-specific block size optimizations for Triton kernels while maintaining full sequence length support on all GPUs.

## Optimized Block Sizes by GPU Architecture

### Pascal (GTX 10xx, P100) - Compute Capability 6.x
```python
# Small sequences (≤256)
BLOCK_M, BLOCK_N, BLOCK_D = 16, 16, 32

# Medium sequences (≤768) 
BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, 32

# Large sequences (>768)
BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, 32  # Keep small to avoid register pressure
```

### Volta/Turing (V100, RTX 20xx) - Compute Capability 7.x
```python
# Small sequences (≤256)
BLOCK_M, BLOCK_N, BLOCK_D = 32, 32, 64

# Medium sequences (≤1024)
BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 64

# Large sequences (>1024)
BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 64
```

### Ampere+ (A100, RTX 30xx/40xx, H100) - Compute Capability 8.x+
```python
# Small sequences (≤256)
BLOCK_M, BLOCK_N, BLOCK_D = 64, 64, 128

# Medium sequences (≤2048)
BLOCK_M, BLOCK_N, BLOCK_D = 128, 128, 128

# Large sequences (>2048)
BLOCK_M, BLOCK_N, BLOCK_D = 128, 128, 128
```

## Key Benefits

1. **Pascal GPUs**: Smaller blocks reduce register pressure and shared memory usage
2. **Volta/Turing**: Medium blocks balance occupancy and data reuse
3. **Ampere+**: Larger blocks maximize throughput with abundant resources

## Implementation

The `get_optimal_block_sizes()` method automatically selects appropriate block sizes based on:
- GPU compute capability (detected at runtime)
- Sequence length
- Available head dimension

All sequences are processed with Triton when CUDA is available, using architecture-optimized block sizes for best performance.

## Usage

No user intervention required - the kernel automatically detects GPU architecture and selects optimal parameters:

```python
module = HilbertAttentionCore(
    hidden_dim=768,
    num_heads=12,
    segment_size=128,
    dilation_rate=2
)

# Automatically uses optimized block sizes for your GPU
output = module(input_tensor)
```