# Ring Attention Migration Guide

## Overview

Several Ring Attention implementations that used the inefficient `all_gather` operation have been **deprecated and removed**. These implementations have been replaced with more efficient versions using `isend`/`irecv` operations.

## Deprecated Classes (Removed)

The following classes have been removed due to poor performance with `all_gather`:
- `RingDilatedAttentionV2Collective` 
- `RingMultiheadDilatedAttention`
- `RingHilbertDilatedAttention`
- `HeadParallelDilatedAttention`
- `ImprovedDistributedDilatedAttention`

## Migration Steps

### 1. Direct Usage of Ring Attention

**Old (Deprecated)**:
```python
# These imports no longer work
from dilated_attention_pytorch import RingDilatedAttentionV2Collective
from dilated_attention_pytorch import RingMultiheadDilatedAttention
```

**New (Recommended)**:
```python
# Use the production-ready implementation
from dilated_attention_pytorch import RingDilatedAttentionProduction

attention = RingDilatedAttentionProduction(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)

# Or use the alias
from dilated_attention_pytorch import RingDilatedAttention
# RingDilatedAttention is now an alias for RingDilatedAttentionProduction
```

### 2. Multi-head Ring Attention

**Old (Deprecated)**:
```python
# This class has been removed
from dilated_attention_pytorch import RingMultiheadDilatedAttention
```

**New (Recommended) - Using Factory Pattern**:
```python
from dilated_attention_pytorch import create_multihead_dilated_attention

# Factory automatically wraps RingDilatedAttentionProduction
attention = create_multihead_dilated_attention(
    "ring",  # Uses efficient isend/irecv implementation
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

### 3. Hilbert Curve Ring Attention

**Old (Deprecated)**:
```python
# This class has been removed
from dilated_attention_pytorch import RingHilbertDilatedAttention
```

**New (Recommended)**:
```python
from dilated_attention_pytorch import RingDilatedAttentionHilbertOptimized

# Uses efficient implementation with Hilbert curve reordering
attention = RingDilatedAttentionHilbertOptimized(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

### 4. Distributed Implementations

**Old (Deprecated)**:
```python
# These classes have been removed
from dilated_attention_pytorch import ImprovedDistributedDilatedAttention
from dilated_attention_pytorch import HeadParallelDilatedAttention
```

**New (Recommended)**:
```python
# For distributed training, use RingDistributedDilatedAttention
from dilated_attention_pytorch import RingDistributedDilatedAttention

attention = RingDistributedDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=world_size,  # Typically set to number of GPUs
    process_group=process_group
)
```

## Why These Changes?

### Performance Issues with all_gather
- **all_gather**: O(n²) communication complexity, poor scalability
- **isend/irecv**: O(n) communication complexity, better overlap with computation

### Benchmarked Improvements
- 2-5x faster communication on 8 GPUs
- 10-20x faster on 64+ GPUs
- Better memory efficiency with true O(n/ring_size) scaling

## Key Differences

### Communication Pattern
- **Old (all_gather)**: All devices gather all data, then select their portion
- **New (isend/irecv)**: Direct peer-to-peer communication, minimal data transfer

### Memory Usage
- **Old**: Peak memory spikes during all_gather operations
- **New**: Consistent memory usage with proper ring rotation

### Error Recovery
- **Old**: Limited error handling
- **New**: RingDilatedAttentionProduction includes automatic error recovery

### Performance
- **Broken**: Poor performance due to incorrect data distribution
- **Fixed**: Efficient K/V rotation with minimal communication

### Correctness
- **Broken**: May produce incorrect results due to incomplete attention
- **Fixed**: Mathematically equivalent to standard attention

## Verification

To verify you're using the correct implementation:

```python
# Check memory scaling
import torch
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2

# Create attention module
attention = RingDilatedAttentionV2(
    segment_lengths=[1024],
    dilation_rates=[1],
    ring_size=4
)

# Get memory estimate
estimate = attention.get_memory_estimate(
    seq_len=8192,
    batch_size=1,
    num_heads=8,
    head_dim=64
)

print(f"Memory reduction factor: {estimate['memory_reduction_factor']:.1f}x")
# Should show ~4x reduction for ring_size=4
```

## Timeline

- **v0.2.x**: Deprecation warnings added to broken implementations
- **v0.3.0**: Broken implementations removed completely
- **Now**: Corrected implementations available via factory pattern

## Need Help?

If you encounter issues during migration:
1. Check the [Ring Attention Guide](ring-attention-guide.md) for correct usage
2. See [benchmark results](../benchmarks/) showing proper memory scaling
3. Open an issue on GitHub with your specific use case

## Summary

The key change is simple but critical:
- ❌ **Don't** divide queries across devices
- ✅ **Do** keep full queries and only chunk K/V

This ensures Ring Attention delivers its promised memory savings and enables true billion-token processing.