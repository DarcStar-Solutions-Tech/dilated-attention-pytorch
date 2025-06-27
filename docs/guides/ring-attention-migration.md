# Ring Attention Migration Guide

## Overview

The original Ring Attention implementations (`RingDilatedAttention`, `RingMultiheadDilatedAttention`) have a fundamental flaw that prevents them from achieving the theoretical O(n/ring_size) memory savings. These implementations will be **removed in v0.3.0**.

## The Problem

The broken implementations incorrectly divide queries across devices:
```python
# WRONG - This is what the broken implementation does
q_local = q[:, start_idx:end_idx]  # Each device gets DIFFERENT queries
```

True Ring Attention requires:
- **Full queries** on each device
- Only K/V tensors are chunked and rotated
- Memory scales as O(n/ring_size) for K/V only

## Migration Steps

### 1. Direct Usage of RingDilatedAttention

**Old (Broken)**:
```python
from dilated_attention_pytorch import RingDilatedAttention

attention = RingDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

**New (Correct)**:
```python
from dilated_attention_pytorch.ring_dilated_attention_v2 import RingDilatedAttentionV2

attention = RingDilatedAttentionV2(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

### 2. Using RingMultiheadDilatedAttention

**Old (Broken)**:
```python
from dilated_attention_pytorch import RingMultiheadDilatedAttention

attention = RingMultiheadDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

**New (Correct) - Using Factory Pattern**:
```python
from dilated_attention_pytorch import create_multihead_dilated_attention

attention = create_multihead_dilated_attention(
    "ring",  # This now uses the corrected V2 implementation
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=4
)
```

### 3. Block-Sparse Ring Attention

The block-sparse ring attention implementations also inherit the broken behavior. For now:

**Temporary Solution**:
```python
# Use standard block-sparse without ring
from dilated_attention_pytorch import create_block_sparse_attention

attention = create_block_sparse_attention(
    embed_dim=768,
    num_heads=12,
    sparsity_ratio=0.1,
    pattern_type='dilated_sparse'
)
```

These will be updated to use the correct ring attention in v0.3.0.

## Key Differences

### Memory Usage
- **Broken**: No actual memory savings, may even use more memory
- **Fixed**: True O(n/ring_size) memory scaling

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