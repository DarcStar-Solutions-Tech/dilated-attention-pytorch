# Migration Guide: RingDistributedDilatedAttention → EnterpriseDistributedDilatedAttention

## Overview

`RingDistributedDilatedAttention` has been renamed to `EnterpriseDistributedDilatedAttention` to accurately reflect its functionality. Despite its original name, this class does NOT implement Ring Attention and does NOT provide O(n/k) memory scaling.

## Key Points

1. **No Ring Attention**: The class wraps `ImprovedMultiheadDilatedAttention` internally
2. **No Memory Scaling**: Each GPU processes the full sequence (O(n) memory per GPU)
3. **Backward Compatible**: The old name still works but emits a deprecation warning

## Migration Steps

### 1. Update Imports

**Before:**
```python
from dilated_attention_pytorch.ring.distributed import RingDistributedDilatedAttention

# Or
from dilated_attention_pytorch.ring.distributed.ring_distributed_dilated_attention import (
    RingDistributedDilatedAttention
)
```

**After:**
```python
from dilated_attention_pytorch.ring.distributed import EnterpriseDistributedDilatedAttention

# Or
from dilated_attention_pytorch.ring.distributed.ring_distributed_dilated_attention import (
    EnterpriseDistributedDilatedAttention
)
```

### 2. Update Class Usage

**Before:**
```python
model = RingDistributedDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=world_size  # This parameter was misleading!
)
```

**After:**
```python
model = EnterpriseDistributedDilatedAttention(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=world_size  # Still accepted for compatibility, but doesn't enable ring attention
)
```

### 3. Update Factory Usage

**Before:**
```python
attention = create_multihead_dilated_attention(
    "multihead_ring_distributed",
    embed_dim=768,
    num_heads=12
)
```

**After:**
```python
attention = create_multihead_dilated_attention(
    "multihead_enterprise_distributed",  # New type name
    embed_dim=768,
    num_heads=12
)
```

## If You Need True Ring Attention

If you were expecting O(n/k) memory scaling with ring attention, use one of these implementations instead:

### 1. **RingDilatedAttentionHilbertGPUOptimized** (Recommended)
```python
from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized

attention = RingDilatedAttentionHilbertGPUOptimized(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.0,
    ring_size=world_size  # This DOES enable true ring attention!
)
```

### 2. **BlockSparseRingDilatedAttention** (Ring + Sparsity)
```python
from dilated_attention_pytorch import BlockSparseRingDilatedAttention

attention = BlockSparseRingDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    sparsity_config={"sparsity_ratio": 0.9}
)
```

### 3. **StandardRingAttention** (When exported)
```python
# Coming soon - clean reference implementation
from dilated_attention_pytorch.ring import StandardRingAttention

attention = StandardRingAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=world_size
)
```

## Memory Comparison

### EnterpriseDistributedDilatedAttention (formerly RingDistributed)
- 1 GPU: 32,768 tokens → 857 MB
- 2 GPUs: 32,768 tokens → 857 MB per GPU (no reduction!)
- 4 GPUs: 32,768 tokens → 857 MB per GPU (no reduction!)

### True Ring Attention Implementations
- 1 GPU: 32,768 tokens → 857 MB
- 2 GPUs: 16,384 tokens per GPU → 428 MB per GPU
- 4 GPUs: 8,192 tokens per GPU → 214 MB per GPU

## Deprecation Timeline

- **Current**: Deprecation warning emitted when using old name
- **v0.4.0**: Warning becomes more prominent
- **v0.5.0**: Old name removed completely

## Questions?

If you have questions about this migration:
1. Check the [Ring Attention Implementations Summary](../ring-attention-implementations-summary.md)
2. See the [Ring Attention Analysis Report](../reports/ring-attention-analysis-2025-07-09-1921-UTC.md)
3. Open an issue on GitHub