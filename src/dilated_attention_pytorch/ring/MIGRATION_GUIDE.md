# Ring Attention Migration Guide

**Date**: 2025-07-09  
**Version**: 0.3.0

## Overview

This guide helps you migrate from deprecated ring attention implementations to the new standardized API. The consolidation reduces 20+ implementations to 4 core variants while maintaining all functionality.

## Why Migrate?

1. **Correct Implementation**: All deprecated versions either use `all_gather` (breaking O(n) memory) or have other issues
2. **Standardized API**: Consistent interface across all ring attention variants
3. **Better Testing**: Comprehensive test coverage for ring communication
4. **Active Maintenance**: Only the core implementations will receive updates

## Migration Map

### Implementations to Remove/Deprecate

| Old Implementation | Issue | Migration Target |
|-------------------|-------|------------------|
| `RingDilatedAttentionHilbertCoreFixed` | Uses `all_gather` | `HilbertRingAttention` |
| `RingDilatedAttentionV2*` | Various versions with issues | `StandardRingAttention` |
| `RingDilatedAttentionFixed*` | Multiple "fixed" versions | `StandardRingAttention` |
| `RingDilatedAttentionMemoryEfficient` | Unclear benefits | `StandardRingAttention` |
| `RingDilatedAttentionSDPA` | Can be a config option | `StandardRingAttention` with SDPA backend |

### Core Implementations to Keep

1. **StandardRingAttention** - Basic ring attention with all features
2. **HilbertRingAttention** - Ring attention with Hilbert curve optimization  
3. **DistributedRingAttention** - Enterprise features (DeepSpeed, monitoring)
4. **BlockSparseRingAttention** - Combined with block-sparse patterns

## Migration Examples

### Example 1: Basic Ring Attention

**Old Code:**
```python
from dilated_attention_pytorch.ring.base import RingDilatedAttentionCorrect

attention = RingDilatedAttentionCorrect(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)
```

**New Code:**
```python
from dilated_attention_pytorch.ring import StandardRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)
attention = StandardRingAttention(config)
```

### Example 2: Hilbert Optimized

**Old Code:**
```python
from dilated_attention_pytorch.ring.hilbert import RingDilatedAttentionHilbertOptimizedFixed

attention = RingDilatedAttentionHilbertOptimizedFixed(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    use_hilbert=True
)
```

**New Code:**
```python
from dilated_attention_pytorch.ring import HilbertRingAttention, RingAttentionConfig

config = RingAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    use_hilbert=True,
    hilbert_curve_level=10
)
attention = HilbertRingAttention(config)
```

### Example 3: Using Presets

**New Code (Recommended):**
```python
from dilated_attention_pytorch.ring import create_ring_attention, get_preset_config

# For development
attention = create_ring_attention("standard", config=get_preset_config("development"))

# For production
attention = create_ring_attention("standard", config=get_preset_config("production"))

# For large scale training
attention = create_ring_attention("distributed", config=get_preset_config("large_scale"))
```

## Configuration Changes

### Old Style (Various Parameters)
```python
attention = RingDilatedAttentionV3(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1,
    use_flash_attention=True,
    memory_efficient=True,
    # ... many inconsistent parameters
)
```

### New Style (Configuration Object)
```python
config = RingAttentionConfig(
    # Required
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    
    # Optional with defaults
    dropout=0.1,
    
    # Ring-specific
    ring_size=None,  # Auto-detect from world_size
    communication_backend="nccl",
    enable_error_recovery=True,
    
    # Memory optimization
    use_memory_pool=True,
    preallocate_buffers=True,
    
    # Performance
    overlap_communication=True,
    use_fused_kernels=True,
    compile_mode="reduce-overhead",
    
    # Monitoring
    log_communication_stats=False,
    enable_profiling=False
)
```

## API Differences

### Forward Method

All implementations now have a consistent forward signature:

```python
def forward(
    self,
    query: Tensor,          # (batch, seq_len, num_heads, head_dim)
    key: Tensor,            # (batch, seq_len, num_heads, head_dim)
    value: Tensor,          # (batch, seq_len, num_heads, head_dim)
    attention_mask: Optional[Tensor] = None,
    is_causal: bool = False,
    already_split: bool = False,  # New: indicates if already split for ring
) -> Tensor:
```

### Key Behavioral Changes

1. **No more all_gather**: Implementations using `all_gather` are deprecated
2. **Sequence splitting**: Always happens BEFORE QKV projection
3. **Error handling**: Graceful degradation instead of crashes
4. **Statistics**: Built-in communication monitoring

## Testing Your Migration

```python
# Verify correctness
def test_migration():
    old_attention = OldRingAttention(...)  # Your old implementation
    new_attention = StandardRingAttention(config)
    
    # Same inputs
    q = torch.randn(2, 4096, 8, 64)
    k = torch.randn(2, 4096, 8, 64)
    v = torch.randn(2, 4096, 8, 64)
    
    # Compare outputs
    old_out = old_attention(q, k, v)
    new_out = new_attention(q, k, v)
    
    # Should be numerically close
    assert torch.allclose(old_out, new_out, rtol=1e-3, atol=1e-5)
```

## Deprecation Timeline

- **v0.3.0** (Current): Deprecated implementations marked with warnings
- **v0.4.0** (Next): Deprecated implementations moved to `legacy/` folder
- **v0.5.0** (Future): Deprecated implementations removed

## Getting Help

If you encounter issues during migration:

1. Check the [examples/standardized_ring_attention_example.py](../../../examples/standardized_ring_attention_example.py)
2. Run tests: `pytest tests/ring/test_ring_communication.py`
3. Enable debug logging: `config.log_communication_stats = True`
4. File an issue with your migration problem

## Summary

The new standardized ring attention provides:
- ✅ Correct O(n/k) memory scaling (no all_gather)
- ✅ Consistent API across all variants
- ✅ Better error handling and recovery
- ✅ Communication statistics and monitoring
- ✅ Comprehensive test coverage
- ✅ Active maintenance and updates