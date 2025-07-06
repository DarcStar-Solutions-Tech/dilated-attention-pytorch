# Migration Guide - v0.3.0

**Last Updated**: 2025-07-06

This guide helps you migrate from older versions to v0.3.0, which includes major refactoring and improvements.

## 🚨 Breaking Changes

### Removed Implementations

The following implementations have been removed and consolidated into other modules:

#### 1. DistributedImprovedDilatedAttention
**Status**: ❌ Removed  
**Replacement**: Use factory pattern
```python
# Old (deprecated)
from dilated_attention_pytorch import DistributedImprovedDilatedAttention

# New (v0.3.0+)
from dilated_attention_pytorch import create_multihead_dilated_attention
attention = create_multihead_dilated_attention("distributed", ...)
```

#### 2. DistributedImprovedMultiheadDilatedAttention
**Status**: ❌ Removed  
**Replacement**: Use `DistributedMultiheadDilatedAttention` or factory
```python
# Old (deprecated)
from dilated_attention_pytorch import DistributedImprovedMultiheadDilatedAttention

# New (v0.3.0+)
from dilated_attention_pytorch import DistributedMultiheadDilatedAttention
# OR
attention = create_multihead_dilated_attention("distributed", ...)
```

#### 3. RingDilatedAttentionV2Collective
**Status**: ❌ Removed  
**Replacement**: Use `RingDilatedAttentionHybrid`
```python
# Old (deprecated)
from dilated_attention_pytorch import RingDilatedAttentionV2Collective

# New (v0.3.0+)
from dilated_attention_pytorch import RingDilatedAttentionHybrid
# OR use the alias
from dilated_attention_pytorch import RingDilatedAttention
```

#### 4. RingMultiheadDilatedAttention
**Status**: ❌ Removed  
**Replacement**: Use `RingMultiheadDilatedAttentionHybrid`
```python
# Old (deprecated)
from dilated_attention_pytorch import RingMultiheadDilatedAttention

# New (v0.3.0+)
from dilated_attention_pytorch import RingMultiheadDilatedAttentionHybrid
```

#### 5. Various V2/V3 Implementations
**Status**: ❌ Consolidated  
**Replacement**: Features merged into Hybrid implementations

### API Changes

#### Memory Pool API
The memory pool API has been unified:

```python
# Old (multiple different pools)
from dilated_attention_pytorch import MemoryPool, UnifiedMemoryPool, AdaptiveMemoryPool

# New (v0.3.0+)
from dilated_attention_pytorch.core.unified_memory_pool import SimplifiedMemoryPool
# OR use the compatibility alias
from dilated_attention_pytorch.core import UnifiedMemoryPool
```

#### Configuration Classes
Now using type-safe dataclasses:

```python
# Old
attention = DilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)

# New (also supported, but with validation)
from dilated_attention_pytorch.core import DilatedAttentionConfig

config = DilatedAttentionConfig(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dropout=0.1
)
attention = DilatedAttention(config)
```

## ✅ Backward Compatibility

### Maintained Aliases

These aliases are maintained for backward compatibility:

```python
# All of these work
RingDilatedAttention = RingDilatedAttentionHybrid  # Recommended
RingDilatedAttentionTrue = RingDilatedAttentionHybrid  # Also works
```

### Import Paths

Most import paths remain the same:

```python
# These all still work
from dilated_attention_pytorch import DilatedAttention
from dilated_attention_pytorch import MultiheadDilatedAttention
from dilated_attention_pytorch import ImprovedDilatedAttention
from dilated_attention_pytorch import ImprovedMultiheadDilatedAttention
from dilated_attention_pytorch import LongNet
```

## 🔄 Step-by-Step Migration

### Step 1: Update Imports

Search and replace deprecated imports:

```bash
# Find deprecated imports
grep -r "DistributedImprovedDilatedAttention" .
grep -r "RingDilatedAttentionV2Collective" .
grep -r "RingMultiheadDilatedAttention" .
```

### Step 2: Use Factory Pattern (Recommended)

The factory pattern automatically selects the best implementation:

```python
from dilated_attention_pytorch import create_multihead_dilated_attention

# Old way - manual selection
if distributed:
    model = DistributedImprovedMultiheadDilatedAttention(...)
else:
    model = ImprovedMultiheadDilatedAttention(...)

# New way - automatic selection
model = create_multihead_dilated_attention(
    "auto",  # or "distributed", "improved", "ring", etc.
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2]
)
```

### Step 3: Update Configuration

If using configuration objects:

```python
# Old
params = {
    'segment_lengths': [2048, 4096],
    'dilation_rates': [1, 2],
    'dropout': 0.1
}
attention = DilatedAttention(**params)

# New (with validation)
from dilated_attention_pytorch.core import DilatedAttentionConfig

config = DilatedAttentionConfig(**params)  # Validates parameters
attention = create_dilated_attention("auto", attention_config=config)
```

### Step 4: Update Tests

Update test imports and expected behaviors:

```python
# Old test
def test_distributed():
    from dilated_attention_pytorch import DistributedImprovedDilatedAttention
    model = DistributedImprovedDilatedAttention(...)

# New test
def test_distributed():
    from dilated_attention_pytorch import create_multihead_dilated_attention
    model = create_multihead_dilated_attention("distributed", ...)
```

## 🆕 New Features to Adopt

### 1. Automatic Hardware Optimization

```python
# Automatically uses FP32 on Pascal GPUs, FP16/BF16 on newer
attention = create_multihead_dilated_attention("auto", ...)
```

### 2. Memory Pool with Adaptive Cleanup

```python
# Automatic memory management
from dilated_attention_pytorch.core import MemoryPoolConfig

config = MemoryPoolConfig(
    cleanup_threshold_mb=16,  # Adaptive cleanup
    enable_profiling=True     # Track memory usage
)
```

### 3. Pattern Caching (2x speedup)

```python
# Enabled by default for ring attention
attention = RingDilatedAttention(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    enable_pattern_cache=True  # Default
)
```

### 4. Flash Attention 3 Support

```python
# Automatic FA3 → FA2 → xformers → SDPA fallback
attention = create_multihead_dilated_attention("auto", ...)
```

## 📋 Migration Checklist

- [ ] Update all imports from deprecated classes
- [ ] Replace manual implementation selection with factory pattern
- [ ] Update configuration code to use new dataclasses (optional)
- [ ] Test with new verification script: `python scripts/test_comprehensive.py`
- [ ] Review memory usage with new pooling system
- [ ] Enable pattern caching for repeated sequences
- [ ] Test on target hardware (Pascal optimization is automatic)

## 🐛 Common Issues

### Issue 1: Import Error
```python
ImportError: cannot import name 'DistributedImprovedDilatedAttention'
```
**Solution**: Use the replacement imports shown above

### Issue 2: Parameter Mismatch
```python
TypeError: __init__() got an unexpected keyword argument
```
**Solution**: Check the new parameter names in the implementation overview

### Issue 3: Memory Pool Errors
```python
AttributeError: 'WeakSet' object has no attribute...
```
**Solution**: Update to v0.3.0+ which fixes the WeakSet issue

## 🔗 Resources

- [Implementation Overview](implementation-overview.md) - List of all current implementations
- [Factory Pattern Guide](factory-pattern-guide.md) - How to use the factory pattern
- [Testing Guide](testing-guide.md) - How to verify your migration
- [Performance Guide](performance-guide.md) - Optimization tips

## 💬 Getting Help

If you encounter issues during migration:

1. Check the [comprehensive test script](../../scripts/test_comprehensive.py)
2. Review the [verification reports](../reports/)
3. Open an issue with your specific migration challenge

Remember: The refactoring provides significant benefits including 50-60% code reduction, better performance, and improved maintainability. The migration effort is worth it!