# Deprecated Memory Pool Implementations

**Status**: DEPRECATED - Scheduled for removal in v0.4.0  
**Replacement**: Use `unified_memory_pool.py` instead

## Deprecated Files

The following memory pool implementations have been consolidated into `unified_memory_pool.py`:

1. **memory_pool.py** - Original UnifiedMemoryPool with excessive complexity
2. **enhanced_memory_pool.py** - Wrapper that combined other pools
3. **bucketed_memory_pool.py** - Size-bucketed allocation (now a feature flag)
4. **fragment_aware_pool.py** - Fragment tracking (now optional feature)
5. **numa_aware_pool.py** - NUMA awareness (now optional feature)

## Migration Guide

### Old Usage (memory_pool.py):
```python
from dilated_attention_pytorch.core.memory_pool import UnifiedMemoryPool

pool = UnifiedMemoryPool(
    strategy="default",
    enable_numa_aware=True,
    enable_bucketing=True
)
```

### New Usage (unified_memory_pool.py):
```python
from dilated_attention_pytorch.core import UnifiedMemoryPool, MemoryPoolConfig

config = MemoryPoolConfig(
    enable_bucketing=True,
    enable_numa_awareness=True,  # Disabled by default
    enable_fragmentation_tracking=False  # Disabled by default
)
pool = UnifiedMemoryPool(config)
```

### Old Usage (enhanced_memory_pool.py):
```python
from dilated_attention_pytorch.core.enhanced_memory_pool import get_enhanced_memory_pool

pool = get_enhanced_memory_pool()
```

### New Usage:
```python
from dilated_attention_pytorch.core import get_global_memory_pool

pool = get_global_memory_pool()  # Uses simplified implementation
```

## Key Changes

1. **Simplified API**: Single configurable class instead of 5 separate implementations
2. **Feature Flags**: Enable only what you need via config
3. **Better Defaults**: NUMA and fragmentation tracking disabled by default (rarely needed)
4. **Reduced Complexity**: ~3000 lines reduced to ~400 lines
5. **Maintained Compatibility**: Aliases ensure existing code continues to work

## Timeline

- v0.3.0 (current): Deprecated files remain but issue warnings
- v0.4.0 (future): Deprecated files will be removed

## Why Consolidate?

1. **Over-engineering**: 5 different pools for essentially the same functionality
2. **Unused Features**: NUMA awareness and fragment tracking add overhead with little benefit
3. **Maintenance Burden**: 5x the code to maintain and test
4. **Confusion**: Unclear which pool to use when
5. **Performance**: Simpler implementation is actually faster