# Memory Pool Integration Summary

Generated: 1751257065.7696989

## Modules Requiring Memory Pool Support

Total modules: 6
High priority: 4
Medium priority: 2

## Integration Template

```python

# Add to imports
from .core import get_global_memory_pool

# Add to __init__ parameters
def __init__(
    self,
    # ... existing parameters ...
    enable_memory_pool: bool = False,
    enable_profiling: bool = False,
    # ... rest of parameters ...
):
    # ... existing init code ...
    
    # Memory pool setup
    self.enable_memory_pool = enable_memory_pool
    self._memory_pool = None
    if self.enable_memory_pool:
        self._memory_pool = get_global_memory_pool(
            enable_profiling=enable_profiling,
        )

# Add allocation method
def _allocate_tensor(
    self,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    zero_init: bool = True,
) -> torch.Tensor:
    """Allocate tensor using memory pool if enabled."""
    if self._memory_pool is not None:
        # Calculate tensor size
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        bytes_per_element = (
            torch.finfo(dtype).bits // 8
            if dtype.is_floating_point
            else torch.iinfo(dtype).bits // 8
        )
        size_mb = (num_elements * bytes_per_element) / (1024 * 1024)
        
        # Only use pool for tensors >= 1MB
        if size_mb >= 1.0:
            tensor = self._memory_pool.allocate(shape, dtype, device)
            if zero_init:
                tensor.zero_()
            return tensor
    
    # Fallback to regular allocation
    if zero_init:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)

# Add deallocation method
def _deallocate_tensor(self, tensor: torch.Tensor) -> None:
    """Return tensor to memory pool if enabled."""
    if self._memory_pool is not None:
        self._memory_pool.deallocate(tensor)

```

## Module-Specific Guides

See individual files in this directory for detailed integration instructions.

## Best Practices

1. **Enable by default**: Keep memory pool disabled by default for backward compatibility
2. **Size threshold**: Only use pool for tensors >= 1MB to avoid overhead
3. **Cleanup**: Always implement proper deallocation in cleanup methods
4. **Reuse**: Consider reusing buffers across forward passes when possible
5. **Profiling**: Enable profiling during development to measure impact

## Testing

After integration, add tests to verify:
- Memory pool is used for large allocations
- Deallocation is properly handled
- No memory leaks occur
- Performance improves for large sequences
