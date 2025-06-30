# Memory Pool Integration Guide

This guide provides detailed instructions for integrating memory pool support into dilated attention modules.

## Overview

Memory pooling is a crucial optimization for handling large sequences efficiently. This guide covers:
- Understanding memory pool benefits
- Integration steps for different module types
- Best practices and common patterns
- Testing and validation

## Why Memory Pooling?

### The Problem

Without memory pooling:
- Each forward pass allocates new tensors
- CUDA allocation is expensive (synchronization overhead)
- Memory fragmentation reduces available memory
- Frequent allocations can cause OOM errors

### The Solution

Memory pooling provides:
- **Pre-allocated buffers**: Reuse tensors across forward passes
- **Reduced allocation overhead**: 10-100x faster than `torch.zeros`
- **Better memory locality**: Contiguous memory regions
- **Adaptive management**: Automatic cleanup under pressure

## Integration Steps

### Step 1: Add Dependencies

```python
from dilated_attention_pytorch.core import get_global_memory_pool
```

### Step 2: Update __init__ Method

Add memory pool parameters to your module's initialization:

```python
class MyDilatedAttention(nn.Module):
    def __init__(
        self,
        segment_lengths: List[int],
        dilation_rates: List[int],
        # ... other parameters ...
        enable_memory_pool: bool = False,  # Disabled by default
        enable_profiling: bool = False,
    ):
        super().__init__()
        
        # ... existing initialization ...
        
        # Memory pool setup
        self.enable_memory_pool = enable_memory_pool
        self._memory_pool = None
        if self.enable_memory_pool:
            self._memory_pool = get_global_memory_pool(
                enable_profiling=enable_profiling,
            )
```

### Step 3: Implement Allocation Methods

Add tensor allocation and deallocation methods:

```python
def _allocate_tensor(
    self,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    zero_init: bool = True,
) -> torch.Tensor:
    """Allocate tensor using memory pool if enabled."""
    if self._memory_pool is not None:
        # Calculate tensor size in MB
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

def _deallocate_tensor(self, tensor: torch.Tensor) -> None:
    """Return tensor to memory pool if enabled."""
    if self._memory_pool is not None:
        self._memory_pool.deallocate(tensor)
```

### Step 4: Replace Tensor Allocations

Identify and replace large tensor allocations in your forward method:

#### Before:
```python
def forward(self, q, k, v):
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Large allocation
    attention_scores = torch.zeros(
        batch_size, num_heads, seq_len, seq_len,
        dtype=q.dtype, device=q.device
    )
```

#### After:
```python
def forward(self, q, k, v):
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Use memory pool
    attention_scores = self._allocate_tensor(
        (batch_size, num_heads, seq_len, seq_len),
        dtype=q.dtype,
        device=q.device,
        zero_init=True
    )
    
    try:
        # ... use attention_scores ...
        return output
    finally:
        # Always deallocate
        self._deallocate_tensor(attention_scores)
```

### Step 5: Add Cleanup Method

Implement proper cleanup for long-running processes:

```python
def cleanup_buffers(self) -> None:
    """Clean up any allocated buffers."""
    if hasattr(self, '_allocated_buffers'):
        for buffer in self._allocated_buffers:
            self._deallocate_tensor(buffer)
        self._allocated_buffers.clear()
    
    # Optional: Clear pool cache
    if self._memory_pool is not None:
        self._memory_pool.clear_cache()
```

## Common Patterns

### Pattern 1: Temporary Buffers

For buffers used within a single forward pass:

```python
def forward(self, x):
    # Allocate temporary buffer
    temp_buffer = self._allocate_tensor(shape, dtype, device)
    
    try:
        # Use buffer
        result = some_operation(x, temp_buffer)
        return result
    finally:
        # Always deallocate
        self._deallocate_tensor(temp_buffer)
```

### Pattern 2: Persistent Buffers

For buffers reused across forward passes:

```python
def __init__(self, ...):
    super().__init__()
    self._persistent_buffers = {}

def _get_or_create_buffer(self, key: str, shape, dtype, device):
    """Get existing buffer or create new one."""
    if key not in self._persistent_buffers:
        self._persistent_buffers[key] = self._allocate_tensor(
            shape, dtype, device, zero_init=False
        )
    
    buffer = self._persistent_buffers[key]
    # Resize if needed
    if buffer.shape != shape:
        self._deallocate_tensor(buffer)
        buffer = self._allocate_tensor(shape, dtype, device, zero_init=False)
        self._persistent_buffers[key] = buffer
    
    return buffer
```

### Pattern 3: Multi-Head Attention

For modules with multiple attention heads:

```python
def forward(self, q, k, v):
    batch_size, seq_len, embed_dim = q.shape
    
    # Allocate for all heads at once
    qkv_buffer = self._allocate_tensor(
        (3, batch_size, self.num_heads, seq_len, self.head_dim),
        dtype=q.dtype,
        device=q.device
    )
    
    try:
        # Split buffer for q, k, v
        q_heads, k_heads, v_heads = qkv_buffer.unbind(0)
        
        # Process attention
        # ...
        
        return output
    finally:
        self._deallocate_tensor(qkv_buffer)
```

## Module-Specific Guidelines

### MultiheadDilatedAttention

Key allocations to replace:
1. QKV projection outputs
2. Attention score matrices
3. Attention weight matrices
4. Output projections

```python
# In forward method
# Replace this:
scores = torch.zeros(batch_size, num_heads, seq_len, seq_len)

# With this:
scores = self._allocate_tensor(
    (batch_size, num_heads, seq_len, seq_len),
    dtype=query.dtype,
    device=query.device
)
```

### Distributed Modules

For distributed training:
1. Pre-allocate communication buffers
2. Reuse gradient accumulation buffers
3. Pool all-reduce temporary tensors

```python
def __init__(self, ...):
    if self.enable_memory_pool:
        # Pre-allocate communication buffers
        self._comm_buffer = self._allocate_tensor(
            (self.world_size, *tensor_shape),
            dtype=torch.float32,
            device=self.device
        )
```

### Transformer Layers

Focus on:
1. Self-attention outputs
2. Cross-attention outputs (decoder)
3. FFN intermediate activations

## Testing Integration

### Unit Tests

Create tests to verify memory pool usage:

```python
def test_memory_pool_integration():
    model = MyDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[1],
        enable_memory_pool=True
    )
    
    # Get initial stats
    pool = model._memory_pool
    initial_allocs = pool.get_stats()['allocation_count']
    
    # Run forward pass
    x = torch.randn(2, 1024, 512)
    output = model(x, x, x)
    
    # Verify allocations were made
    final_allocs = pool.get_stats()['allocation_count']
    assert final_allocs > initial_allocs
```

### Memory Leak Tests

Ensure proper deallocation:

```python
def test_no_memory_leak():
    model = MyDilatedAttention(
        segment_lengths=[1024],
        dilation_rates=[1],
        enable_memory_pool=True
    )
    
    # Baseline memory
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    
    # Run multiple forward passes
    for _ in range(100):
        x = torch.randn(2, 1024, 512, device='cuda')
        output = model(x, x, x)
        del output
    
    # Check memory hasn't grown significantly
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated()
    assert final - baseline < 10 * 1024 * 1024  # < 10MB growth
```

### Performance Benchmarks

Compare with and without pooling:

```python
import time

def benchmark_memory_pool():
    # Without pool
    model_no_pool = MyDilatedAttention(
        segment_lengths=[4096],
        dilation_rates=[1],
        enable_memory_pool=False
    )
    
    # With pool
    model_with_pool = MyDilatedAttention(
        segment_lengths=[4096],
        dilation_rates=[1],
        enable_memory_pool=True
    )
    
    x = torch.randn(1, 4096, 512, device='cuda')
    
    # Warmup
    for _ in range(10):
        _ = model_no_pool(x, x, x)
        _ = model_with_pool(x, x, x)
    
    # Benchmark without pool
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model_no_pool(x, x, x)
    torch.cuda.synchronize()
    time_no_pool = time.time() - start
    
    # Benchmark with pool
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model_with_pool(x, x, x)
    torch.cuda.synchronize()
    time_with_pool = time.time() - start
    
    print(f"Without pool: {time_no_pool:.3f}s")
    print(f"With pool: {time_with_pool:.3f}s")
    print(f"Speedup: {time_no_pool / time_with_pool:.2f}x")
```

## Best Practices

### 1. Size Threshold

Only use memory pool for large tensors (>= 1MB):
```python
if size_mb >= 1.0:
    return self._memory_pool.allocate(...)
else:
    return torch.zeros(...)  # Regular allocation
```

### 2. Error Handling

Always deallocate in finally blocks:
```python
buffer = self._allocate_tensor(...)
try:
    # Use buffer
    result = process(buffer)
    return result
finally:
    self._deallocate_tensor(buffer)
```

### 3. Buffer Reuse

Reuse buffers when shapes are consistent:
```python
if hasattr(self, '_cached_buffer') and self._cached_buffer.shape == shape:
    return self._cached_buffer.zero_()
else:
    self._cached_buffer = self._allocate_tensor(shape, ...)
    return self._cached_buffer
```

### 4. Profiling

Enable profiling during development:
```python
model = MyDilatedAttention(
    ...,
    enable_memory_pool=True,
    enable_profiling=True  # Track detailed statistics
)
```

### 5. Cleanup

Implement proper cleanup:
```python
def __del__(self):
    """Cleanup on deletion."""
    if hasattr(self, 'cleanup_buffers'):
        self.cleanup_buffers()
```

## Troubleshooting

### Issue: No Performance Improvement

Possible causes:
- Tensors too small (< 1MB threshold)
- Pool overhead exceeds allocation cost
- Memory bandwidth limited

Solution:
- Profile tensor sizes
- Adjust size threshold
- Check GPU utilization

### Issue: OOM Errors

Possible causes:
- Pool holding too much memory
- Fragmentation
- Cleanup not triggered

Solution:
```python
# Enable aggressive cleanup
pool = get_global_memory_pool(cleanup_threshold=0.8)

# Manual cleanup
model._memory_pool.cleanup(force=True)
```

### Issue: Incorrect Results

Possible causes:
- Buffer not zeroed when needed
- Buffer reused incorrectly
- Race conditions

Solution:
- Always use `zero_init=True` for accumulators
- Don't share buffers across operations
- Use proper synchronization

## Advanced Topics

### Custom Memory Pools

Create specialized pools:
```python
from dilated_attention_pytorch.core.memory_pool import MemoryPool

class MyCustomPool(MemoryPool):
    def __init__(self):
        super().__init__(
            max_cached_buffers=100,
            cleanup_threshold=0.9,
            enable_profiling=True
        )
    
    def allocate(self, shape, dtype, device):
        # Custom allocation logic
        return super().allocate(shape, dtype, device)
```

### Integration with PyTorch

Work with PyTorch's caching allocator:
```python
# Coordinate with PyTorch's cache
torch.cuda.empty_cache()
model._memory_pool.cleanup()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

### Multi-GPU Considerations

Each GPU needs its own pool:
```python
class DistributedModel(nn.Module):
    def __init__(self, ..., device_id):
        self.device = torch.device(f'cuda:{device_id}')
        self._memory_pool = get_global_memory_pool(device=self.device)
```

## Conclusion

Memory pooling is essential for:
- Long sequence processing (>32K tokens)
- Memory-constrained environments
- Production deployments

Start with basic integration and gradually optimize based on profiling results.