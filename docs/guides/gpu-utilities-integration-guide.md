# GPU Utilities Integration Guide

This guide explains the comprehensive GPU utilities system and how it integrates with the Hilbert attention implementation.

## Overview

The GPU utilities system provides:
- Automatic GPU architecture detection
- Optimal attention backend selection (FA3, FA2, SDPA, xformers, manual)
- Architecture-specific dtype recommendations
- Memory and compute capability information
- Backend benchmarking capabilities

## Components

### 1. GPU Detection (`gpu_utils.py`)

The `GPUInfo` dataclass provides comprehensive GPU information:

```python
from dilated_attention_pytorch.utils import get_gpu_info

# Get GPU information
gpu_info = get_gpu_info()

# Access properties
print(f"GPU: {gpu_info.name}")
print(f"Architecture: {gpu_info.architecture}")
print(f"Compute capability: {gpu_info.compute_capability}")
print(f"Memory: {gpu_info.total_memory_gb:.1f} GB")
print(f"Optimal dtype: {gpu_info.optimal_dtype}")
print(f"Has Flash Attention 3: {gpu_info.has_flash_attn_3}")
```

### 2. Backend Selection

The system automatically selects the best attention backend based on:
- GPU architecture
- Available libraries (Flash Attention, xformers)
- Sequence length
- Whether custom masks or dilation are used

```python
from dilated_attention_pytorch.utils import select_gpu_attention_backend

# Automatic selection
backend = select_gpu_attention_backend(
    seq_len=1024,
    use_dilation=True,  # Limits backend options
    is_causal=True
)
```

### 3. Backend Benchmarking

Benchmark available backends to find the fastest:

```python
from dilated_attention_pytorch.utils import benchmark_attention_backends

results = benchmark_attention_backends(
    batch_size=2,
    seq_len=1024,
    num_heads=12,
    head_dim=64
)

for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
    print(f"{backend}: {time_ms:.2f} ms")
```

## Integration with Hilbert Attention

### RingDilatedAttentionHilbertGPUOptimized

The new `RingDilatedAttentionHilbertGPUOptimized` class integrates all GPU utilities:

```python
from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized

# Create with automatic GPU optimization
model = RingDilatedAttentionHilbertGPUOptimized(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[1024, 2048, 1024],
    dilation_rates=[1, 2, 4],
    # Optional parameters:
    attention_backend=None,      # Auto-select best backend
    dtype=None,                  # Auto-select optimal dtype
    benchmark_backends=True,     # Benchmark on initialization
)

# Check what was selected
print(f"Using backend: {model.attention_backend}")
print(f"Using dtype: {model.dtype}")
print(f"GPU: {model.gpu_info.name}")
```

### Backend-Specific Attention Computation

The implementation uses different backends optimally:

1. **Flash Attention 3** (H100): Uses FP8 when available
2. **Flash Attention 2** (A100): Optimized for Ampere
3. **SDPA**: PyTorch's native scaled_dot_product_attention
4. **xformers**: Memory-efficient attention
5. **Manual**: Fallback implementation

## SDPA with Dilation Masks

For dilated attention patterns, custom masks can be used with SDPA:

```python
import torch
import torch.nn.functional as F

def create_dilated_mask(seq_len, dilation_rate, device, dtype):
    """Create a dilated attention mask for SDPA."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    
    for i in range(seq_len):
        # Each position attends to positions at dilated intervals
        attend_positions = torch.arange(i % dilation_rate, seq_len, dilation_rate, device=device)
        mask[i, attend_positions] = 0.0
    
    return mask

# Use with SDPA
mask = create_dilated_mask(seq_len=64, dilation_rate=2, device=device, dtype=dtype)
output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

## Architecture-Specific Optimizations

### Pascal and Older (GTX 1080, P100)
- Uses FP32 by default (limited FP16 performance)
- Prefers xformers or SDPA over Flash Attention
- Smaller block sizes (32)

### Volta (V100)
- Good FP16 performance but no Flash Attention support
- Uses xformers or SDPA
- Medium block sizes (64)

### Turing (T4, RTX 2000)
- Supports Flash Attention 1/2
- Good FP16 performance
- Medium block sizes (64)

### Ampere (A100, RTX 3000)
- Excellent Flash Attention 2 support
- BF16 support for better training stability
- Larger block sizes (128)

### Hopper (H100)
- Flash Attention 3 with FP8 support
- Exceptional performance (up to 75% utilization)
- Largest block sizes (256)

## Usage Examples

### 1. Basic Usage

```python
from dilated_attention_pytorch import RingDilatedAttentionHilbertGPUOptimized

# Let the system choose optimal settings
model = RingDilatedAttentionHilbertGPUOptimized(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
)

# Forward pass
x = torch.randn(2, 6144, 768, device=model.device, dtype=model.dtype)
output = model(x)
```

### 2. Force Specific Backend

```python
# Force Flash Attention 2
model = RingDilatedAttentionHilbertGPUOptimized(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048],
    dilation_rates=[1],
    attention_backend="flash_attn_2",
    dtype=torch.float16,
)
```

### 3. Benchmark Different Configurations

```python
# Run benchmarks on initialization
model = RingDilatedAttentionHilbertGPUOptimized(
    embed_dim=768,
    num_heads=12,
    segment_lengths=[1024, 2048, 1024],
    dilation_rates=[1, 2, 4],
    benchmark_backends=True,  # Will log benchmark results
)
```

## Testing and Verification

Run the provided test scripts:

```bash
# Test GPU detection and utilities
python scripts/test_gpu_utilities.py

# Quick verification
python scripts/verify_gpu_integration.py

# SDPA with dilation demo
python scripts/demo/sdpa_dilated_mask_demo.py
```

## Performance Tips

1. **Let the system auto-select**: The GPU detector usually makes optimal choices
2. **Use optimal dtype**: FP16 on most GPUs, BF16 on Ampere+, FP32 on Pascal
3. **Benchmark your workload**: Different sequence lengths may favor different backends
4. **Consider memory**: Flash Attention uses less memory than SDPA with masks
5. **Profile with your data**: The benchmarking utilities help find the best configuration

## Troubleshooting

### Backend Not Available

If a backend isn't available:
```python
gpu_info = get_gpu_info()
print(f"Flash Attention available: {gpu_info.has_flash_attn}")
print(f"xformers available: {gpu_info.has_xformers}")
print(f"SDPA available: {gpu_info.has_sdpa}")
```

### Poor Performance

1. Check if using optimal dtype for your GPU
2. Verify backend selection matches your workload
3. Run benchmarks to compare backends
4. Consider sequence length and batch size effects

### Memory Issues

1. Use Flash Attention backends when available (lower memory)
2. Reduce batch size or sequence length
3. Use chunked processing for very long sequences