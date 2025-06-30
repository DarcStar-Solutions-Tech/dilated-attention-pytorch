# Pascal GPU Optimization Guide

## Overview

Starting from version 0.2.2, dilated-attention-pytorch includes automatic GPU architecture detection to optimize performance on Pascal GPUs (GTX 1060/1070/1080, Tesla P100, etc.).

## The Problem

Pascal GPUs have very limited FP16 (half precision) performance:
- No Tensor Core support (introduced in Volta/V100)
- Limited FP16 compute units
- FP16 operations can be **5-10x slower** than FP32

Our benchmarks show:
- **FP32**: 5.75 ms (712,795 tokens/s) 
- **FP16**: 47.14 ms (86,894 tokens/s) - **8x slower!**

## The Solution

The library now automatically detects GPU architecture and selects the optimal data type:
- **Pascal and older** (compute < 7.0): Uses FP32 for optimal performance
- **Volta and newer** (compute >= 7.0): Uses FP16 with Tensor Cores

## Usage

### Automatic Mode (Recommended)

```python
# Let the library choose the optimal dtype
model = RingDilatedAttentionV2Collective(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    # dtype not specified - auto-selects based on GPU
)
```

On Pascal GPUs, this will:
1. Detect the GPU architecture
2. Select FP32 for optimal performance
3. Show a one-time informational message

### Explicit Control

```python
# Force a specific dtype
model = RingDilatedAttentionV2Collective(
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    dtype=torch.float32,  # Explicitly use FP32
)
```

If you explicitly use FP16 on Pascal, you'll see a warning:
```
Warning: Using torch.float16 on NVIDIA GeForce GTX 1080 (compute 6.1) may result 
in significantly reduced performance. Pascal GPUs can be 5-10x slower with FP16. 
Consider using dtype=torch.float32 for optimal performance.
```

## Affected GPUs

### Pascal GPUs (Auto-select FP32)
- GeForce GTX 1060, 1070, 1080, 1080 Ti
- Tesla P100, P40, P4
- Quadro P-series
- Titan X (Pascal), Titan Xp

### Modern GPUs (Auto-select FP16)
- Volta: V100, Titan V
- Turing: RTX 2060/2070/2080, T4
- Ampere: RTX 3060/3070/3080/3090, A100
- Ada Lovelace: RTX 4060/4070/4080/4090
- Hopper: H100

## Performance Impact

On Pascal GPUs with the optimization:
- **Before**: 86,894 tokens/s (FP16)
- **After**: 712,795 tokens/s (FP32)
- **Speedup**: 8.2x faster!

## Multi-GPU Considerations

This optimization is especially important for multi-GPU setups with Pascal GPUs:
- FP16 overhead compounds with communication overhead
- FP32 ensures consistent performance across all GPUs
- No code changes needed - automatic detection works per GPU

## Migration Guide

### From v0.2.1 or earlier

If you were manually specifying `dtype=torch.float16`:
```python
# Old (slow on Pascal)
model = RingDilatedAttentionV2Collective(
    ...,
    dtype=torch.float16
)

# New (optimal)
model = RingDilatedAttentionV2Collective(
    ...,
    # Remove dtype parameter for auto-selection
)
```

### Checking Your Configuration

```python
# Check what dtype was selected
print(f"Model dtype: {model.dtype}")
print(f"GPU: {torch.cuda.get_device_name()}")

# Get compute capability
props = torch.cuda.get_device_properties(0)
print(f"Compute capability: {props.major}.{props.minor}")
```

## FAQ

**Q: Will this affect my model accuracy?**
A: No, FP32 provides better numerical precision than FP16.

**Q: What about memory usage?**
A: FP32 uses 2x more memory than FP16. However, on Pascal GPUs, the massive speed improvement outweighs the memory cost.

**Q: Can I still use FP16 if I want?**
A: Yes, explicitly specify `dtype=torch.float16`. You'll see a warning but the library will respect your choice.

**Q: What about mixed precision training?**
A: Use PyTorch's Automatic Mixed Precision (AMP) which intelligently manages dtypes. The library's auto-selection applies to the attention computation specifically.