# Pascal GPU FP16 Performance Issue

## Summary

We've identified that **FP16 is 8x SLOWER than FP32** on Pascal GPUs (GTX 1080, P100), which explains the significant performance drops in multi-GPU scenarios.

## Test Results

On NVIDIA GeForce GTX 1080 (Pascal, Compute Capability 6.1):

- **FP32**: 5.75 ± 0.63 ms (712,795 tokens/s)
- **FP16**: 47.14 ± 1.82 ms (86,894 tokens/s)
- **FP16 "Speedup"**: 0.12x (actually 8x SLOWER!)

## Root Cause

Pascal GPUs have very limited FP16 performance:
- No Tensor Core support (introduced in Volta/V100)
- Limited FP16 compute units
- FP16 is only ~2x FP32 theoretical performance (vs 16x on V100)
- In practice, often slower due to conversion overhead

## Code Issue

In `ring_dilated_attention_v2_collective.py`, lines 96-98:
```python
self.dtype = dtype or (
    torch.float16 if self.device.type == "cuda" else torch.float32
)
```

This defaults to FP16 on ALL CUDA devices, which is catastrophic for Pascal GPUs.

## Proposed Solution

### Option 1: Auto-detect Pascal and use FP32
```python
def _get_default_dtype(device):
    """Get appropriate default dtype based on GPU architecture."""
    if device.type == "cuda":
        # Check compute capability
        props = torch.cuda.get_device_properties(device)
        if props.major < 7:  # Pascal (6.x) or older
            # Use FP32 for Pascal and older
            return torch.float32
        else:  # Volta (7.x) and newer
            # Use FP16 for Volta+ with Tensor Cores
            return torch.float16
    return torch.float32

# In __init__:
self.dtype = dtype or _get_default_dtype(self.device)
```

### Option 2: Add Pascal warning
```python
if self.dtype == torch.float16 and self.device.type == "cuda":
    props = torch.cuda.get_device_properties(self.device)
    if props.major < 7:
        warnings.warn(
            f"FP16 is significantly slower than FP32 on Pascal GPUs ({props.name}). "
            f"Consider using dtype=torch.float32 for better performance.",
            RuntimeWarning
        )
```

### Option 3: Change default to FP32 everywhere
Simply change the default to FP32 and let users explicitly opt into FP16:
```python
self.dtype = dtype or torch.float32
```

## Impact

This issue affects ALL Pascal GPU users:
- GTX 1060, 1070, 1080, 1080 Ti
- Tesla P100, P40
- Quadro P-series

For these users, the current defaults make the library unusably slow.

## Recommendation

Implement Option 1 (auto-detect) or Option 3 (conservative default). This will:
- Fix performance regression for Pascal users
- Maintain FP16 benefits for modern GPUs (when explicitly requested)
- Improve out-of-box experience