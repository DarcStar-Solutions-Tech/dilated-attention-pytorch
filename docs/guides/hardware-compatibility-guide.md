# Hardware Compatibility and Performance Guide

## Overview

This guide provides detailed information about hardware compatibility, performance characteristics, and optimization recommendations for Dilated Attention implementations across different GPU architectures.

## GPU Architecture Compatibility

### Tensor Core Support

| Architecture | GPU Examples | Tensor Cores | FP16 Performance | Recommended |
|-------------|--------------|--------------|------------------|-------------|
| **Hopper** (H100) | H100, H800 | 4th Gen | 16x FP32 | ✅ Best |
| **Ada Lovelace** | RTX 4090, 4080 | 4th Gen | 8x FP32 | ✅ Excellent |
| **Ampere** | A100, RTX 3090 | 3rd Gen | 4x FP32 | ✅ Excellent |
| **Turing** | RTX 2080, T4 | 1st/2nd Gen | 2x FP32 | ✅ Good |
| **Volta** | V100 | 1st Gen | 2x FP32 | ✅ Good |
| **Pascal** | GTX 1080, P100 | None | **0.03x FP32** | ⚠️ Limited |
| **Maxwell** | GTX 980, M40 | None | No FP16 | ❌ Avoid |

### Critical Performance Data

#### Pascal Architecture (GTX 1080/1080 Ti) Limitations

Based on extensive benchmarking (June 2025), Pascal GPUs show severe FP16 limitations:

```
GTX 1080 Measured Performance:
- FP32: 8.9 TFLOPS (theoretical), ~5.0 TFLOPS (measured)
- FP16: 0.3 TFLOPS (theoretical), ~0.3 TFLOPS (measured)
- FP16/FP32 ratio: 1/32 (vs 2x on modern GPUs)
```

**Real-world Impact on Ring Attention:**
- FP32: 312.92 ms for 4K sequence
- FP16: 695.04 ms for 4K sequence (2.2x SLOWER)
- Memory saved: 22% (864MB → 676MB)
- Performance lost: 120%

## Precision Recommendations by GPU

### Use FP16/BF16
- ✅ **Hopper** (H100): Use FP8 where possible, FP16/BF16 otherwise
- ✅ **Ada/Ampere** (RTX 40/30, A100): Always use FP16/BF16
- ✅ **Turing/Volta** (RTX 20, V100): Use FP16 with AMP
- ✅ **A100 and newer**: Prefer BF16 over FP16

### Use FP32
- ⚠️ **Pascal** (GTX 1080/1080 Ti): Always use FP32
- ⚠️ **P100**: Use FP32 despite having better FP16 than GTX 1080
- ❌ **Maxwell and older**: Only FP32 supported

## Operation-Specific Performance

### Attention-Critical Operations on Pascal

Benchmarked on GTX 1080:
| Operation | FP32 Time | FP16 Time | Slowdown |
|-----------|-----------|-----------|----------|
| Softmax | 0.41 ms | 1.29 ms | **3.1x slower** |
| Exponential | 0.26 ms | 0.87 ms | **3.3x slower** |
| Division | 0.19 ms | 0.83 ms | **4.3x slower** |
| MatMul (2K) | 54.82 ms | 60.44 ms | **1.1x slower** |

These operations are critical for attention mechanisms, explaining the severe performance degradation.

## Flash Attention Compatibility

### Flash Attention 3
- ✅ **H100/H800**: Full support, 1.5-2x faster than FA2
- ❌ **All others**: Not supported

### Flash Attention 2
- ✅ **Ampere and newer**: Full support
- ⚠️ **Turing/Volta**: Partial support
- ❌ **Pascal and older**: Not supported

### Fallback Behavior
```python
# Automatic fallback on older GPUs
if compute_capability < (8, 0):  # Pre-Ampere
    # Falls back to PyTorch native attention
    # Warning: "FlashAttention only supports Ampere GPUs or newer"
```

## Multi-GPU Scaling

### Communication Performance

Tested on 2x GTX 1080 with NCCL:
- Communication overhead: 0.2-0.6% (excellent)
- All-gather latency: ~1.89 ms for 4K sequence
- Bandwidth utilization: >90% of theoretical

### When Multi-GPU Helps

| Sequence Length | Single GPU | Multi-GPU | Recommendation |
|----------------|------------|-----------|----------------|
| < 16K tokens | Faster | Slower | Use single GPU |
| 16K-64K tokens | Limited | Faster | Consider multi-GPU |
| > 64K tokens | OOM | Works | Required |

## Memory Considerations

### Memory Usage by Precision

For 4K sequence, batch=1, 8 heads:
- **FP32**: 864 MB per GPU
- **FP16**: 676 MB per GPU (22% reduction)
- **FP8** (H100): ~340 MB per GPU (60% reduction)

### Memory Pooling Benefits

Tested impact:
- Pattern cache: 10-15% performance improvement
- Memory pool: Mixed results on Pascal
- Best on: Ampere and newer

## Optimization Strategies

### For Pascal (GTX 1080/1080 Ti)

```python
# Optimal settings for Pascal
from dilated_attention_pytorch import RingDilatedAttentionProduction

model = RingDilatedAttentionProduction(
    segment_lengths=[1024, 2048, 4096],
    dilation_rates=[1, 2, 4],
    dtype=torch.float32,  # ALWAYS use FP32
    use_pattern_cache=True,
    enable_memory_pool=False,  # Mixed results
)

# Avoid these on Pascal:
# - torch.cuda.amp.autocast()  # Makes it slower
# - model.half()  # Severe performance penalty
# - Mixed precision training  # Not beneficial
```

### For Modern GPUs (Ampere+)

```python
# Optimal settings for Ampere and newer
from dilated_attention_pytorch import RingDilatedAttentionProduction

model = RingDilatedAttentionProduction(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dtype=torch.float16,  # or torch.bfloat16
    use_pattern_cache=True,
    enable_memory_pool=True,
)

# Use AMP for training
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(q, k, v)
```

## Platform-Specific Notes

### NVIDIA H100
- Use FP8 when possible
- Enable Flash Attention 3
- Use BF16 over FP16
- Enable all optimizations

### NVIDIA A100
- Use BF16 for training stability
- Enable Flash Attention 2
- Use tensor core optimizations
- Memory pooling highly effective

### NVIDIA GTX 1080/1080 Ti
- **Critical**: Always use FP32
- Disable mixed precision
- Focus on memory efficiency
- Consider upgrading for production

### AMD GPUs
- Currently not well tested
- Theoretical support via ROCm
- Expect compatibility issues

## Performance Expectations

### Relative Performance (4K sequence, 8 heads)

| GPU | Precision | Time (ms) | Throughput | Memory |
|-----|-----------|-----------|------------|---------|
| H100 | FP8 | ~10 | 400K tok/s | 340 MB |
| A100 | FP16 | ~25 | 160K tok/s | 450 MB |
| V100 | FP16 | ~80 | 50K tok/s | 600 MB |
| 3090 | FP16 | ~40 | 100K tok/s | 500 MB |
| 1080 | FP32 | ~313 | 13K tok/s | 865 MB |
| 1080 | FP16 | ~695 | 5.9K tok/s | 676 MB |

## Troubleshooting

### "Flash Attention only supports Ampere GPUs or newer"
- Expected on Pascal/Volta/Turing
- Automatically falls back to native PyTorch
- No action needed

### FP16 Slower than FP32
- Check GPU architecture
- If Pascal: This is expected, use FP32
- If Ampere+: Check for driver issues

### OOM on Multi-GPU
- Reduce batch size
- Enable gradient checkpointing
- Use DeepSpeed ZeRO

## Recommendations Summary

1. **Production Deployment**: Use Ampere or newer
2. **Development**: Pascal acceptable with FP32
3. **Research**: H100 for maximum sequence lengths
4. **Budget Option**: Used V100 or T4

## Future Hardware Support

### Expected Improvements
- **Blackwell** (B100): Est. 2025, 5x H100 performance
- **Intel Gaudi**: Alternative to NVIDIA, being tested
- **AMD MI300**: Competitive with H100, ROCm support improving

### Deprecated Soon
- Pascal architecture support may be dropped in v0.4.0
- Maxwell already unsupported
- Kepler removed in v0.2.0

---

*Last updated: June 2025*
*Based on extensive benchmarking with dilated-attention-pytorch v0.2.x*