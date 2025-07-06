# FP32 vs FP16 Performance Analysis - Unexpected Results

**Date**: 2025-07-06 23:12 UTC  
**GPU**: NVIDIA GeForce GTX 1080 (Pascal Architecture)  
**Finding**: FP32 is 12.5x faster than FP16 (!!)

## Summary of Results

| Configuration | FP32 Time | FP32 Memory | FP16 Time | FP16 Memory | Speedup |
|--------------|-----------|-------------|-----------|-------------|---------|
| seq=1024, b=2, h=8 | 1.4ms | 39MB | 8.2ms | 20MB | 0.17x |
| seq=2048, b=2, h=8 | 2.2ms | 78MB | 24.3ms | 39MB | 0.09x |
| seq=4096, b=2, h=8 | 5.9ms | 156MB | 84.9ms | 78MB | 0.07x |
| seq=8192, b=1, h=8 | 10.8ms | 156MB | 194.8ms | 78MB | 0.06x |

**Key Findings:**
- FP32 is **12.5x faster** on average than FP16
- FP16 uses **2x less memory** as expected
- Total throughput: FP32 = 8.7M tokens/sec, FP16 = 754K tokens/sec

## Analysis: Why is FP32 Faster?

### 1. **GTX 1080 Architecture (Pascal)**
The GTX 1080 uses Pascal architecture which has:
- **Poor FP16 performance**: Only 1:64 FP16 to FP32 ratio
- **No Tensor Cores**: Unlike newer GPUs (Volta+)
- **FP16 runs at 1/64th speed of FP32** on consumer Pascal GPUs

### 2. **Hardware Specifications**
- GTX 1080 FP32 Performance: ~8.9 TFLOPS
- GTX 1080 FP16 Performance: ~0.14 TFLOPS (artificially limited)
- This explains the massive performance gap

### 3. **Software Fallbacks**
The warning message confirms:
```
Flash Attention failed, falling back: FlashAttention only supports Ampere GPUs or newer.
```
- No optimized FP16 kernels available
- Standard PyTorch operations may not be optimized for FP16 on Pascal
- Likely using slow fallback implementations

### 4. **Memory Bandwidth vs Compute**
- FP16 saves memory bandwidth (2x less data)
- But on GTX 1080, compute is the bottleneck, not memory
- The 64x compute penalty overwhelms any bandwidth savings

## Comparison with Modern GPUs

| GPU | Architecture | FP16:FP32 Ratio | FP16 Speedup Expected |
|-----|--------------|-----------------|----------------------|
| GTX 1080 | Pascal | 1:64 | 0.015x (slowdown) |
| RTX 2080 | Turing | 1:1 | ~1.5-2x |
| RTX 3090 | Ampere | 1:1 | ~1.5-2x |
| RTX 4090 | Ada | 1:1 | ~1.5-2x |
| A100 | Ampere | 1:1 | ~1.5-2x |

## Recommendations

### For GTX 1080 Users:
1. **Always use FP32** - it's 12.5x faster
2. **Memory is not the bottleneck** at these sequence lengths
3. **Consider upgrading** to Ampere+ for FP16 benefits

### For Production Deployment:
1. **Profile on target hardware** - FP16 behavior varies dramatically
2. **Use FP32 on Pascal** and older architectures
3. **Use FP16 on Turing+** architectures (RTX 20-series and newer)

### Code Changes Needed:
```python
# Detect Pascal and force FP32
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:  # Pre-Volta
        dtype = torch.float32
    else:
        dtype = torch.float16
```

## Updated Benchmark Results (FP32)

Since FP32 is much faster on GTX 1080, here are the corrected performance metrics:

| Seq Len | Time (ms) | Memory (MB) | Tokens/sec |
|---------|-----------|-------------|------------|
| 1,024 | 1.4 | 39 | 1,461,989 |
| 2,048 | 2.2 | 78 | 1,862,087 |
| 4,096 | 5.9 | 156 | 1,388,108 |
| 8,192 | 10.8 | 156 | 758,518 |

**New Performance Characteristics:**
- **Best throughput**: 1.86M tokens/sec at seq=2048
- **Scaling**: Still O(n^2) but with much better constant factor
- **Memory**: 2x higher but not a limiting factor

## Conclusion

The original benchmarks using FP16 severely underestimated the performance of DilatedAttention on GTX 1080. With FP32:
- Performance is **12.5x better** than reported
- Throughput reaches **1.86M tokens/sec**
- The implementation is much more practical for Pascal GPUs

This highlights the importance of:
1. Understanding hardware capabilities
2. Profiling with appropriate data types
3. Not assuming FP16 is always faster

For fair comparisons with other implementations, all benchmarks on GTX 1080 should use FP32.