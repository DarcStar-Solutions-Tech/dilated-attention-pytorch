# Memory Pool Optimization Benchmark - All Implementations

Generated: 2025-06-27T22:16:57.488603Z

## Configuration

- Device: cuda
- Batch Size: 2
- Sequence Length: 16384
- Num Heads: 8
- Head Dim: 64
- Iterations: 10
- Segment Lengths: [2048, 4096, 8192]
- Dilation Rates: [1, 2, 4]
- PyTorch Version: 2.7.1+cu126

## Results Summary

| Implementation | Without Pool | With Pool | Time Improvement | Memory Improvement |
|----------------|--------------|-----------|------------------|--------------------||
| DilatedAttention | 0.0562s / 548MB | 0.0957s / 1188MB | -70.2% | -116.8% |
| ImprovedDilatedAttention | 0.0853s / 1156MB | 0.0331s / 1836MB | +61.2% | -58.8% |
| RingDilatedAttentionV2 | Error | Error | - | - |

## Key Findings

### Optimizations Applied:
1. **1MB threshold**: Only use memory pool for tensors ≥ 1MB
2. **Disabled by default**: Memory pools are opt-in, not default
3. **Smart allocation**: Avoid pool overhead for small temporary tensors
4. **Fixed SDPA warning**: Using `torch.is_grad_enabled()` instead of `.training`

### Performance Analysis:

**DilatedAttention**:
- ⚠️ Time: 70.2% slower with pool
- ⚠️ Memory: 116.8% increase with pool

**ImprovedDilatedAttention**:
- ✅ Time: 61.2% faster with pool
- ⚠️ Memory: 58.8% increase with pool
