# Memory Pool Optimization Benchmark - All Implementations

Generated: 2025-06-27T22:17:17.297334Z

## Configuration

- Device: cuda
- Batch Size: 1
- Sequence Length: 8192
- Num Heads: 8
- Head Dim: 64
- Iterations: 20
- Segment Lengths: [2048, 4096, 8192]
- Dilation Rates: [1, 2, 4]
- PyTorch Version: 2.7.1+cu126

## Results Summary

| Implementation | Without Pool | With Pool | Time Improvement | Memory Improvement |
|----------------|--------------|-----------|------------------|--------------------||
| DilatedAttention | 0.0089s / 139MB | 0.0174s / 284MB | -95.7% | -104.3% |
| ImprovedDilatedAttention | 0.0442s / 276MB | 0.0118s / 606MB | +73.4% | -119.6% |
| RingDilatedAttentionV2 | 0.0853s / 722MB | 0.1153s / 981MB | -35.2% | -35.8% |

## Key Findings

### Optimizations Applied:
1. **1MB threshold**: Only use memory pool for tensors ≥ 1MB
2. **Disabled by default**: Memory pools are opt-in, not default
3. **Smart allocation**: Avoid pool overhead for small temporary tensors
4. **Fixed SDPA warning**: Using `torch.is_grad_enabled()` instead of `.training`

### Performance Analysis:

**DilatedAttention**:
- ⚠️ Time: 95.7% slower with pool
- ⚠️ Memory: 104.3% increase with pool

**ImprovedDilatedAttention**:
- ✅ Time: 73.4% faster with pool
- ⚠️ Memory: 119.6% increase with pool

**RingDilatedAttentionV2**:
- ⚠️ Time: 35.2% slower with pool
- ⚠️ Memory: 35.8% increase with pool
