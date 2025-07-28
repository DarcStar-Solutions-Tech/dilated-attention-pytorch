# Dilated vs Standard Ring Attention Benchmark Report

**World Size**: 1 GPU(s)
**Device**: cuda:0
**Dtype**: torch.float32

## Key Differences

- **StandardRingAttention**: Computes full attention matrix, uses ring communication for O(n/k) memory
- **RingDilatedAttentionSDPA**: Computes sparse dilated patterns, further reduces computation

## Performance Summary

| Sequence Length | Standard Throughput | Dilated Throughput | Speedup | Sparsity |
|-----------------|--------------------|--------------------|---------|----------|
| 2,048 | 211,573 | 523,423 | 2.47x | 0.0% |
| 4,096 | 112,708 | 881,500 | 7.82x | 37.5% |
