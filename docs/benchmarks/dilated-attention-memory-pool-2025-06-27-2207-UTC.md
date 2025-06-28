# DilatedAttention Memory Pool Integration Benchmark

Generated: 2025-06-27T22:07:53.942480Z

## Configuration

- Device: cuda
- Batch Size: 2
- Sequence Length: 16384
- Num Heads: 8
- Head Dim: 64
- Iterations: 10
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

| Configuration | Time per iteration | Peak Memory | Time Improvement | Memory Improvement |
|---------------|-------------------|-------------|------------------|--------------------||
| Without Pool | 0.2438s | 612.0MB | - | - |
| Full Pool | 0.1997s | 612.0MB | 18.1% | 0.0% |
| Lightweight Pool | 0.2120s | 612.0MB | 13.0% | 0.0% |

## Key Findings

- ✅ **Performance improvement**: 13.0% faster with lightweight pool
- ⚠️ **Memory overhead**: 0.0% more memory usage

### Memory Pool Features:
- Enhanced memory pool integration with DilatedAttention core
- Configurable pool strategies (full vs lightweight)
- Automatic strategy selection for tensor allocation
- Optional memory profiling and monitoring
- Based on lessons learned from ImprovedDilatedAttention
