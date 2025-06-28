# DilatedAttention Memory Pool Integration Benchmark

Generated: 2025-06-27T21:30:54.906476Z

## Configuration

- Device: cuda
- Batch Size: 2
- Sequence Length: 8192
- Num Heads: 8
- Head Dim: 64
- Iterations: 20
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

| Configuration | Time per iteration | Peak Memory | Time Improvement | Memory Improvement |
|---------------|-------------------|-------------|------------------|--------------------||
| Without Pool | 0.0133s | 226.0MB | - | - |
| Full Pool | 0.0221s | 226.0MB | -66.3% | 0.0% |
| Lightweight Pool | 0.0896s | 226.0MB | -573.2% | 0.0% |

## Key Findings

- ⚠️ **Performance cost**: 573.2% slower with lightweight pool
- ⚠️ **Memory overhead**: 0.0% more memory usage

### Memory Pool Features:
- Enhanced memory pool integration with DilatedAttention core
- Configurable pool strategies (full vs lightweight)
- Automatic strategy selection for tensor allocation
- Optional memory profiling and monitoring
- Based on lessons learned from ImprovedDilatedAttention
