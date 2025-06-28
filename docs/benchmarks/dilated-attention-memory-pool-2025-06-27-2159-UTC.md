# DilatedAttention Memory Pool Integration Benchmark

Generated: 2025-06-27T21:59:51.818675Z

## Configuration

- Device: cuda
- Batch Size: 1
- Sequence Length: 32768
- Num Heads: 8
- Head Dim: 64
- Iterations: 5
- PyTorch Version: 2.7.1+cu126

## Performance Comparison

| Configuration | Time per iteration | Peak Memory | Time Improvement | Memory Improvement |
|---------------|-------------------|-------------|------------------|--------------------||
| Without Pool | 0.0202s | 452.0MB | - | - |
| Full Pool | 0.0211s | 452.0MB | -4.5% | 0.0% |
| Lightweight Pool | 0.0294s | 452.0MB | -45.7% | 0.0% |

## Key Findings

- ⚠️ **Performance cost**: 45.7% slower with lightweight pool
- ⚠️ **Memory overhead**: 0.0% more memory usage

### Memory Pool Features:
- Enhanced memory pool integration with DilatedAttention core
- Configurable pool strategies (full vs lightweight)
- Automatic strategy selection for tensor allocation
- Optional memory profiling and monitoring
- Based on lessons learned from ImprovedDilatedAttention
