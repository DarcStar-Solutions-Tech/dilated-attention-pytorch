# Memory Pool Integration - Realistic Scenarios Benchmark

Generated: 2025-06-27T21:07:56.075511Z

## Configuration

- Device: cuda
- PyTorch Version: 2.7.1+cu126

## Training Simulation Results

- Total Steps: 250
- Time per Step: without_pool=0.0915s, with_pool=0.1296s
- Peak Memory: without_pool=311.3MB, with_pool=326.3MB
- **Time Improvement**: -41.6%
- **Memory Improvement**: -4.8%

## Key Findings

- ⚠️ **Training Performance**: 41.6% slower with pool overhead

### Memory Pool Benefits:
- Enhanced tensor allocation with strategy selection
- Automatic memory pool management and reuse
- NUMA-aware allocation for multi-socket systems
- Fragment-aware allocation to reduce fragmentation
- Optional memory profiling and monitoring
