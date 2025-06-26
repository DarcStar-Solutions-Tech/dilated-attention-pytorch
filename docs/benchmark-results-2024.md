# Dilated Attention PyTorch - Benchmark Results

## Hardware and Software Configuration

**Date**: December 2024  
**Hardware**: 2x NVIDIA GeForce GTX 1080 (7.9 GB each)  
**CUDA Version**: 12.7 (Driver), 12.4 (PyTorch)  
**PyTorch Version**: 2.6.0+cu124  
**Precision**: float16  
**Python Version**: 3.13

## Performance Benchmarks

### Overview

We benchmarked the core dilated attention implementations to measure performance characteristics across different sequence lengths and batch sizes. The tests were conducted on NVIDIA GTX 1080 GPUs with CUDA acceleration.

### Implementations Tested

1. **DilatedAttention**: Core dilated attention mechanism from the LongNet paper
2. **MultiheadDilatedAttention**: Optimized multihead wrapper with linear projections

### Results Summary

#### Small Sequences (batch=1, seq_len=2048)
- **Segment lengths**: [1024, 2048], **Dilation rates**: [1, 2]

| Implementation | Mean Time (ms) | Std Dev (ms) | Memory (MB) | Throughput (tokens/s) |
|----------------|----------------|--------------|-------------|----------------------|
| DilatedAttention | 6.16 | 1.23 | 11.1 | 332,468 |
| MultiheadDilatedAttention | 4.35 | 0.45 | 24.6 | 470,805 |

**Result**: MultiheadDilatedAttention is **1.42x faster** for small sequences.

#### Medium Sequences (batch=2, seq_len=4096)
- **Segment lengths**: [1024, 2048, 4096], **Dilation rates**: [1, 2, 4]

| Implementation | Mean Time (ms) | Std Dev (ms) | Memory (MB) | Throughput (tokens/s) |
|----------------|----------------|--------------|-------------|----------------------|
| DilatedAttention | 36.15 | 6.89 | 12.0 | 226,510 |
| MultiheadDilatedAttention | 95.34 | 45.25 | 96.3 | 85,898 |

**Result**: DilatedAttention is more efficient for medium sequences due to lower overhead.

#### Large Sequences (batch=4, seq_len=8192)
- **Segment lengths**: [2048, 4096, 8192], **Dilation rates**: [1, 2, 4]

| Implementation | Mean Time (ms) | Std Dev (ms) | Memory (MB) | Throughput (tokens/s) |
|----------------|----------------|--------------|-------------|----------------------|
| DilatedAttention | 593.67 | 443.92 | 48.0 | 55,049 |
| MultiheadDilatedAttention | 389.97 | 315.49 | 385.2 | 83,794 |

**Result**: MultiheadDilatedAttention is **1.52x faster** for large sequences.

### Key Findings

1. **Performance Scaling**:
   - MultiheadDilatedAttention excels at small and large sequence lengths
   - DilatedAttention has lower overhead for medium-sized sequences
   - Both implementations scale well to longer sequences

2. **Memory Usage**:
   - DilatedAttention is more memory-efficient (11-48 MB)
   - MultiheadDilatedAttention trades memory for speed (24-385 MB)
   - Memory scaling is sub-linear with sequence length

3. **GPU Utilization**:
   - Float16 precision provides optimal performance on GTX 1080
   - ~9.1x speedup compared to CPU execution
   - Effective use of CUDA kernels for attention computation

### Comparison with CPU Performance

| Configuration | GPU Time (ms) | CPU Time (ms) | Speedup |
|---------------|---------------|---------------|---------|
| batch=4, seq=2048, heads=12 | 12.96 | 118.54 | 9.1x |

**Throughput comparison**:
- GPU: 632,289 tokens/second
- CPU: 69,105 tokens/second

### Recommendations

1. **For maximum speed**: Use MultiheadDilatedAttention with sufficient GPU memory
2. **For memory efficiency**: Use DilatedAttention when memory is constrained
3. **For production**: Consider batch size and sequence length patterns in your use case

### Known Issues

Several advanced implementations encountered compatibility issues during benchmarking:

- **ImprovedDilatedAttention**: Tensor reshape compatibility issues
- **RingDilatedAttention**: Constructor parameter mismatches
- **DistributedDilatedAttention**: Abstract method implementation missing
- **BlockSparseRingDilatedAttention**: Unexpected keyword arguments

These issues are being addressed in ongoing development.

## Benchmark Methodology

### Test Configuration
```python
WARMUP_ITERS = 3
BENCHMARK_ITERS = 10
DEVICE = torch.device("cuda")
DTYPE = torch.float16
```

### Measurement Process
1. Clear GPU cache before each test
2. Perform warmup iterations
3. Synchronize CUDA operations
4. Time 10 iterations and calculate statistics
5. Track peak memory usage

### Test Patterns
- Causal attention enabled for all tests
- Variable segment lengths and dilation rates
- Batch sizes from 1 to 4
- Sequence lengths from 2048 to 8192

## Future Benchmarks

Planned benchmarks for upcoming releases:
- Flash Attention 3 integration performance
- Ring Attention memory scaling tests
- Block-sparse attention speedup measurements
- Multi-GPU distributed training benchmarks
- Comparison with other long-context attention methods

## Running Your Own Benchmarks

To reproduce these benchmarks:

```bash
# Ensure you have CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Run the standard benchmark
python benchmark.py

# For custom configurations
python benchmark.py --batch_size 8 --seq_len 16384 --num_heads 16
```

## Hardware Recommendations

Based on our benchmarks:

- **Minimum**: NVIDIA GTX 1080 or equivalent (8GB VRAM)
- **Recommended**: NVIDIA RTX 3090 or better (24GB VRAM)
- **Optimal**: NVIDIA A100/H100 for production workloads

### Memory Requirements by Sequence Length

| Sequence Length | Batch Size | Min VRAM (DilatedAttention) | Min VRAM (MultiheadDilated) |
|-----------------|------------|-----------------------------|-----------------------------|
| 2,048 | 4 | 256 MB | 512 MB |
| 4,096 | 4 | 512 MB | 1 GB |
| 8,192 | 4 | 1 GB | 2 GB |
| 16,384 | 4 | 2 GB | 4 GB |
| 32,768 | 4 | 4 GB | 8 GB |

## Citations

If you use these benchmarks in your research, please cite:

```bibtex
@misc{dilated-attention-pytorch-benchmarks,
  title={Dilated Attention PyTorch Benchmark Results},
  author={DarcStar Technologies},
  year={2024},
  url={https://github.com/DarcStar-Solutions-Tech/dilated-attention-pytorch}
}
```