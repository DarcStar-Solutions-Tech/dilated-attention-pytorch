# RingDilatedAttentionV2Collective Multi-GPU Benchmark Suite

This benchmark suite validates the distributed functionality of `RingDilatedAttentionV2Collective` and measures performance scaling across different GPU configurations.

## Overview

The benchmark tests:
- **Single vs Multi-GPU Performance**: Compares performance with 1, 2, 4, and 8 GPUs
- **Scaling Efficiency**: Measures how well performance scales with additional GPUs
- **Communication Overhead**: Quantifies the cost of distributed communication
- **Memory Usage**: Tracks memory consumption across configurations
- **Throughput**: Measures tokens processed per second

## Quick Start

### 1. Test Installation

First, verify the setup works:

```bash
# Single GPU test
python scripts/test_ring_v2_collective.py

# Multi-GPU test (2 GPUs)
torchrun --nproc_per_node=2 scripts/test_ring_v2_collective.py
```

### 2. Run Benchmarks

Use the launch script for easy benchmarking:

```bash
# Run with default settings (1 GPU)
./scripts/launch_ring_v2_benchmark.sh

# Run with 4 GPUs
./scripts/launch_ring_v2_benchmark.sh 4

# Run all GPU configurations
./scripts/launch_ring_v2_benchmark.sh --all

# Quick test with smaller sequences
./scripts/launch_ring_v2_benchmark.sh --quick

# Test with very large sequences
./scripts/launch_ring_v2_benchmark.sh --large 4
```

### 3. Direct Benchmark Execution

For more control, run the benchmark directly:

```bash
# Single GPU
python benchmarks/specialized/benchmark_ring_v2_collective_distributed.py \
    --seq_lengths 16384 32768 65536 \
    --ring_sizes 1 2 4 \
    --batch_size 2

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 \
    benchmarks/specialized/benchmark_ring_v2_collective_distributed.py \
    --seq_lengths 16384 32768 65536 131072 \
    --ring_sizes 1 2 4 \
    --batch_size 2
```

## Benchmark Parameters

### Command Line Arguments

- `--batch_size`: Batch size for benchmarking (default: 2)
- `--seq_lengths`: Space-separated list of sequence lengths (default: 16384 32768 65536)
- `--embed_dim`: Embedding dimension (default: 768)
- `--num_heads`: Number of attention heads (default: 12)
- `--segment_lengths`: Segment lengths for dilated attention (default: 2048 4096 8192)
- `--dilation_rates`: Dilation rates (default: 1 2 4)
- `--ring_sizes`: Ring sizes to test (default: 1 2 4 8)
- `--warmup_runs`: Number of warmup iterations (default: 3)
- `--benchmark_runs`: Number of benchmark iterations (default: 10)
- `--no_flash_attention`: Disable Flash Attention
- `--dtype`: Data type - float16, bfloat16, float32 (default: float16)
- `--profile`: Enable profiling for the first configuration
- `--output_dir`: Directory for saving results (default: benchmark_results)
- `--no_save`: Don't save results to file

### Environment Variables (for launch script)

- `BATCH_SIZE`: Override default batch size
- `SEQ_LENGTHS`: Override sequence lengths (space-separated)
- `NUM_HEADS`: Override number of heads
- `EMBED_DIM`: Override embedding dimension
- `WARMUP_RUNS`: Override warmup runs
- `BENCHMARK_RUNS`: Override benchmark runs
- `OUTPUT_DIR`: Override output directory

## Analyzing Results

After running benchmarks, analyze the results:

```bash
# Run analysis on benchmark results
python analysis/ring_v2_collective_scaling_analysis.py \
    --results_dir benchmark_results/ring_v2_collective

# The analysis generates:
# - scaling_efficiency_analysis.png: Throughput, speedup, and efficiency plots
# - memory_analysis.png: Memory usage and efficiency plots
# - performance_heatmaps.png: Performance visualization across configurations
# - scaling_analysis_report.txt: Comprehensive text report
```

## Expected Results

### Scaling Efficiency

With proper setup, you should see:
- **2 GPUs**: 1.5-1.8x speedup (75-90% efficiency)
- **4 GPUs**: 2.8-3.5x speedup (70-87% efficiency)
- **8 GPUs**: 5.0-6.5x speedup (62-81% efficiency)

### Communication Overhead

- Typically 5-15% of total runtime for well-balanced workloads
- Increases with ring size but offset by parallelism benefits
- Lower with larger sequence lengths (better computation/communication ratio)

### Memory Usage

- Ring attention provides O(n) memory complexity
- Memory per GPU decreases with ring size
- Enables processing of sequences that don't fit on a single GPU

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or sequence length
   - Increase ring size to distribute memory load
   - Use `--dtype bfloat16` or `--dtype float32` if needed

2. **Distributed Process Group Errors**
   - Ensure all GPUs are visible: `nvidia-smi`
   - Check NCCL environment: `export NCCL_DEBUG=INFO`
   - Verify network connectivity between nodes (if multi-node)

3. **Performance Lower Than Expected**
   - Check GPU utilization: `nvidia-smi dmon -s pucvmet`
   - Ensure Flash Attention is enabled
   - Verify no other processes are using GPUs

### Debug Mode

For detailed debugging:

```bash
# Enable NCCL debug output
export NCCL_DEBUG=INFO

# Run with single sequence length for debugging
torchrun --nproc_per_node=2 \
    benchmarks/specialized/benchmark_ring_v2_collective_distributed.py \
    --seq_lengths 8192 \
    --warmup_runs 1 \
    --benchmark_runs 1
```

## Performance Tips

1. **Optimal Ring Size**
   - Use ring_size = world_size for maximum parallelism
   - Smaller ring sizes can be beneficial for very long sequences

2. **Sequence Length Considerations**
   - Longer sequences have better computation/communication ratio
   - Ensure sequence length is divisible by (ring_size × largest_segment_length)

3. **Hardware Considerations**
   - Best performance with NVLink or high-bandwidth interconnect
   - PCIe-only systems will have higher communication overhead

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "config": {
    "batch_size": 2,
    "seq_lengths": [16384, 32768],
    "embed_dim": 768,
    "num_heads": 12,
    ...
  },
  "world_size": 4,
  "timestamp": "20231215_143052",
  "results": [
    {
      "seq_length": 16384,
      "ring_size": 4,
      "world_size": 4,
      "rank": 0,
      "forward_time_ms": 15.23,
      "backward_time_ms": 28.45,
      "total_time_ms": 43.68,
      "memory_allocated_gb": 2.34,
      "memory_reserved_gb": 3.12,
      "throughput_tokens_per_sec": 1500234,
      "effective_batch_size": 2,
      "distributed_overhead_ms": 2.1,
      "communication_time_ms": 1.8
    },
    ...
  ]
}
```

## Integration with CI/CD

For automated testing:

```bash
# Quick validation
./scripts/launch_ring_v2_benchmark.sh --quick 2

# Comprehensive benchmark (nightly)
./scripts/launch_ring_v2_benchmark.sh --all

# Parse results for CI
python -c "
import json
with open('benchmark_results/ring_v2_collective/latest.json') as f:
    data = json.load(f)
    # Check scaling efficiency
    for result in data['results']:
        if result['ring_size'] > 1:
            efficiency = (result['throughput_tokens_per_sec'] / 
                         (result['ring_size'] * baseline_throughput))
            assert efficiency > 0.7, f'Poor scaling: {efficiency:.2%}'
"
```