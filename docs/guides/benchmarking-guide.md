# Benchmarking Guide

This guide provides an overview of the benchmarking scripts available for evaluating dilated attention implementations.

## Quick Start

For a quick performance validation (< 1 minute):
```bash
python benchmarks/benchmark_quick_validation.py
```

For comprehensive sequence length benchmarking:
```bash
python benchmarks/benchmark_sequence_ranges.py --range production
```

For optimization impact analysis:
```bash
python benchmarks/benchmark_optimization_impact.py
```

## Available Benchmark Scripts

### 1. Quick Validation (`benchmark_quick_validation.py`)

**Purpose**: Rapid performance testing and functionality validation

**Features**:
- Runs in under 1 minute
- Tests tiny (1K), small (4K), and medium (16K) sequences
- Validates output correctness
- Suitable for CI/CD pipelines

**Usage**:
```bash
# Basic validation
python benchmarks/benchmark_quick_validation.py

# Test specific implementations
python benchmarks/benchmark_quick_validation.py --implementations dilated improved ring_v2

# Save results to custom location
python benchmarks/benchmark_quick_validation.py --output results/validation.json
```

### 2. Sequence Range Benchmarking (`benchmark_sequence_ranges.py`)

**Purpose**: Systematic benchmarking across different sequence length ranges

**Features**:
- Predefined ranges: production (1K-16K), medium (32K-128K), long (256K-1M)
- Real-world lengths: documents, context windows, book-length texts
- Automatic batch size adjustment
- Comprehensive performance comparison

**Usage**:
```bash
# Test production sequence lengths (1K-16K)
python benchmarks/benchmark_sequence_ranges.py --range production

# Test all predefined ranges
python benchmarks/benchmark_sequence_ranges.py --range all

# Test custom sequence lengths
python benchmarks/benchmark_sequence_ranges.py --range custom --custom-lengths 2048 4096 8192

# Enable optimizations
python benchmarks/benchmark_sequence_ranges.py --enable-pattern-cache --enable-memory-pool
```

**Available Ranges**:
- `production`: 1K, 2K, 4K, 8K, 12K, 16K tokens
- `medium`: 32K, 49K, 64K, 98K, 128K tokens
- `long`: 256K, 384K, 512K, 768K, 1M tokens
- `document`: 512, 1K, 2K, 4K tokens (typical documents)
- `context`: 4K, 8K, 16K, 32K tokens (LLM context windows)
- `book`: 50K, 100K, 200K tokens (long-form content)

### 3. Optimization Impact Analysis (`benchmark_optimization_impact.py`)

**Purpose**: Measure the impact of pattern caching and memory pooling

**Features**:
- Tests each optimization individually and combined
- Measures speedup and memory reduction
- Tracks cache hit rates and pool statistics
- Generates visualization plots

**Usage**:
```bash
# Basic optimization impact analysis
python benchmarks/benchmark_optimization_impact.py

# Test specific implementations
python benchmarks/benchmark_optimization_impact.py --implementations ring_v2 ring_v3

# Custom sequence lengths
python benchmarks/benchmark_optimization_impact.py --sequence-lengths 8192 16384 32768 65536
```

**Metrics Collected**:
- Baseline performance (no optimizations)
- Pattern cache speedup and hit rate
- Memory pool speedup and allocation statistics
- Combined optimization benefits
- Memory usage reduction percentages

### 4. Other Specialized Benchmarks

#### Extreme Sequences (`benchmark_extreme_sequences.py`)
Tests implementation limits with very long sequences (up to 1M+ tokens).

#### Ring Billion Tokens (`benchmark_ring_billion_tokens.py`)
Approaches billion-token sequences using Ring Attention.

#### Multi-GPU Benchmarks
Test distributed training scenarios across multiple GPUs.

#### Block Sparse Benchmarks
Evaluate sparsity patterns and their performance impact.

## Benchmark Output

All benchmarks generate:
1. **Markdown Report** - Human-readable results and analysis
2. **JSON Data** - Machine-readable raw results
3. **Visualizations** - Performance plots (when applicable)

Output location: `benchmark_results/` (default)

## Performance Tips

### For Accurate Results:
1. **GPU Warmup**: Scripts include warmup steps
2. **Clean Environment**: Close other GPU applications
3. **Consistent Hardware**: Use same GPU for comparisons
4. **Multiple Runs**: Consider averaging multiple runs

### Memory Considerations:
- Batch size automatically adjusts based on sequence length
- Use memory pooling for sequences > 32K tokens
- Enable pattern caching for repeated configurations

## Interpreting Results

### Key Metrics:
- **Time (ms)**: Forward pass latency
- **Memory (MB)**: Peak GPU memory usage
- **Throughput (tokens/sec)**: Processing speed
- **Speedup**: Relative performance vs baseline

### What to Look For:
1. **Crossover Points**: Where advanced implementations become beneficial
2. **Memory Efficiency**: Reduction in peak memory usage
3. **Scalability**: How performance changes with sequence length
4. **Optimization Impact**: Benefits from caching and pooling

## Common Use Cases

### Development Workflow:
```bash
# Quick check after code changes
python benchmarks/benchmark_quick_validation.py

# Detailed performance analysis
python benchmarks/benchmark_sequence_ranges.py --range production
```

### Production Evaluation:
```bash
# Test with production-like settings
python benchmarks/benchmark_sequence_ranges.py \
    --range context \
    --implementations improved ring_v2 \
    --enable-pattern-cache \
    --enable-memory-pool
```

### Optimization Tuning:
```bash
# Measure optimization benefits
python benchmarks/benchmark_optimization_impact.py \
    --sequence-lengths 4096 8192 16384 32768 65536
```

## Troubleshooting

### Out of Memory Errors:
- Reduce batch size with `--batch-size 1`
- Test shorter sequences first
- Enable memory pooling

### Slow Performance:
- Ensure GPU is available and being used
- Check for thermal throttling
- Close other GPU applications

### Validation Failures:
- Check sequence length divisibility by segment lengths
- Verify implementation compatibility
- Review error messages in output

## Adding Custom Benchmarks

To add your own benchmark:

1. Import required modules:
```python
from dilated_attention_pytorch import YourImplementation
from benchmarks.benchmark_utils import BenchmarkOutputManager
```

2. Follow the pattern of existing benchmarks:
- Use dataclasses for results
- Include warmup steps
- Handle errors gracefully
- Generate both human and machine-readable output

3. Consider adding to `benchmark_all.py` for inclusion in comprehensive tests.