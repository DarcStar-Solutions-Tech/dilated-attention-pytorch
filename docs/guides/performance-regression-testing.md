# Performance Regression Testing Guide

This guide explains how to use the performance regression test suite to track and prevent performance degradation in the dilated attention implementations.

## Overview

The performance regression test suite automatically tracks execution time and memory usage for each implementation across different configurations. It compares current performance against established baselines and alerts when performance degrades beyond acceptable thresholds.

## Key Features

- **Automated Performance Tracking**: Measures execution time, memory usage, and CUDA metrics
- **Baseline Management**: Stores and updates performance baselines
- **Regression Detection**: Alerts when performance degrades beyond threshold (default: 15%)
- **Historical Tracking**: Maintains performance history for trend analysis
- **Interactive Visualizations**: Generates charts showing performance over time
- **CI/CD Integration**: Automated testing in pull requests

## Running Performance Tests

### Quick Start

```bash
# Run all performance regression tests
python scripts/run_performance_regression.py

# Test specific implementation
python scripts/run_performance_regression.py --implementation DilatedAttention

# Generate report only (no tests)
python scripts/run_performance_regression.py --report-only
```

### Establishing Baselines

When running tests for the first time or after major changes:

```bash
# Establish baselines for all implementations
python scripts/establish_baselines.py

# Or manually delete baselines to reset
rm tests/performance_baselines/baselines.json
```

### Test Configurations

The suite tests multiple configurations:
- **Small**: 2,048 sequence length
- **Medium**: 4,096 sequence length  
- **Large**: 8,192 sequence length

Each configuration uses:
- Batch size: 1
- Number of heads: 8
- Head dimension: 64

## Understanding Results

### Performance Metrics

Each test measures:
- **Execution Time**: Average forward pass time in milliseconds
- **Memory Allocated**: GPU memory allocated in MB
- **Memory Reserved**: Total GPU memory reserved in MB
- **CUDA Time**: GPU kernel execution time (when available)

### Regression Detection

A test fails if performance degrades more than the threshold:
- Default threshold: 15% slower than baseline
- Negative percentages indicate performance improvements
- Test results show: ✓ (passed) or ✗ (failed)

### Example Output

```
DilatedAttention b1_s2048_h8_d64:
  Baseline: 1.04ms
  Current:  0.96ms
  Change:   -7.7%
  Memory:   16.0MB allocated
PASSED
```

## Visualizing Performance

Generate interactive performance charts:

```bash
# Create performance dashboard and comparison charts
python scripts/visualize_performance.py

# Specify output directory
python scripts/visualize_performance.py --output-dir reports/
```

This creates:
- `performance_dashboard.html`: Time-series charts of performance metrics
- `performance_comparison.html`: Side-by-side implementation comparison

## CI/CD Integration

The performance regression tests run automatically:
- On pull requests (when commit message contains `[perf]`)
- Nightly scheduled runs
- Results posted as PR comments

To trigger in a PR:
```bash
git commit -m "Optimize attention computation [perf]"
```

## Customizing Tests

### Modifying Configurations

Edit `CONFIGS` in `test_performance_regression.py`:

```python
CONFIGS = [
    (batch_size, seq_len, num_heads, head_dim),
    # Add more configurations as needed
]
```

### Changing Regression Threshold

Update `REGRESSION_THRESHOLD` in the test class:

```python
class TestPerformanceRegression:
    REGRESSION_THRESHOLD = 15.0  # Allow up to 15% degradation
```

Or via command line:
```bash
python scripts/run_performance_regression.py --threshold 20.0
```

### Adding New Implementations

1. Create a new test method in `TestPerformanceRegression`
2. Follow the naming pattern: `test_{implementation}_performance`
3. Use the same structure as existing tests
4. Run to establish baseline

## Best Practices

1. **Regular Testing**: Run performance tests before merging significant changes
2. **Update Baselines**: After intentional optimizations, update baselines
3. **Monitor Trends**: Review performance dashboard weekly
4. **Investigate Regressions**: Any failure should be investigated immediately
5. **Document Changes**: Note performance impacts in PR descriptions

## Troubleshooting

### Common Issues

**"No baseline found"**
- Normal for first run
- Test will establish baseline automatically

**"CUDA out of memory"**
- Reduce test configurations
- Close other GPU applications
- Use smaller batch sizes

**"Performance regression detected"**
- Review recent changes
- Check GPU thermal throttling
- Ensure consistent testing environment
- Compare with historical data

### Baseline Management

```bash
# View current baselines
cat tests/performance_baselines/baselines.json | python -m json.tool

# View performance history
cat tests/performance_baselines/history.json | python -m json.tool

# Reset specific baseline
# Edit baselines.json and remove the specific entry
```

## Performance Optimization Tips

When optimizing for performance:

1. **Profile First**: Use PyTorch profiler to identify bottlenecks
2. **Memory Efficiency**: Reduce allocations and reuse buffers
3. **Kernel Fusion**: Combine operations when possible
4. **Batch Operations**: Process multiple items together
5. **Hardware Utilization**: Optimize for specific GPU architectures

## Future Enhancements

Planned improvements:
- Multi-GPU performance testing
- Distributed training benchmarks
- Memory fragmentation analysis
- Automatic performance bisection
- Performance prediction models