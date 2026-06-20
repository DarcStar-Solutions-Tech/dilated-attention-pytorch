# Benchmarks Directory

This directory contains performance benchmark results for various dilated attention implementations.

## Purpose
- Store benchmark results with timestamps
- Track performance improvements over time
- Compare different implementations
- Document configuration and hardware used

## Directory Structure
```
benchmarks/
├── latest/              # Most recent benchmark results
├── YYYY-MM-DD-*/        # Date-specific benchmark runs  
└── *.md/json/png        # Individual benchmark files
```

## File Types
- `.json` - Raw benchmark data (timings, configurations)
- `.md` - Human-readable benchmark reports
- `.png` - Performance visualization charts
- `.txt` - Additional benchmark outputs

## Naming Convention
Files follow the format: `{benchmark-type}-YYYY-MM-DD-HHMM-UTC.{ext}`

Examples:
- `ring-attention-comprehensive-2025-06-27-2150-UTC.md`
- `benchmark-long-sequences-2025-06-27-1231-UTC.png`

## Common Benchmark Types
- **all-implementations**: Comparison across all implementations
- **long-sequences**: Performance on long sequence lengths
- **extreme-sequences**: Stress testing with very long sequences
- **ring-attention**: Ring attention specific benchmarks
- **block-sparse**: Block-sparse pattern benchmarks
- **memory-pool**: Memory optimization benchmarks
- **hilbert**: Hilbert curve optimization benchmarks

## Latest Results
Check the `latest/` directory for the most recent benchmark results:
- `billion-token.md` - Billion token processing benchmarks
- `comprehensive.json/png` - All implementations comparison
- `long-sequences.json/png` - Long sequence performance
- `benchmark-all-implementations.json/png` - Detailed implementation comparison

## Hardware Configurations
Benchmarks are typically run on:
- NVIDIA GTX 1080 (Consumer GPU testing)
- NVIDIA A100 (Data center GPU testing)  
- NVIDIA H100 (Latest data center GPU)
- Multi-GPU setups (2-8 GPUs)

## Reading Benchmark Results
1. Check `latest/` for most recent results
2. Look for `.md` files for summaries
3. View `.png` files for visual comparisons
4. Use `.json` files for detailed analysis

## Running New Benchmarks
See the main project documentation for benchmark scripts in the `benchmarks/` source directory.

