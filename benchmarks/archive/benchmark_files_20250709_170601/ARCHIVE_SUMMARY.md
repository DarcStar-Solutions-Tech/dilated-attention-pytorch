# Benchmark Archive - 20250709_170601

## Reason for Archive

These benchmark files were archived after consolidation into:

1. **Core Framework** (`benchmarks/core/`)
   - `base_benchmark.py` - Base classes
   - `config.py` - Configuration system
   - `unified_runner.py` - Unified benchmark runner
   - `utils/` - Shared utilities

2. **Consolidated Suites** (`benchmarks/suites/consolidated/`)
   - `benchmark_basic_comparison.py` - Basic performance comparison
   - `benchmark_extreme_sequences.py` - Extreme sequence lengths
   - `benchmark_block_sparse.py` - Block sparse variants
   - (more to come)

## Statistics

- Files archived: 103
- Files kept: 25
- Reduction: ~80%

## Benefits

- Eliminated duplicate timing/memory/setup code
- Unified configuration system
- Consistent output formatting
- Easier to add new benchmarks
- Better maintainability
