# Benchmark Consolidation Plan

## Current State
- **122 total benchmark files**
- 15 core framework files (good foundation)
- 107 suite files with massive redundancy

## Categories and Consolidation Strategy

### 1. Core Framework (15 files) - KEEP AS IS
Already well-organized with:
- `base_benchmark.py` - Base classes
- `utils/` - Shared utilities
- `output_manager.py`, `storage.py` - Results handling

### 2. Basic Suites (10 files) → 2 files
Consolidate to:
- `benchmark_quick_validation.py` - Quick smoke tests
- `benchmark_basic_comparison.py` - Basic performance comparison

### 3. Distributed Suites (29 files) → 4 files
Consolidate to:
- `benchmark_distributed_basic.py` - Basic multi-GPU tests
- `benchmark_distributed_ring.py` - Ring attention specific
- `benchmark_distributed_scaling.py` - Scaling studies
- `benchmark_distributed_communication.py` - Communication patterns

### 4. Extreme Suites (15 files) → 3 files
Consolidate to:
- `benchmark_extreme_sequences.py` - Maximum sequence lengths
- `benchmark_extreme_memory.py` - Memory limits
- `benchmark_extreme_sparsity.py` - Extreme sparsity patterns

### 5. Specialized Suites (52 files) → 8 files
Consolidate to:
- `benchmark_implementation_comparison.py` - Compare all implementations
- `benchmark_block_sparse_variants.py` - All block sparse variations
- `benchmark_hilbert_optimization.py` - Hilbert curve specific
- `benchmark_memory_optimization.py` - Memory usage analysis
- `benchmark_flash_attention.py` - Flash attention integration
- `benchmark_precision_comparison.py` - FP16/FP32/BF16 comparison
- `benchmark_pattern_analysis.py` - Attention pattern analysis
- `benchmark_comprehensive_report.py` - Full report generation

## Redundancy Patterns to Eliminate

### 1. Duplicate Timing Functions
- `time_attention()`, `measure_time()`, `benchmark_attention()` → Use base class
- Custom CUDA timing → Use `utils/timing.py`

### 2. Duplicate Setup Code
- Device selection, dtype selection → Use base class
- Memory cleanup → Use `utils/memory.py`

### 3. Duplicate Result Formatting
- CSV/JSON/table output → Use `output_manager.py`
- Plot generation → Create unified plotting utility

### 4. Similar Test Configurations
- Many files test same sequence lengths/batch sizes
- Consolidate into configuration classes

## Implementation Plan

### Phase 1: Create Unified Benchmark Runner
- Extend `run_benchmark.py` to handle all benchmark types
- Add configuration file support (YAML/JSON)
- Implement result aggregation

### Phase 2: Create Consolidated Suites
- Start with basic suites (easiest)
- Move to specialized (most redundancy)
- Handle distributed carefully (preserve functionality)

### Phase 3: Archive Old Benchmarks
- Move to `benchmarks/archive/`
- Keep for reference during transition

### Phase 4: Documentation
- Create benchmark guide
- Document configuration options
- Add examples for common scenarios

## Expected Benefits
- **Reduction**: 122 files → ~20 files (>80% reduction)
- **Maintainability**: Easier to add new benchmarks
- **Consistency**: Same timing/memory/output methods
- **Flexibility**: Configuration-driven benchmarks
- **Performance**: Reuse setup/teardown code