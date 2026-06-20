# Immediate Benchmark Cleanup Plan

## Quick Wins (Can be done immediately)

### 1. Files to Delete (62 files)
These are clearly temporary debugging/investigation files that have served their purpose:

#### Debug Files (17 files)
```bash
rm debug_1024_issue.py
rm debug_4k_d8.py
rm debug_4k_triton_performance.py
rm debug_fix_issue.py
rm debug_hilbert_kernel.py
rm debug_hilbert_performance.py
rm debug_kernel_params.py
rm debug_optimized_shapes.py
rm debug_refactoring_performance.py
rm debug_remaining_issues.py
rm debug_sparse_output_mismatch.py
rm debug_sparse_performance.py
rm debug_sparse_regression.py
rm debug_sparse_triton_issue.py
rm debug_triton_overhead.py
rm debug_triton_performance.py
rm debug_unified_kernel.py
```

#### Investigation Files (7 files)
```bash
rm investigate_4k_d2_issue.py
rm investigate_4k_d4_anomaly.py
rm investigate_4k_issue.py
rm investigate_4k_regression.py
rm investigate_8k_performance_dip.py
rm investigate_sparse_performance.py
rm isolate_4k_issue.py
```

#### Temporary Test Files (20 files)
```bash
rm test_4k_fix.py
rm test_4k_fix_corrected.py
rm verify_4k_fix_final.py
rm verify_8k_fix.py
rm verify_fix_performance.py
rm verify_sparse_improvements.py
rm verify_sparse_improvements_fp32.py
rm verify_sparse_kernel_usage.py
rm confirm_normalization_bug.py
rm diagnose_hilbert_performance.py
rm diagnose_performance_issues.py
rm profile_implementation_overhead.py
rm test_sparse_performance_fix.py
rm test_sparse_hilbert_fix.py
rm test_integrated_sparse_opt.py
rm test_sparse_optimization_fixed.py
rm triton_fixes.py
rm simple_hilbert_benchmark.py
rm test_hilbert_256.py
rm check_gpu_info.py
```

#### Redundant Quick Files (5 files)
```bash
rm quick_hilbert_benchmark.py
rm quick_hilbert_performance_test.py
rm quick_performance_check.py
rm quick_performance_test.py
rm quick_sequence_limit_test.py
```

#### Obsolete Analysis Files (13 files)
```bash
rm analyze_hilbert_functions.py
rm analyze_hilbert_issue.py
rm analyze_kernel_redundancy.py
rm analyze_ordering_issue.py
rm analyze_position_processing.py
rm analyze_triton_issues.py
rm analyze_enhanced_wins.py
rm analyze_optimized_wins.py
rm analyze_fused_scaling.py
rm definitive_hilbert_test.py
rm sparse_pattern_analysis.py
rm gpu_info_simple.py
rm ring_attention_minimal.py
```

### 2. Keep and Organize (Recent Refactoring Work)

Create a new subdirectory for the recent refactoring benchmarks:

```bash
mkdir -p refactoring
mv analyze_lse_vs_softmax.py refactoring/
mv analyze_sparse_regression.py refactoring/
mv benchmark_after_fix.py refactoring/
mv benchmark_refactored_vs_enhanced.py refactoring/
mv compare_before_after_fix.py refactoring/
mv compare_sparse_configs.py refactoring/
mv final_performance_comparison.py refactoring/
mv focused_benchmark_enhanced.py refactoring/
mv quick_benchmark_after_fix.py refactoring/
mv quick_enhanced_comparison.py refactoring/
mv test_fused_softmax.py refactoring/
mv test_sparse_fix.py refactoring/
mv benchmark_enhanced_refactoring.py refactoring/
```

### 3. Main Benchmarks to Keep (Root Level)

These are the primary benchmarks that should remain at the root level:

```
benchmark_hilbert_attention.py          # Main Hilbert benchmark
benchmark_block_sparse_ring_attention.py # Block sparse benchmark
benchmark_block_sparse_ring_simple.py    # Simple block sparse
benchmark_all_hilbert_implementations.py # Compare all implementations
run_benchmark.py                        # Main runner
```

### 4. Documentation to Keep

```
README.md
CLEANUP_SUMMARY.md
BENCHMARK_DIRECTORY_ANALYSIS.md
IMMEDIATE_CLEANUP_PLAN.md (this file)
```

### 5. Summary Reports to Archive

Create an archive directory for historical reports:

```bash
mkdir -p archive/reports
mv *_summary.md archive/reports/
mv *_summary.png archive/reports/
mv *_analysis.md archive/reports/
mv *_results.md archive/reports/
mv corrected_implementation_benchmark_results.md archive/reports/
```

## Expected Results

- **Before**: 170+ files
- **After**: ~50 files (well-organized)
- **Deleted**: 62 temporary/debug files
- **Organized**: 13 refactoring benchmarks
- **Archived**: ~20 reports

## Benefits

1. **80% reduction in clutter**
2. **Clear organization of recent work**
3. **Preservation of important benchmarks**
4. **Easy to find relevant files**
5. **No loss of important functionality**

## Command to Execute Cleanup

```bash
# Run from benchmarks directory
# First, create directories
mkdir -p refactoring archive/reports

# Then execute the cleanup (copy and verify before running!)
# ... (all rm commands from above)
# ... (all mv commands from above)
```