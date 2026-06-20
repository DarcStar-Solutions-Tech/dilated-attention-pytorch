# Tests Directory Cleanup Plan

## Immediate Actions

### 1. Organize Root Level Files (9 files)
Move files from root to appropriate subdirectories:

```bash
# Move kernel-related tests
mv test_consolidated_kernel.py kernels/
mv test_enhanced_refactoring.py kernels/
mv test_hilbert_attention_simplified.py kernels/
mv test_kernel_consolidation.py kernels/
mv test_simplified_correctness.py kernels/
mv test_triton_hilbert.py kernels/
mv test_unified_hilbert_attention.py kernels/
mv test_unified_hilbert_basic.py kernels/

# Move utility test
mv test_cache_management.py utils/
```

### 2. Remove Empty Directory
```bash
rmdir analysis/
```

### 3. Check for Duplicate Tests
Based on TEST_REDUNDANCY_ANALYSIS.md, investigate:
- Performance regression tests (keep only the "_all" version)
- Hilbert gradient tests (consolidate 3 into 1)
- Flash Attention tests (consider merging)

## Detailed Cleanup Tasks

### Phase 1: Organization (Immediate)
1. Move 9 root-level files to subdirectories
2. Remove empty `analysis/` directory
3. Update any imports affected by moves

### Phase 2: Consolidation (Next)
1. **Performance Regression Tests**
   - Check if both exist: `test_performance_regression.py` and `test_performance_regression_all.py`
   - Keep only the "_all" version if both exist

2. **Hilbert Gradient Tests**
   - Review these files in `ring/hilbert/`:
     - test_hilbert_gradient_comparison.py
     - test_hilbert_gradient_simple.py (if exists)
     - test_hilbert_backward_pass.py
   - Consolidate into single comprehensive test

3. **Flash Attention Tests**
   - Review: test_flash_attention_3.py vs test_flash_attention_integration.py
   - Consider if they serve different purposes or can be merged

### Phase 3: Recovery Assessment (Optional)
The 26 test files from benchmarks that were deleted might include actual tests worth recovering:
- test_hilbert_correctness.py
- test_sequence_limits.py
- test_sparse_patterns.py
- etc.

Consider reviewing git history to identify which were genuine tests vs temporary scripts.

## Expected Results

### Before
- 57 Python files
- 10 files at root level
- Empty directory
- Some duplicate tests

### After Phase 1
- ~57 files (better organized)
- 1 file at root level (just __init__.py)
- No empty directories
- Clear subdirectory organization

### After Phase 2
- ~50-52 files (consolidated)
- No duplicate test coverage
- Cleaner test suite

## Benefits

1. **Better Organization**: All tests in logical subdirectories
2. **No Redundancy**: Single source of truth for each test
3. **Easier Navigation**: Clear where to find/add tests
4. **Faster Test Runs**: No duplicate coverage

## Commands Summary

```bash
# Phase 1: Organization
cd tests/

# Move kernel tests
mv test_consolidated_kernel.py test_enhanced_refactoring.py \
   test_hilbert_attention_simplified.py test_kernel_consolidation.py \
   test_simplified_correctness.py test_triton_hilbert.py \
   test_unified_hilbert_attention.py test_unified_hilbert_basic.py \
   kernels/

# Move utility test
mv test_cache_management.py utils/

# Remove empty directory
rmdir analysis/

# Phase 2: Check for duplicates (example)
# Check performance regression tests
ls misc/test_performance_regression*.py

# Check Hilbert gradient tests
ls ring/hilbert/test_hilbert_gradient*.py
ls ring/hilbert/test_hilbert_backward*.py
```

This plan provides a structured approach to cleaning up the tests directory while being more conservative than the benchmarks cleanup, ensuring no actual tests are lost.