# Documentation Cleanup Plan

## Date: January 30, 2025

## Phase 1: Quick Wins (Immediate)

### 1. Remove Empty Directories
```bash
rm -rf benchmarks/archive/
rm -rf benchmarks/exports/
rm -rf benchmarks/comparisons/
rm -rf benchmarks/by-type/attention-comparison/2025-06-27-1559-UTC/
```

### 2. Clean Benchmark Organization Redundancy
**Decision**: Keep only the original date-based structure, remove derivative organizations

**To Remove**:
- `benchmarks/by-date/` (entire directory tree)
- `benchmarks/by-type/` (entire directory tree)
- `benchmarks/latest/` (if using copies instead of symlinks)

**To Keep**:
- `benchmarks/YYYY-MM-DD-HHMM-UTC/` (original benchmark results)

**Estimated Reduction**: ~75 files

## Phase 2: Archive Review (Careful Review Required)

### Review June 2025 Archive Files
The archive/ directory contains 69 files from June 2025. Need to determine:
1. Which are truly obsolete (delete)
2. Which are historical reference (keep)
3. Which were mistakenly archived (restore)

**Suggested Approach**:
- Group by topic
- Compare with current docs
- Keep only if historically significant

## Phase 3: Reports Consolidation

### Merge Duplicate Reports
**Pattern**: Keep latest version, archive or remove older versions

Examples to consolidate:
1. **Sparse Optimization** (2 versions)
2. **Memory Pool Integration** (2 versions)  
3. **Implementation Analysis** (2 versions)
4. **Block Sparse Analysis** (multiple versions)

**Estimated Reduction**: ~50-75 files

### Remove Iteration/Debug Reports
Look for patterns like:
- `*-debug-*`
- `*-test-*`
- `*-temp-*`
- `*-comparison-v1/v2/v3*`

## Phase 4: Version-Specific Cleanup

### Review v2/v3 References
31 files reference deprecated implementations:
- If implementation removed → remove docs
- If implementation exists → keep docs
- Update references in remaining docs

## Phase 5: Structure Improvement

### Add Navigation Aids
1. **Create INDEX.md in each directory**
   ```markdown
   # [Directory Name] Documentation
   
   ## Purpose
   [Explain what goes here]
   
   ## Contents
   - [List key documents]
   
   ## Naming Convention
   [Explain naming rules]
   ```

2. **Add .gitignore rules**
   ```gitignore
   # Prevent documentation bloat
   docs/**/*-temp-*
   docs/**/*-debug-*
   docs/**/*.tmp
   docs/**/*.bak
   ```

3. **Create docs/README.md**
   - Overall structure explanation
   - Where to put new docs
   - Lifecycle rules

## Execution Commands

### Phase 1 (Safe to run immediately):
```bash
# Remove empty directories
find docs -type d -empty -delete

# Remove benchmark redundancy (after verification)
rm -rf docs/benchmarks/by-date/
rm -rf docs/benchmarks/by-type/
rm -rf docs/benchmarks/latest/  # Only if not using symlinks
```

### Phase 2-3 (Requires review):
```bash
# List candidates for removal
find docs/archive -name "*2025-06*" -type f
find docs/reports -name "*-debug-*" -o -name "*-temp-*"

# Group duplicate reports
find docs/reports -name "*.md" | sed 's/-[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-[0-9]\{4\}-UTC//' | sort | uniq -d
```

## Expected Outcome

### Metrics
- **Before**: 514 files, confusing structure
- **After Phase 1**: ~440 files (-75 files)
- **After Phase 2-3**: ~350 files (-90 more)
- **After Phase 4-5**: ~300 files with clear structure

### Benefits
1. 40% reduction in file count
2. Single organizational scheme
3. Clear navigation structure
4. Prevented future bloat
5. Easier to find documentation

## Risk Mitigation
1. Create full backup before starting
2. Use git for all deletions (can restore)
3. Review each phase before proceeding
4. Keep historically significant docs
5. Update any references to moved/deleted files