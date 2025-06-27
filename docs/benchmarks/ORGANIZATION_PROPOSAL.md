# Benchmark Organization Proposal

## Current Issues
- 21+ files already in a flat directory structure
- Difficult to find specific benchmark results
- No clear versioning or categorization
- Mixing of different benchmark types

## Proposed Structure

```
docs/benchmarks/
├── README.md                    # Index of all benchmarks with descriptions
├── latest/                      # Symlinks to most recent results of each type
│   ├── comprehensive.json
│   ├── long-sequences.json
│   ├── distributed.json
│   └── regression.json
├── by-date/                     # Organized by date
│   ├── 2025-06/
│   │   ├── 26/
│   │   │   ├── comprehensive/
│   │   │   ├── long-sequences/
│   │   │   └── distributed/
│   │   └── 27/
│   │       ├── comprehensive/
│   │       ├── long-sequences/
│   │       └── distributed/
├── by-type/                     # Organized by benchmark type
│   ├── comprehensive/
│   │   ├── 2025-06-26-1729-UTC/
│   │   └── 2025-06-27-0618-UTC/
│   ├── long-sequences/
│   │   ├── 2025-06-27-0638-UTC/
│   │   └── 2025-06-27-0653-UTC/
│   ├── distributed/
│   └── regression/
├── comparisons/                 # Special comparison reports
│   └── phase1-bugfix-2025-06-26/
└── archive/                     # Old results (>30 days)
```

## Implementation Plan

1. **Automatic Organization Script** (`scripts/organize_benchmarks.py`):
   ```python
   # Automatically organize files based on:
   # - Type detection from filename
   # - Date extraction from timestamp
   # - Create symlinks for latest results
   ```

2. **Benchmark Runner Updates**:
   - Modify benchmark scripts to save directly to organized structure
   - Auto-generate index entries in README.md
   - Create "latest" symlinks after each run

3. **Benchmark Types**:
   - `comprehensive`: Full implementation comparison
   - `long-sequences`: 32K+ token benchmarks  
   - `distributed`: Multi-GPU benchmarks
   - `regression`: Performance regression tests
   - `memory`: Memory usage analysis
   - `ablation`: Feature ablation studies

4. **Metadata Standards**:
   ```json
   {
     "benchmark_type": "long-sequences",
     "timestamp": "2025-06-27-0653-UTC",
     "git_commit": "9ac5433",
     "hardware": {
       "gpu": "GTX 1080",
       "memory_gb": 8
     },
     "parameters": {
       "sequences": [32768, 65536],
       "implementations": ["RingDilatedAttention", ...]
     }
   }
   ```

5. **Cleanup Policy**:
   - Keep all results for 30 days
   - Archive older results (compressed)
   - Always keep "milestone" benchmarks

## Benefits

1. **Easy Navigation**: Find benchmarks by date, type, or browse latest
2. **Comparison**: Easy to compare results across dates
3. **Storage Efficient**: Archive old results
4. **CI/CD Friendly**: Latest symlinks for automated checks
5. **Documentation**: Auto-generated index with descriptions

## Migration

1. Create organization script
2. Run migration on existing files
3. Update all benchmark scripts
4. Document new structure in README