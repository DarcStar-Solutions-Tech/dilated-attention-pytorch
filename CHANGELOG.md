# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (January 2025)
- **Standardized Ring Attention**: Consolidated 12+ ring variants into 4 high-quality implementations
  - `StandardRingAttention`: Base ring attention with true O(n/k) memory scaling
  - `DistributedRingAttention`: Multi-GPU with DeepSpeed ZeRO integration
  - `HilbertRingAttention`: Cache-optimized with Hilbert space-filling curves
  - `BlockSparseRingAttention`: Combines ring communication with block sparsity
  - All use efficient isend/irecv (no all_gather operations)
  - Factory pattern support via `create_ring_attention()`
  - Type-safe `RingAttentionConfig` for configuration
  - Top-level exports for easy access

### Changed (January 2025)  
- **Renamed Misleading Classes**:
  - `RingDistributedDilatedAttention` → `EnterpriseDistributedDilatedAttention`
  - Class doesn't implement ring attention, uses O(n) memory per GPU
  - Old name still available with deprecation warning

### Removed (January 2025)
- **Redundant Ring Implementations** (all used problematic all_gather):
  - `ring_dilated_attention_v2_collective.py`
  - `ring_dilated_attention_refactored.py`
  - `ring_hilbert_dilated_attention.py`
  - `ring_dilated_attention_fixed.py`
  - `improved_distributed_dilated_attention.py`
  - `block_sparse_ring_dilated_attention_original.py`
  - `head_parallel_dilated_attention.py`
  - `improved_distributed_dilated_attention.py`
  - `ring_multihead_dilated_attention.py`

### Removed
- **RingDilatedAttentionProduction**: Not actually ring attention (July 2025)
  - Despite its name, it computed full O(n²) attention matrices
  - Failed at 16K sequence length when true ring attention handles 1M+
  - See `docs/reports/ring-production-not-ring-attention-2025-07-08-0327-UTC.md`
  - Use `RingDistributedDilatedAttention` for true distributed ring attention

### Added
- **Block-Sparse Consolidation**: Merged redundant implementations
  - Enhanced `BlockSparseRingDilatedAttention` with optimizations from `block_sparse_optimized.py`
  - Added `PersistentPatternCache` for device-aware pattern caching with LRU eviction
  - Added batched block operations for efficiency (threshold: 32 blocks)
  - Added smart buffer reuse strategies
  - All block-sparse implementations now extend the enhanced base class

### Removed
- **BlockSparseOptimized**: Merged into `BlockSparseRingDilatedAttention`
  - All optimizations preserved in base implementation
  - Use `BlockSparseRingDilatedAttention` directly
- **BlockSparseTorchSparse**: Removed as it provided no benefits
  - Did not actually use PyTorch sparse tensors
  - Sequential processing made it slower than base implementation
  - Use `BlockSparseRingDilatedAttention` instead

### Changed
- **Block-Sparse Class Hierarchy**: Updated inheritance
  - `BlockSparseHierarchical` now extends `BlockSparseRingDilatedAttention`
  - `BlockSparseAdaptive` now extends `BlockSparseRingDilatedAttention`
  - All specialized implementations now inherit optimizations from base

### Added
- **Hilbert Curve Integration**: Added optimized Hilbert curve ordering for improved cache locality
  - `RingDilatedAttentionHilbertOptimized`: Production-ready implementation with Hilbert curve reordering
  - `utils/hilbert_curve.py`: Fast Hilbert curve computation utilities
  - Comprehensive benchmarks showing 15-30% performance improvement
- **Benchmark Suite Refactoring**: Eliminated ~60% code duplication
  - Created shared utilities in `benchmarks/core/`
  - `base_benchmark.py`: Base classes for consistent benchmarking
  - `utils/distributed.py`: Distributed computing utilities
  - `utils/memory.py`: Memory management and profiling
  - `utils/timing.py`: CUDA-aware timing utilities
  - `utils/data.py`: Standard data generation

### Removed
- **Deprecated all_gather Implementations**: Removed poorly performing classes
  - `head_parallel_dilated_attention.py` - Used inefficient all_gather
  - `improved_distributed_dilated_attention.py` - Poor scalability with all_gather
  - `ring_dilated_attention_v2_collective.py` - O(n²) communication complexity
  - `ring_hilbert_dilated_attention.py` - Replaced with optimized version
  - `ring_multihead_dilated_attention.py` - Depended on deprecated V2Collective
  - Use `RingDilatedAttentionProduction` or `RingDistributedDilatedAttention` instead
- **Redundant Benchmark Files**: Removed 140+ duplicate benchmark scripts
  - Consolidated into organized test suites
  - `test_improved_suite.py`: All improved implementation tests
  - `test_distributed_suite.py`: All distributed tests
  - `verify_all.py`: Comprehensive verification

### Changed
- **Documentation Updates**: Updated all guides to reference non-deprecated classes
  - `CLAUDE.md`: Removed deprecated class references, added benchmark info
  - `README.md`: Added benchmark infrastructure documentation
  - `docs/guides/ring-attention-migration.md`: Complete rewrite for new implementations
  - `docs/guides/optimization-guide.md`: Updated class references
  - `docs/guides/hardware-compatibility-guide.md`: Updated examples
  - `docs/guides/horovod-integration-guide.md`: Updated integration examples

### Added
- **RingMultiheadDilatedAttention**: Proper multihead wrapper for Ring Attention
  - Drop-in replacement for nn.MultiheadAttention with O(n) memory scaling
  - Supports MAGNETO LayerNorm and all Ring Attention optimizations
  - Compatible with factory pattern
- **Pattern Caching**: Global pattern cache for Ring Attention implementations
  - 2x speedup for repeated forward passes
  - 23% memory reduction through CPU storage
  - Enabled by default for all Ring Attention variants
- **Smart GPU Detection**: Automatic dtype selection based on GPU architecture
  - Pascal GPUs (GTX 10-series): Automatically use FP32 for 10x better performance
  - Modern GPUs (RTX 20-series+): Use FP16/BF16 for optimal performance
  - No user configuration needed - fully automatic
- **Flash Attention Integration**: Intelligent backend selection with fallback
  - Flash Attention 3 support (when available)
  - Automatic fallback: FA3 → FA2 → FA1 → xformers → SDPA → standard
  - GPU architecture-aware optimization
- **Enhanced Memory Pool**: Adaptive memory management
  - 16MB threshold for optimal performance
  - 15-30% reduction in peak memory usage
  - LRU cache with 50 buffer entries
  - Memory-pinned allocations for faster transfers
- **Benchmark Infrastructure**: Reorganized and comprehensive
  - `benchmarks/core/`: Core benchmark suite
  - `benchmarks/specialized/`: Specialized benchmarks
  - Consolidated Ring Attention benchmarks
  - Comprehensive README with usage guide
- **Documentation**: 
  - Pascal GPU FP16 performance analysis
  - OOM analysis and mitigation strategies
  - Ring Attention optimization reports
  - Memory pool integration guide

### Removed
- **ImprovedDilatedAttentionV2**: Removed experimental V2 implementation
  - Performance was unpredictable (slower in 75% of cases)
  - AttentionBufferManager added 3.5x overhead vs direct allocation
  - Users should use ImprovedDilatedAttention instead
  - Removed related test files and buffer manager module
- **RingDilatedAttentionV2**: Removed deprecated implementation
  - Had distributed communication issues with isend/irecv
  - Users should use RingDilatedAttention (alias for V2Collective) instead
- **Educational Implementations**: Moved to examples directory
  - TrueRingDilatedAttention → examples/ring_attention/reference_implementation.py
  - SimulatedRingDilatedAttention → examples/ring_attention/single_gpu_simulation.py

### Changed
- **Ring Attention V2 Optimizations**:
  - Pattern caching enabled by default
  - Memory pool enabled with 16MB threshold
  - Smart dtype selection integrated
  - Flash Attention/xformers backend support
- **Performance Improvements**:
  - Pascal GPUs: 10.14x speedup with automatic FP32 selection
  - All GPUs: 2x speedup from pattern caching
  - Memory efficiency: 15-30% reduction in peak usage
  - Backend optimization: Up to 4.74x speedup with xformers
- **CI/CD Improvements**:
  - Dependabot configuration for automated dependency updates
  - GitHub issue templates (bug report, feature request)
  - Pull request template with checklist
  - Code of Conduct
  - Contributing guidelines
  - `.gitattributes` for consistent line endings
  - `py.typed` file for PEP 561 compliance
  - GPU testing workflow with cloud GPU support
  - Code coverage reporting with Codecov integration
  - Coverage HTML report generation as artifacts
  - `.codecov.yml` configuration for coverage thresholds
- **Development Tooling**:
  - Pre-commit hooks with Ruff and mypy
  - Ruff replaces Black, isort, and flake8
  - Enhanced Ruff rules for better code quality
  - Stricter mypy configuration
- **Python Support**: Minimum Python version raised to 3.13
- **Repository URLs**: Updated to DarcStar-Solutions-Tech organization

### Fixed
- **Pascal GPU Performance**: FP16 operations now automatically use FP32
  - Fixes 8x performance regression on GTX 10-series GPUs
  - Inheritance issue resolved in RingDilatedAttentionV2Collective
- **Memory Issues**: OOM errors reduced through optimized memory pooling
- **Import Errors**: Fixed distributed testing module imports

### Deprecated
- **RingDilatedAttentionV3**: Deprecated in favor of V2
  - V3's complex caching adds 15-45% overhead without benefits
  - V2 provides better performance with simpler implementation
  - Will be removed in v0.3.0

### Removed
- Python 3.9, 3.10, 3.11, and 3.12 support
- Poetry configuration files (using Hatch)
- `setup.cfg` (moved all configuration to `pyproject.toml`)
- Redundant benchmark scripts (consolidated into organized structure)

## [0.2.0] - 2025-01-25

### Added
- **Core Architecture Refactoring**: New `core` module with shared base classes and utilities
  - `BaseDilatedAttention` and `BaseMultiheadDilatedAttention` abstract base classes
  - Type-safe configuration system with dataclasses
  - Unified memory pool for efficient buffer management
  - Factory pattern for easy module creation with auto-selection
- **Factory Pattern**: New factory functions for creating attention modules
  - `create_dilated_attention()` and `create_multihead_dilated_attention()`
  - `create_block_sparse_attention()` and `create_adaptive_sparse_attention()`
  - Auto-selection of optimal implementation based on hardware
- **Utility Organization**: Moved utility functions to dedicated `utils` directory
  - `validation.py`: Input validation utilities
  - `attention_utils.py`: Common attention operations
  - `sparse_pattern_utils.py`: Sparse pattern generation and optimization
- **Documentation**: Comprehensive documentation updates
  - Migration guide for upgrading from v0.1.x
  - Factory pattern usage guide
  - Updated README with v0.2.0 features
  - Updated all examples to show factory pattern usage

### Changed
- **Code Reduction**: 50-60% reduction in code duplication through shared implementations
- **Import Structure**: All implementations now inherit from core base classes
- **Memory Management**: Consolidated memory pools across all implementations
- **Configuration**: Moved from individual parameters to type-safe config dataclasses

### Fixed
- Import path errors in utility modules
- Thread safety issues in sparse pattern generation
- Shape mismatches in attention utilities
- Memory leaks in distributed implementations
- Various flake8 linting issues

### Deprecated
- Direct instantiation of attention classes (still supported for backward compatibility)
- Old `DistributedMultiheadDilatedAttention` import (commented out)

## [0.1.0] - 2024-XX-XX

### Added
- Initial implementation of DilatedAttention from LongNet paper
- MultiheadDilatedAttention with MAGNETO improvements
- ImprovedDilatedAttention with Flash Attention support
- Ring attention implementations for O(n) memory scaling
- Block-sparse ring attention for 5-50x speedup
- Distributed training support
- Comprehensive test suite
- Benchmarking scripts
- Example usage scripts

[0.2.0]: https://github.com/fkodom/dilated-attention-pytorch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fkodom/dilated-attention-pytorch/releases/tag/v0.1.0