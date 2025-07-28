# Project Structure

**Last Updated**: 2025-07-10

This document describes the organization of the dilated-attention-pytorch codebase after the comprehensive reorganization.

## Overview

The project contains:
- **69 Python source files** organized into logical subdirectories
- **25+ implementations** of dilated attention variants
- **Comprehensive test suite** with 525 tests mirroring source structure
- **Modern Python tooling** with uv, hatch, and poetry support

## Directory Structure

```
dilated-attention-pytorch/
│
├── src/                               # Source code (Hatch standard)
│   └── dilated_attention_pytorch/     # Main package
│       ├── __init__.py               # Package exports and aliases
│       │
│       ├── base/                     # Core implementations (6 files)
│       │   ├── __init__.py          
│       │   ├── dilated_attention.py # Core dilated attention
│       │   ├── multihead_dilated_attention.py # Multihead wrapper
│       │   ├── improved_dilated_attention.py # Enhanced version
│       │   ├── improved_multihead_dilated_attention.py # Enhanced multihead
│       │   ├── distributed_dilated_attention.py # Multi-GPU support
│       │   └── head_parallel_dilated_attention_optimized.py # Head-parallel
│       │
│       ├── ring/                     # Ring attention variants (O(n) memory)
│       │   ├── __init__.py          
│       │   ├── base/                # Base ring implementations (5 files)
│       │   │   ├── ring_dilated_attention_correct.py
│       │   │   ├── ring_dilated_attention_fixed_simple.py
│       │   │   ├── ring_dilated_attention_memory_efficient.py
│       │   │   ├── ring_dilated_attention_sdpa.py
│       │   │   └── ring_dilated_attention_v3.py
│       │   ├── distributed/         # Distributed ring attention (1 file)
│       │   │   └── ring_distributed_dilated_attention.py
│       │   ├── hilbert/            # Hilbert-optimized ring (8 files)
│       │   │   ├── ring_dilated_attention_hilbert_core.py
│       │   │   ├── ring_dilated_attention_hilbert_core_fixed.py
│       │   │   ├── ring_dilated_attention_hilbert_gpu_optimized.py
│       │   │   ├── ring_dilated_attention_hilbert_optimized_correct.py
│       │   │   ├── ring_dilated_attention_hilbert_optimized_fixed.py
│       │   │   ├── ring_dilated_attention_hilbert_optimized_fixed_v2.py
│       │   │   └── ring_dilated_attention_hilbert_proper.py
│       │   └── utils/              # Ring attention utilities (6 files)
│       │       ├── ring_attention_autograd.py
│       │       ├── ring_attention_fixed_deadlock.py
│       │       ├── ring_attention_lse.py
│       │       ├── ring_attention_memory_efficient.py
│       │       ├── ring_attention_utils.py
│       │       └── ring_attention_utils_fixed.py
│       │
│       ├── sparse/                  # Block-sparse implementations (11 files)
│       │   ├── __init__.py         
│       │   ├── block_sparse_ring_dilated_attention.py # Core block-sparse
│       │   ├── block_sparse_ring_dilated_attention_fixed.py # Fixed API
│       │   ├── block_sparse_ring_dilated_attention_hilbert_post_pattern.py
│       │   ├── block_sparse_ring_multihead_dilated_attention.py
│       │   ├── block_sparse_ring_distributed_dilated_attention.py
│       │   ├── block_sparse_adaptive.py # Content-adaptive patterns
│       │   ├── block_sparse_adaptive_fixed.py
│       │   ├── block_sparse_factory.py # Factory for sparse variants
│       │   ├── distributed_memory_optimization.py
│       │   ├── distributed_sparse_config.py
│       │   └── sparse_pattern_generator.py
│       │
│       ├── models/                  # Full models (2 files)
│       │   ├── __init__.py
│       │   ├── transformer.py      # Transformer with dilated attention
│       │   └── long_net.py         # Full LongNet architecture
│       │
│       ├── core/                    # Core infrastructure (17 files)
│       │   ├── __init__.py         
│       │   ├── base.py             # Base classes for all implementations
│       │   ├── config.py           # Type-safe configuration dataclasses
│       │   ├── constants.py        # Feature detection and constants
│       │   ├── factory.py          # Factory pattern for module creation
│       │   ├── memory_pool.py      # Unified memory pool
│       │   ├── standardized_api.py # Standardized API wrappers
│       │   ├── pattern_cache.py    # Pattern caching
│       │   ├── memory_profiler.py  # Memory profiling tools
│       │   ├── memory_visualizer.py # Memory visualization
│       │   └── (various memory pool implementations)
│       │
│       ├── utils/                   # Utility modules (11 files)
│       │   ├── __init__.py         
│       │   ├── validation.py       # Input validation utilities
│       │   ├── attention_utils.py  # Common attention operations
│       │   ├── sparse_pattern_utils.py # Sparse pattern generation
│       │   ├── hilbert_curve.py    # Hilbert curve utilities
│       │   ├── dynamic_segment_selector.py # Dynamic segment sizing
│       │   ├── flash_attention_utils.py # Flash Attention support
│       │   ├── flash_attention_3_utils.py # Flash Attention 3
│       │   ├── gpu_utils.py        # GPU utilities
│       │   ├── hilbert_attention_mixin.py # Hilbert mixin
│       │   └── return_standardizer.py # Output standardization
│       │
│       ├── kernels/                 # Custom kernels (3 files)
│       │   ├── __init__.py
│       │   ├── hilbert_attention_core.py # Hilbert kernel
│       │   ├── hilbert_attention_triton_wrapper.py # Triton wrapper
│       │   └── README.md
│       │
│       └── dynamic_dilated_attention.py # Dynamic segment sizing wrapper
│
├── tests/                           # Test suite (42 test files)
│   ├── __init__.py                 
│   ├── base/                       # Base implementation tests
│   │   ├── test_dilated_attention.py
│   │   ├── test_multihead_dilated_attention.py
│   │   ├── test_improved_dilated_attention.py
│   │   ├── test_improved_multihead.py
│   │   └── test_distributed_dilated_attention.py
│   ├── ring/                       # Ring attention tests
│   │   ├── test_ring_attention.py
│   │   ├── test_distributed_ring_attention.py
│   │   ├── test_ring_hybrid_implementations.py
│   │   └── hilbert/               # Hilbert-specific tests
│   │       ├── test_hilbert_gradient_comparison.py
│   │       ├── test_multigpu_hilbert_ring.py
│   │       └── test_per_segment_hilbert.py
│   ├── sparse/                     # Block-sparse tests
│   │   ├── test_block_sparse_attention.py
│   │   ├── test_block_sparse_adaptive.py
│   │   ├── test_block_sparse_ring_multihead.py
│   │   └── test_block_sparse_distributed_optimizations.py
│   ├── models/                     # Model tests
│   │   ├── test_long_net.py
│   │   └── test_transformer.py
│   ├── core/                       # Core infrastructure tests
│   │   ├── test_factory.py
│   │   ├── test_memory_pool.py
│   │   ├── test_core_refactoring.py
│   │   └── test_config_validation.py
│   ├── utils/                      # Utility tests
│   │   ├── test_validation.py
│   │   └── test_dynamic_segment_selection.py
│   ├── misc/                       # Miscellaneous tests
│   │   ├── test_edge_cases_validation.py
│   │   ├── test_thread_safety.py
│   │   ├── test_flash_attention_3.py
│   │   ├── test_memory_optimizations.py
│   │   └── test_memory_pool_consolidated.py
│   └── TEST_REDUNDANCY_ANALYSIS.md # Test cleanup documentation
│
├── scripts/                         # Utility scripts
│   ├── test_comprehensive.py       # Quick comprehensive test
│   ├── launch_distributed_training.py # Distributed launcher
│   ├── simple_distributed_test.py  # Simple distributed test
│   ├── create_dilated_attention_diagram.py # Visualization
│   ├── verify_block_sparse_fixes.py # Verification scripts
│   └── verify_block_sparse_merge.py
│
├── benchmarks/                      # Performance benchmarking
│   ├── core/                       # Shared benchmark infrastructure
│   │   ├── base_benchmark.py      # Base benchmark classes
│   │   └── utils/                 # Benchmark utilities
│   │       ├── distributed.py     # Distributed utilities
│   │       ├── memory.py          # Memory utilities
│   │       ├── timing.py          # Timing utilities
│   │       └── data.py            # Data generation
│   ├── suites/                     # Benchmark suites
│   │   ├── specialized/           # Specialized benchmarks
│   │   │   ├── benchmark_dynamic_segment_sizing.py
│   │   │   └── benchmark_aggressive_dynamic_sizing.py
│   │   └── (other benchmark suites)
│   ├── test_improved_suite.py     # Consolidated improved tests
│   ├── test_distributed_suite.py  # Consolidated distributed tests
│   ├── verify_all.py              # Comprehensive verification
│   ├── benchmark.py               # Main benchmark script
│   ├── benchmark_all.py           # Comprehensive benchmarks
│   ├── benchmark_ring_billion_tokens.py # Billion-token tests
│   └── benchmark_sequence_limits.py # Sequence limit testing
│
├── analysis/                        # Analysis scripts
│   ├── billion_token_analysis.py  # Scaling analysis
│   ├── ring_attention_analysis.py # Ring performance analysis
│   ├── ring_performance_analysis.py
│   ├── compare_implementations.py # Implementation comparison
│   ├── detailed_memory_analysis.py # Memory profiling
│   ├── memory_estimation.py       # Memory usage estimation
│   ├── multihead_memory_analysis.py
│   └── simple_comparison.py       # Simple comparisons
│
├── examples/                        # Example usage
│   ├── basic_dilated_attention.py # Basic examples
│   ├── distributed_training_example.py # Distributed training
│   ├── factory_pattern_example.py # Factory pattern usage
│   ├── dynamic_segment_example.py # Dynamic segment sizing
│   ├── simple_usage.py            # Simple usage examples
│   └── ring_attention/            # Ring attention examples
│
├── docs/                           # Documentation
│   ├── guides/                    # User guides
│   │   ├── implementation-overview.md
│   │   ├── ring-attention-guide.md
│   │   ├── block-sparse-attention-guide.md
│   │   ├── distributed-training-guide.md
│   │   ├── factory-pattern-guide.md
│   │   └── practical-usage-guide.md
│   ├── reports/                   # Technical reports
│   │   ├── test-cleanup-2025-07-09-1446-UTC.md
│   │   ├── test-verification-2025-07-09-1434-UTC.md
│   │   └── (various timestamped reports)
│   ├── benchmarks/                # Benchmark results
│   └── feasibility/               # Feasibility studies
│
├── # Root Configuration Files
├── pyproject.toml                 # Modern Python packaging
├── setup.py                       # Legacy packaging support
├── README.md                      # Project documentation
├── CLAUDE.md                      # AI assistant instructions
├── PROJECT_STRUCTURE.md           # This file
├── LICENSE                        # MIT license
├── .gitignore                     # Git ignore rules
├── .github/                       # GitHub workflows
└── validate_changes.py            # Component verification
```

## Key Features by Directory

### `/src/dilated_attention_pytorch/base/`
Core implementations with standard interfaces:
- Base dilated attention with configurable parameters
- Multihead wrappers with MAGNETO improvements
- Improved versions with Flash Attention support
- Distributed implementations for multi-GPU

### `/src/dilated_attention_pytorch/ring/`
Ring attention variants achieving O(n) memory complexity:
- Base implementations with different optimization strategies
- Distributed ring attention for multi-node scaling
- Hilbert-optimized variants for improved cache locality
- Comprehensive utilities for ring communication patterns

### `/src/dilated_attention_pytorch/sparse/`
Block-sparse implementations for 5-50x speedup:
- Core block-sparse with multiple pattern types
- Multihead and distributed variants
- Content-adaptive sparsity patterns
- Factory pattern for easy creation

### `/src/dilated_attention_pytorch/core/`
Infrastructure supporting all implementations:
- Base classes reducing code duplication by 50-60%
- Type-safe configuration with validation
- Unified memory pool management
- Factory pattern for module creation
- Standardized API wrappers

### `/src/dilated_attention_pytorch/utils/`
Common utilities shared across implementations:
- Attention computation helpers
- Flash Attention 3 support
- Sparse pattern generation
- Validation and GPU utilities
- Dynamic segment selection

## Implementation Categories

### 1. **Standard Implementations** (6)
Basic dilated attention with configurable parameters in `base/`

### 2. **Ring Attention** (20+)
O(n) memory complexity for billion-token sequences in `ring/`

### 3. **Block-Sparse** (11)
5-50x speedup through sparsity in `sparse/`

### 4. **Full Models** (2)
Complete architectures in `models/`

### 5. **Infrastructure** (17+)
Supporting utilities and base classes in `core/`

## Recent Changes (2025-07-09)

### Reorganization
- Moved all implementations into logical subdirectories
- Created `base/`, `ring/`, `sparse/`, `models/` directories
- Organized tests to mirror source structure
- Consolidated 5 redundant test files (32 tests removed)
- Moved 12 non-test files to appropriate directories

### Improvements
- Clear separation of concerns
- Easier navigation and maintenance
- Consistent import paths
- Better discoverability of implementations

## Import Examples

All implementations are available through the package's `__init__.py`:

```python
# Direct imports (backward compatible)
from dilated_attention_pytorch import DilatedAttention, MultiheadDilatedAttention

# Factory pattern (recommended)
from dilated_attention_pytorch import create_multihead_dilated_attention

# Full path imports
from dilated_attention_pytorch.base.dilated_attention import DilatedAttention
from dilated_attention_pytorch.ring.distributed.ring_distributed_dilated_attention import RingDistributedDilatedAttention
from dilated_attention_pytorch.sparse.block_sparse_adaptive import BlockSparseAdaptive
```

## Development

The project uses modern Python tooling:
- **Hatch**: Environment and dependency management
- **uv**: Fast package installation (10-100x faster than pip)
- **pytest**: Testing with 525 tests
- **ruff**: Linting and formatting

See [CLAUDE.md](CLAUDE.md) for detailed development instructions.