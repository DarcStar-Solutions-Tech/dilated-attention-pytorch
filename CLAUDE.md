# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an unofficial PyTorch implementation of DilatedAttention from the LongNet paper "LongNet: Scaling Transformers to 1,000,000,000 Tokens". The project provides efficient attention mechanisms for handling very long sequences.

## Core Architecture

The project contains **21 active dilated attention implementations** organized into several categories.

### Main Components

- **DilatedAttention** (`dilated_attention_pytorch/base/dilated_attention.py`): Core dilated attention mechanism that supports variable segment lengths and dilation rates
- **MultiheadDilatedAttention** (`dilated_attention_pytorch/base/multihead_dilated_attention.py`): Drop-in replacement for nn.MultiheadAttention with dilated attention and MAGNETO improvements
- **ImprovedDilatedAttention** (`dilated_attention_pytorch/base/improved_dilated_attention.py`): Enhanced version with additional optimizations
- **ImprovedMultiheadDilatedAttention** (`dilated_attention_pytorch/base/improved_multihead_dilated_attention.py`): Enhanced multihead version with further optimizations
- **RingDilatedAttentionProduction** (`dilated_attention_pytorch/ring/hilbert/ring_dilated_attention_hilbert_gpu_optimized.py`): Production-ready ring attention with O(n) memory complexity and advanced error recovery
- **RingDistributedDilatedAttention** (`dilated_attention_pytorch/ring/distributed/ring_distributed_dilated_attention.py`): Enterprise-grade distributed implementation with DeepSpeed integration
- **LongNet** (`dilated_attention_pytorch/models/long_net.py`): Full transformer architecture for language modeling
- **Transformer** (`dilated_attention_pytorch/models/transformer.py`): General transformer with dilated attention

### Key Parameters

All dilated attention modules require:
- `segment_lengths`: Geometric sequence (e.g., [2048, 4096, 8192])
- `dilation_rates`: Corresponding dilation rates (e.g., [1, 2, 4])
- Sequence length must be divisible by the largest segment length

## Development Commands

### Testing and Verification
```bash
# Single GPU tests
hatch run test                         # Run all tests with coverage
pytest tests/test_dilated_attention.py # Run specific test file
pytest tests/ -v                       # Verbose output

# Multi-GPU tests (MUST use torchrun)
torchrun --nproc_per_node=2 tests/test_ring_attention.py
torchrun --nproc_per_node=4 tests/test_distributed_ring_attention.py

# Quick verification scripts
python scripts/test_comprehensive.py   # Quick comprehensive test
python verify_all_components.py        # Component verification

# Coverage reporting
pytest tests/ --cov=dilated_attention_pytorch --cov-report=html
```

### Project Tooling

This project uses a modern Python toolchain:
- **Hatch**: Environment management and task runner
- **uv**: Fast dependency installation (replaces pip)
- **torchrun**: Required for multi-GPU execution

### Environment Management with Hatch

```bash
# Enter the development environment
hatch shell

# Create/recreate environments
hatch env create                       # Create default environment
hatch env create test                  # Create test environment
hatch env create benchmark             # Create benchmark environment

# Run commands in specific environments
hatch run test                         # Run tests with coverage
hatch run lint                         # Run linting (ruff)
hatch run format                       # Format code (ruff)
hatch run typecheck                    # Type checking (mypy)
hatch run all                          # Run all checks

# Benchmark environment commands
hatch run benchmark:run                # Run benchmarks
hatch run benchmark:profile            # Run with profiling
```

### Dependency Management with uv

```bash
# ALWAYS use uv for installing dependencies (not pip)
uv pip install -e .                    # Install package
uv pip install -e .[dev]               # Install with dev dependencies
uv pip install -e .[test]              # Install with test dependencies
uv pip install -e .[benchmark]         # Install with benchmark dependencies
uv pip install -e .[distributed]       # Install with distributed training dependencies
uv pip install -e .[all]               # Install all optional dependencies

# Add new dependencies
uv pip install <package>               # Install a new package

# Why uv?
# - 10-100x faster than pip
# - Better resolver for complex dependencies
# - Automatic cleanup of unused packages
```

### Multi-GPU Execution with torchrun

When running any script that uses multiple GPUs, you MUST use `torchrun`:

```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=2 benchmarks/test_ring_attention.py
torchrun --nproc_per_node=4 scripts/train_model.py

# Multi-node execution
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=29500 train.py

# Common torchrun options:
# --nproc_per_node: Number of GPUs per node
# --nnodes: Total number of nodes
# --node_rank: Rank of this node (0-based)
# --master_addr: IP address of rank 0 node
# --master_port: Port for communication

# Environment variables set by torchrun:
# RANK: Global rank of the process
# LOCAL_RANK: Local rank on the node
# WORLD_SIZE: Total number of processes
# MASTER_ADDR: Address of the master node
# MASTER_PORT: Port of the master node
```

### Common Workflows

```bash
# Development setup
hatch shell                            # Enter dev environment
uv pip install -e .[all]              # Install all dependencies

# Run tests
hatch run test                         # Single GPU tests
torchrun --nproc_per_node=2 tests/test_ring_attention.py  # Multi-GPU tests

# Benchmarking
hatch run benchmark:run                # Single GPU benchmarks
torchrun --nproc_per_node=4 benchmarks/test_distributed_suite.py  # Multi-GPU benchmarks

# Code quality
hatch run all                          # Format, lint, typecheck, and test
```


### Benchmarking

The project includes a comprehensive benchmarking suite with shared utilities to reduce code duplication:

```bash
# Core benchmark utilities (NEW)
benchmarks/core/
    ├── base_benchmark.py        # Base classes for all benchmarks
    ├── utils/
│       ├── distributed.py      # Distributed computing utilities
│       ├── memory.py          # Memory management utilities
│       ├── timing.py          # Timing utilities with CUDA events
│       └── data.py            # Data generation utilities

# Single GPU benchmarks
hatch run benchmark:run                # Run benchmarks with default settings
hatch run benchmark:run --batch_size 2 --total_tokens 26 --heads 8  # Custom parameters
hatch run benchmark:profile            # Run with profiling

# Multi-GPU benchmarks (MUST use torchrun)
torchrun --nproc_per_node=2 benchmarks/test_distributed_suite.py
torchrun --nproc_per_node=4 benchmarks/test_ring_attention.py
torchrun --nproc_per_node=8 benchmarks/benchmark_ring_billion_tokens.py

# Direct execution (single GPU only)
python benchmarks/test_improved_suite.py      # Test all improved implementations
python benchmarks/verify_all.py               # Comprehensive verification

# Using uv for direct execution
uv run --extra benchmark python benchmark.py  # Run with benchmark dependencies
```

## Implementation Notes

### Device and Memory Considerations
- Uses CUDA when available, falls back to CPU
- Prefers float16/bfloat16 for performance on GPU
- Sequence lengths should be multiples of largest segment length
- Memory usage scales with sequence length and number of segments

### Dependencies
- **torch**: Core PyTorch functionality (>=2.0.0)
- **xformers**: Efficient attention operations (>=0.0.20)
- **einops**: Tensor rearrangement utilities (>=0.8.0)
- **flash-attn**: Flash attention implementation (>=2.8.0, supports FA3)
- **plotly**: Visualization for benchmarks (>=5.16.0)

### Flash Attention 3 Support
This project supports Flash Attention 3 for significant performance improvements:
- 1.5-2.0x faster than Flash Attention 2
- Up to 75% H100 GPU utilization
- Automatic detection and optimization
- See `FLASH_ATTENTION_3_SETUP.md` for detailed installation instructions

### Code Patterns
- All attention modules expect `batch_first=True` format
- Shape convention: `(batch_size, seq_len, num_heads, embed_dim)` for raw attention
- Shape convention: `(batch_size, seq_len, embed_dim)` for multihead attention
- Uses `is_causal` parameter for causal/non-causal attention

### Test Structure
Tests use pytest with parameterized testing:
- Multiple segment lengths and dilation rates
- Different sequence lengths and batch sizes  
- Both causal and non-causal attention modes
- GPU/CPU compatibility testing

## Ring Attention Implementation

### Ring Attention Classes

The project includes advanced Ring Attention implementations that provide O(n) memory complexity for arbitrarily long sequences:

- **RingDilatedAttention**: Alias for RingDilatedAttentionProduction (recommended for general use)
- **RingDilatedAttentionProduction** (`dilated_attention_pytorch/ring_dilated_attention_production.py`): Production-ready implementation with advanced error recovery and monitoring
- **RingDilatedAttentionProductionFixed** (`dilated_attention_pytorch/ring_dilated_attention_production_fixed.py`): Fixed version with standardized API wrapper
- **RingDistributedDilatedAttention** (`dilated_attention_pytorch/ring_distributed_dilated_attention.py`): Enterprise-grade distributed implementation with DeepSpeed integration
- **RingDilatedAttentionHilbertOptimizedFixed** (`dilated_attention_pytorch/ring_dilated_attention_hilbert_optimized_fixed.py`): Ring attention with Hilbert optimization and standardized API

## Block-Sparse Attention Implementation

### Block-Sparse Attention Classes

The project includes revolutionary Block-Sparse Attention implementations that combine O(n) memory complexity with 5-50x additional speedup:

- **BlockSparseRingDilatedAttention** (`dilated_attention_pytorch/block_sparse_ring_dilated_attention.py`): Core block-sparse ring attention with multiple pattern types
- **BlockSparseRingDilatedAttentionFixed** (`dilated_attention_pytorch/block_sparse_ring_dilated_attention_fixed.py`): Fixed version with standardized API wrapper
- **BlockSparseRingMultiheadDilatedAttention** (`dilated_attention_pytorch/block_sparse_ring_multihead_dilated_attention.py`): Drop-in replacement for nn.MultiheadAttention with block-sparse optimization
- **BlockSparseRingDistributedDilatedAttention** (`dilated_attention_pytorch/block_sparse_ring_distributed_dilated_attention.py`): Enterprise distributed implementation with hierarchical sparsity
- **BlockSparseAdaptive** (`dilated_attention_pytorch/block_sparse_adaptive.py`): Content-adaptive sparsity patterns that learn optimal attention
- **BlockSparseAdaptiveFixed** (`dilated_attention_pytorch/block_sparse_adaptive_fixed.py`): Fixed API wrapper for BlockSparseAdaptive
- **BlockSparseRingDilatedAttentionHilbertPostPattern** (`dilated_attention_pytorch/block_sparse_ring_dilated_attention_hilbert_post_pattern.py`): Hilbert curve optimization for block processing order (up to 2.53x speedup)

### Sparse Pattern Types

1. **Local Window**: Each position attends to nearby positions only
2. **Dilated Sparse**: Multi-scale attention with different dilation rates
3. **Global-Local**: Combination of global tokens and local windows
4. **Content-Adaptive**: Neural network learns optimal sparsity patterns

### Usage Examples

#### Using Factory Pattern (v0.2.0 - Recommended)

```python
# Auto-select best implementation based on hardware
from dilated_attention_pytorch.core import create_multihead_dilated_attention

attention = create_multihead_dilated_attention("auto",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

# Explicitly choose implementation
ring_attention = create_multihead_dilated_attention("ring",
    embed_dim=768,
    num_heads=12,
    segment_lengths=[2048, 4096],
    dilation_rates=[1, 2],
    ring_size=8
)

# Use with type-safe configuration
from dilated_attention_pytorch.core import DilatedAttentionConfig, MultiheadConfig

attention_config = DilatedAttentionConfig(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.1
)

multihead_config = MultiheadConfig(
    embed_dim=768,
    num_heads=12,
    layer_norm=True,
    gamma_init=1.0  # MAGNETO
)

attention = create_multihead_dilated_attention("improved",
    multihead_config=multihead_config,
    attention_config=attention_config
)
```

#### Block-Sparse Attention (Direct Import)

```python
# Quick block-sparse attention
from dilated_attention_pytorch import create_block_sparse_attention

attention = create_block_sparse_attention(
    embed_dim=768,
    num_heads=12,
    sparsity_ratio=0.1,  # 90% sparse = 10x speedup
    pattern_type='dilated_sparse'
)

# Adaptive sparse attention
from dilated_attention_pytorch import create_adaptive_sparse_attention

adaptive = create_adaptive_sparse_attention(
    embed_dim=768,
    num_heads=12
)
```

#### Conditional Implementation Selection

```python
def create_optimal_attention(seq_len, num_gpus=1):
    """Select best implementation based on context."""
    
    # Factory pattern makes this easy
    if seq_len > 100_000:
        impl = "ring"  # O(n) memory for very long sequences
    elif seq_len > 50_000 and num_gpus > 1:
        impl = "distributed"  # Multi-GPU optimization
    else:
        impl = "auto"  # Let factory decide
    
    return create_multihead_dilated_attention(impl,
        embed_dim=768,
        num_heads=12,
        segment_lengths=[2048, 4096, 8192],
        dilation_rates=[1, 2, 4]
    )

# Usage
attention_10k = create_optimal_attention(10_000)     # Likely "improved"
attention_1m = create_optimal_attention(1_000_000)  # Selects "ring"
```

## Core Refactoring (December 2024) - COMPLETE ✅

The codebase has been successfully refactored to reduce duplication and improve maintainability. New core modules provide shared functionality:

**Status**: Refactoring complete - 7/8 implementations refactored, 1 preserved by design (Block-Sparse)

### Implementations Status

**Refactored (using new core architecture):**
1. ✅ DilatedAttention
2. ✅ MultiheadDilatedAttention  
3. ✅ ImprovedDilatedAttention
4. ✅ ImprovedMultiheadDilatedAttention
5. ✅ RingDilatedAttentionHybrid (uses base classes)
6. ✅ DistributedMultiheadDilatedAttention
7. ✅ Various block-sparse implementations

**Not Refactored (by design):**
8. ⚡ BlockSparseRingDilatedAttention - Preserved for performance optimization

### Core Module Structure

- **core/base.py**: Abstract base classes for all implementations
  - `BaseDilatedAttention`: Common interface and caching
  - `BaseMultiheadDilatedAttention`: Multihead wrapper base
  
- **core/config.py**: Type-safe configuration dataclasses
  - Validation on initialization
  - Consistent parameter handling
  
- **core/memory_pool.py**: Unified memory management
  - Adaptive cleanup based on memory pressure
  - Thread-safe buffer allocation
  - Hot cache for frequently accessed patterns
  
- **core/attention_utils.py**: Common attention utilities
  - `optimize_attention_computation()`: Auto-selects best backend
  - Pattern generation functions
  - Positional encoding utilities
  
- **core/factory.py**: Simple API for module creation
  ```python
  from dilated_attention_pytorch.core import create_multihead_dilated_attention
  
  # Auto-select best implementation
  attention = create_multihead_dilated_attention("auto")
  
  # Create specific type
  attention = create_multihead_dilated_attention(
      "improved",
      embed_dim=768,
      num_heads=12,
      segment_lengths=[2048, 4096],
      dilation_rates=[1, 2]
  )
  ```

### Recent Changes (July 2025)

#### **Removed Implementations**
The following implementations were removed during cleanup:
- `ring_dilated_attention_v2_collective.py` - Superseded by Production version
- `ring_dilated_attention_refactored.py` - Merged into Production version
- `ring_hilbert_dilated_attention.py` - Functionality in HilbertOptimizedFixed
- `ring_dilated_attention_fixed.py` - Replaced by ProductionFixed
- `improved_distributed_dilated_attention.py` - Use distributed_dilated_attention.py
- `block_sparse_ring_dilated_attention_original.py` - Used deprecated APIs

### Recent Fixes and Optimizations (Latest Update - December 2024)

#### **Test Suite Improvements**
- Fixed 63 failing tests, achieving 93% pass rate (283/303 tests)
- Added pickle/deepcopy support for distributed training
- Fixed Ring Attention dimension mismatch with dilation_rates > 1
- Improved validation and error messages
- Added thread-safe operations for concurrent execution

### Recent Changes (July 2025)

### Deprecated Class Removal

Removed all implementations that used the poorly-performing `all_gather` operation:
- ~~`head_parallel_dilated_attention.py`~~ - Used all_gather with poor scalability
- ~~`improved_distributed_dilated_attention.py`~~ - Used all_gather 
- ~~`ring_dilated_attention_v2_collective.py`~~ - Used all_gather
- ~~`ring_hilbert_dilated_attention.py`~~ - Used all_gather
- ~~`ring_multihead_dilated_attention.py`~~ - Depended on deprecated V2Collective

Use `RingDilatedAttentionProduction` or `RingDistributedDilatedAttention` instead, which use efficient isend/irecv operations.

### Benchmark Suite Refactoring

Consolidated and refactored the benchmark suite to eliminate ~60% code duplication:
- Created shared utilities in `benchmarks/core/` for consistent benchmarking
- Consolidated redundant benchmark files into organized test suites
- Removed 17 redundant files while maintaining all testing capabilities
- Net reduction of ~500 lines of code with improved maintainability

## Recent Optimizations (December 2024)

#### **Block Sparse Ring Distributed Attention Optimizations**

All performance optimizations from Ring Distributed Attention have been successfully ported to Block Sparse Ring Distributed Attention:

##### **1. Adaptive Memory Pool Management**
- **Implementation**: `AdaptiveMemoryPool` class with dynamic cleanup thresholds
- **Features**:
  - Dynamic threshold adjustment based on GPU memory (aggressive when <10%, conservative when >50%)
  - Hot key cache for frequent access patterns (50 entries)
  - LRU eviction with usage statistics
  - Support for pinned memory allocations
- **Benefit**: 15-30% reduction in peak memory usage

##### **2. Smart Buffer Reuse**
- **Implementation**: `_get_smart_buffer()` method with intelligent reuse strategies
- **Features**:
  - Attempts reshape for same element count
  - Uses slicing for oversized buffers
  - Falls back to `resize_` operations
  - Integrated with memory pool for new allocations
- **Benefit**: Reduced allocation overhead, better memory locality

##### **3. LRU Cache Management**
- **Implementation**: OrderedDict-based buffer cache with access tracking
- **Features**:
  - Configurable cache size (default: 50 buffers)
  - Access count tracking for intelligent eviction
  - Thread-safe with buffer lock
- **Benefit**: Maintains performance while preventing memory bloat

##### **4. Optimized Gradient Communication**
- **Implementation**: `OptimizedGradientCommunicator` class
- **Features**:
  - Gradient bucketing with size + count thresholds (25MB OR 32 tensors)
  - Top-k gradient compression with error feedback
  - Asynchronous all-reduce operations
  - Automatic gradient hook registration
- **Benefit**: 90% bandwidth reduction, better communication efficiency

##### **5. Memory-Pinned Allocations**
- **Implementation**: Integrated into `AdaptiveMemoryPool`
- **Features**:
  - Automatic detection of CUDA availability
  - Non-blocking GPU transfers
  - Fallback to standard allocation on CPU
- **Benefit**: Reduced CPU-GPU transfer latency

##### **6. Enhanced Error Recovery**
- **Implementation**: Specialized error handlers for different failure types
- **Features**:
  - **OOM Recovery**: Aggressive memory clearing, precision reduction, gradient checkpointing
  - **Communication Recovery**: Gradient synchronization, single-node fallback
  - **Shape Recovery**: Automatic padding to power-of-2 sizes
  - **Generic Recovery**: Multi-level strategies with proper cleanup
- **Benefit**: Robust training with automatic failure recovery

#### **Performance Impact:**
- **Memory Efficiency**: 15-30% reduction in peak memory usage
- **Communication Speed**: ~2x faster with optimized gradient bucketing
- **Allocation Overhead**: Significant reduction through buffer reuse
- **Error Resilience**: Automatic recovery from common failure modes
- **Scalability**: Better handling of variable sequence lengths and batch sizes

#### **Original Ring Attention Optimizations (Previous Update)**

##### **Errors Fixed:**
1. **Critical Syntax Error** (ring_distributed_dilated_attention.py:337): Fixed incomplete parameter `ring_advancex` → `segment_lengths`
2. **Import Compatibility** (ring_dilated_attention.py): Added fallback for `torch.nn.attention` module in older PyTorch versions
3. **Dependencies**: Addressed protobuf compatibility issues in distributed class

#### **Compatibility Notes:**
- All classes now support PyTorch versions 1.9+ (automatic fallback for missing features)
- Improved error handling for distributed environments
- Better memory management for long-running training sessions
- Thread-safe operations for concurrent execution

## File Organization

```
src/
    └── dilated_attention_pytorch/
        ├── __init__.py              # Package init with exports
        ├── base/                    # Core implementations
        │   ├── __init__.py         # Base module exports
        │   ├── dilated_attention.py # Core dilated attention
        │   ├── multihead_dilated_attention.py  # Multi-head wrapper
        │   ├── improved_dilated_attention.py   # Enhanced version
        │   ├── improved_multihead_dilated_attention.py # Enhanced multihead
        │   ├── distributed_dilated_attention.py # Multi-GPU support
        │   └── head_parallel_dilated_attention_optimized.py # Head-parallel
        ├── ring/                    # Ring attention variants
        │   ├── __init__.py         # Ring module exports
        │   ├── base/               # Base ring implementations
        │   │   ├── ring_dilated_attention_correct.py
        │   │   ├── ring_dilated_attention_fixed_simple.py
        │   │   ├── ring_dilated_attention_memory_efficient.py
        │   │   ├── ring_dilated_attention_sdpa.py
        │   │   └── ring_dilated_attention_v3.py
        │   ├── distributed/        # Distributed ring attention
        │   │   └── ring_distributed_dilated_attention.py
        │   ├── hilbert/            # Hilbert-optimized ring attention
        │   │   ├── ring_dilated_attention_hilbert_core.py
        │   │   ├── ring_dilated_attention_hilbert_gpu_optimized.py
        │   │   ├── ring_dilated_attention_hilbert_optimized_fixed.py
        │   │   └── ring_dilated_attention_hilbert_proper.py
        │   └── utils/              # Ring attention utilities
        │       ├── ring_attention_autograd.py
        │       ├── ring_attention_lse.py
        │       └── ring_attention_utils.py
        ├── sparse/                  # Block-sparse implementations
        │   ├── __init__.py         # Sparse module exports
        │   ├── block_sparse_ring_dilated_attention.py
        │   ├── block_sparse_ring_dilated_attention_fixed.py
        │   ├── block_sparse_ring_dilated_attention_hilbert_post_pattern.py
        │   ├── block_sparse_ring_multihead_dilated_attention.py
        │   ├── block_sparse_ring_distributed_dilated_attention.py
        │   ├── block_sparse_adaptive.py
        │   ├── block_sparse_adaptive_fixed.py
        │   ├── block_sparse_factory.py
        │   └── sparse_pattern_generator.py
        ├── models/                  # Full models
        │   ├── __init__.py
        │   ├── transformer.py      # Transformer with dilated attention
        │   └── long_net.py         # Full LongNet architecture
        ├── core/                    # Core refactored components
        │   ├── __init__.py         # Core module exports
        │   ├── base.py             # Base classes for all implementations
        │   ├── config.py           # Configuration dataclasses
        │   ├── constants.py        # Feature detection and constants
        │   ├── memory_pool.py      # Unified memory pool
        │   ├── factory.py          # Factory pattern for module creation
        │   └── standardized_api.py # Standardized API wrappers
        ├── utils/                   # Utility modules
        │   ├── __init__.py         # Utils module exports
        │   ├── validation.py       # Validation utilities
        │   ├── attention_utils.py  # Common attention utilities
        │   ├── sparse_pattern_utils.py # Sparse pattern generation
        │   ├── hilbert_curve.py    # Hilbert curve utilities
        │   └── dynamic_segment_selector.py # Dynamic segment sizing
        ├── kernels/                 # CUDA/Triton kernels (experimental)
        │   ├── __init__.py
        │   ├── hilbert_attention_core.py
        │   └── hilbert_attention_triton_wrapper.py
        └── dynamic_dilated_attention.py # Dynamic segment sizing wrapper

tests/
    ├── __init__.py               # Tests package init
    ├── base/                     # Base implementation tests
    │   ├── test_dilated_attention.py
    │   ├── test_multihead_dilated_attention.py
    │   ├── test_improved_dilated_attention.py
    │   └── test_improved_multihead.py
    ├── ring/                     # Ring attention tests
    │   ├── test_ring_attention.py
    │   ├── test_distributed_ring_attention.py
    │   └── hilbert/             # Hilbert-specific tests
    │       ├── test_hilbert_gradient_comparison.py
    │       ├── test_multigpu_hilbert_ring.py
    │       └── test_per_segment_hilbert.py
    ├── sparse/                   # Block-sparse tests
    │   ├── test_block_sparse_attention.py
    │   ├── test_block_sparse_adaptive.py
    │   └── test_block_sparse_ring_multihead.py
    ├── models/                   # Model tests
    │   ├── test_long_net.py
    │   └── test_transformer.py
    ├── core/                     # Core infrastructure tests
    │   ├── test_factory.py
    │   ├── test_memory_pool.py
    │   └── test_core_refactoring.py
    ├── utils/                    # Utility tests
    │   ├── test_validation.py
    │   └── test_dynamic_segment_selection.py
    ├── misc/                     # Miscellaneous tests
    │   ├── test_edge_cases_validation.py
    │   ├── test_thread_safety.py
    │   ├── test_flash_attention_3.py
    │   └── test_memory_pool_consolidated.py
    └── TEST_REDUNDANCY_ANALYSIS.md # Test cleanup documentation

docs/                       # Extensive documentation
    ├── README.md               # Documentation overview
    ├── guides/                 # User guides (permanent names)
│       ├── ring-attention-guide.md
│       ├── block-sparse-attention-guide.md
│       ├── distributed-training-guide.md
│       ├── practical-usage-guide.md
│       └── factory-pattern-guide.md
    ├── benchmarks/             # Benchmark results (timestamped)
│       └── benchmark-results-YYYY-MM-DD-HHMM-UTC.md
    ├── feasibility/            # Feasibility studies (timestamped)
│       └── feasibility-study-YYYY-MM-DD-HHMM-UTC.md
    ├── reports/                # Technical reports (mixed naming)
│       └── defect-analysis-YYYY-MM-DD-HHMM-UTC.md
    └── archive/                # Historical/obsolete documentation

examples/                   # Example scripts
    ├── distributed_training_example.py # Distributed training example
    ├── basic_dilated_attention.py # Basic usage examples
    ├── distributed_ring_attention.py # Ring attention distributed example
    ├── factory_pattern_example.py # Factory pattern usage examples
    ├── simple_usage.py         # Simple usage examples
    └── ring_attention/         # Ring Attention educational implementations

scripts/                    # Utility scripts
    └── launch_distributed_training.py # Launch distributed training

benchmarks/                 # Performance benchmarking
    ├── core/                   # Shared benchmark utilities (NEW)
│       ├── base_benchmark.py   # Base classes for benchmarks
│       └── utils/             # Utility modules
│           ├── distributed.py # Distributed utilities
│           ├── memory.py      # Memory utilities
│           ├── timing.py      # Timing utilities
│           └── data.py        # Data generation
    ├── test_improved_suite.py  # Consolidated improved tests
    ├── test_distributed_suite.py # Consolidated distributed tests
    ├── verify_all.py           # Comprehensive verification
    ├── benchmark.py            # Main benchmark script
    ├── benchmark_all.py        # Comprehensive benchmarks
    ├── benchmark_ring_billion_tokens.py # Billion-token tests
    └── benchmark_sequence_limits.py # Sequence limit testing

analysis/                   # Analysis scripts
    ├── billion_token_analysis.py # Billion-token scaling
    ├── ring_attention_analysis.py # Ring attention analysis
    └── ring_performance_analysis.py # Performance analysis

README.md                  # Project documentation
CLAUDE.md                  # This file - AI instructions
PROJECT_STRUCTURE.md       # Project organization guide
pyproject.toml            # Modern Python package configuration
setup.py                  # Legacy package configuration
validate_changes.py       # Validation script
```

## Project Structure Rules

### IMPORTANT: Maintain Organized Directory Structure

When creating new files, ALWAYS place them in the correct directory:

1. **Documentation Files**:
   - User guides, tutorials → `docs/guides/`
   - Technical reports, analysis → `docs/reports/`
   - NEVER place documentation in root directory
   - Example: `docs/guides/new-feature-guide.md`, NOT `new-feature-guide.md`

2. **Test Files**:
   - Unit tests → `tests/test_*.py`
   - Verification scripts → `tests/verify_*.py`
   - NEVER place test files in root directory
   - Example: `tests/test_new_feature.py`, NOT `test_new_feature.py`

3. **Benchmark Scripts**:
   - Performance benchmarks → `benchmarks/benchmark_*.py`
   - Benchmark results → `benchmarks/*.txt` or `benchmarks/*.md`
   - Example: `benchmarks/benchmark_new_feature.py`, NOT `benchmark_new_feature.py`

4. **Analysis Scripts**:
   - Research/analysis code → `analysis/*_analysis.py`
   - Example: `analysis/new_feature_analysis.py`, NOT `new_feature_analysis.py`

5. **Utility Scripts**:
   - Debug scripts → `scripts/debug/`
   - Demo scripts → `scripts/demo/`
   - Other utilities → `scripts/`
   - Example: `scripts/debug/debug_new_feature.py`, NOT `debug_new_feature.py`

6. **Source Code**:
   - Base implementations → `dilated_attention_pytorch/base/`
   - Ring attention → `dilated_attention_pytorch/ring/`
   - Block-sparse → `dilated_attention_pytorch/sparse/`
   - Full models → `dilated_attention_pytorch/models/`
   - Utilities → `dilated_attention_pytorch/utils/`
   - Core infrastructure → `dilated_attention_pytorch/core/`

### File Creation Rules:
- ALWAYS check if an appropriate directory exists before creating a file
- NEVER create files in root unless they are project-level configs (pyproject.toml, etc.)
- When in doubt, ask the user where to place the file
- Maintain consistent naming conventions within each directory

### Root Directory:
The root directory should contain ONLY:
- README.md, LICENSE, CHANGELOG.md
- CONTRIBUTING.md, CODE_OF_CONDUCT.md
- CLAUDE.md, PROJECT_STRUCTURE.md
- Package config files (setup.py, pyproject.toml, etc.)
- Git config files (.gitignore, .gitattributes)
- validate_changes.py (project validation script)

## Documentation Structure and Naming Conventions

### Directory Structure:
- `docs/` - Main documentation directory
- `docs/guides/` - User guides and tutorials (permanent names)
- `docs/benchmarks/` - Benchmark results (timestamped)
- `docs/feasibility/` - Feasibility studies and analysis (timestamped)
- `docs/reports/` - Technical reports and analysis (timestamped)
- `docs/archive/` - Historical/obsolete documentation

### Naming Conventions:

#### 1. **Timestamped Documents** (results that may change over time):
These documents capture a snapshot at a specific moment and should be timestamped:

- **Benchmark Results**: `docs/benchmarks/benchmark-{description}-YYYY-MM-DD-HHMM-UTC.{ext}`
  - Example: `docs/benchmarks/benchmark-ring-attention-2025-06-26-1456-UTC.md`
  - Example: `docs/benchmarks/benchmark-1B-tokens-2025-06-26-1456-UTC.png`

- **Feasibility Studies**: `docs/feasibility/{topic}-feasibility-YYYY-MM-DD-HHMM-UTC.md`
  - Example: `docs/feasibility/1t-parameter-training-feasibility-2025-06-26-1456-UTC.md`

- **Defect Reports**: `docs/reports/defect-{type}-YYYY-MM-DD-HHMM-UTC.md`
  - Example: `docs/reports/defect-analysis-2025-06-26-1456-UTC.md`

- **Performance Analysis**: `docs/reports/{analysis-type}-YYYY-MM-DD-HHMM-UTC.md`
  - Example: `docs/reports/memory-optimization-analysis-2025-06-26-1456-UTC.md`

#### 2. **Permanent Documents** (guides and references):
These documents have stable content and use descriptive names without timestamps:

- **User Guides**: `docs/guides/{feature}-guide.md`
  - Example: `docs/guides/ring-attention-guide.md`
  - Example: `docs/guides/distributed-training-guide.md`

- **API Documentation**: `docs/guides/api-{module}.md`
  - Example: `docs/guides/api-dilated-attention.md`

- **Tutorials**: `docs/guides/tutorial-{topic}.md`
  - Example: `docs/guides/tutorial-custom-attention-patterns.md`

#### 3. **Timestamp Format**:
- Always use UTC time zone
- Format: `YYYY-MM-DD-HHMM-UTC`
- Example: `2025-06-26-1456-UTC` (June 26, 2025, 14:56 UTC)
- Generate with: `datetime.utcnow().strftime('%Y-%m-%d-%H%M-UTC')`

#### 4. **File Naming Rules**:
- Use lowercase with hyphens (kebab-case)
- Be descriptive but concise
- Include the document type in the name
- For timestamped files, timestamp goes at the end before extension
- Extensions: `.md` for markdown, `.png`/`.jpg` for images, `.json` for data

## Ring Attention Implementation Guidelines

### CRITICAL: Avoid Common Implementation Errors

#### 1. **Process Local Sequences Only**
The most critical error in ring attention is processing the full sequence before splitting:

```python
# WRONG - Defeats O(n/k) memory benefit!
qkv = self.qkv_proj(x)  # x is [batch, seq_len, embed_dim]
# Then splits AFTER projection - too late!

# CORRECT - Split first, then project
if self.world_size > 1 and not already_split:
    x_local = x[:, start:end, :].contiguous()
qkv = self.qkv_proj(x_local)  # Process local chunk only
```

#### 2. **Never Use all_gather**
- `all_gather` creates O(n²) communication and defeats the purpose
- Always use `isend/irecv` for ring communication pattern
- Removed implementations that used `all_gather` due to poor performance

#### 3. **Ring Communication Pattern**
```python
def ring_pass_forward(tensor):
    src = (rank - 1) % world_size
    dst = (rank + 1) % world_size
    
    recv_buffer = torch.empty_like(tensor)
    send_op = dist.isend(tensor.contiguous(), dst)
    recv_op = dist.irecv(recv_buffer, src)
    
    send_op.wait()
    recv_op.wait()
    return recv_buffer
```

#### 4. **Memory Management**
- Always ensure tensors are contiguous before communication
- Use aggressive memory cleanup for long sequences:
  ```python
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.synchronize()
  ```
- Pre-allocate communication buffers to reduce allocation overhead

#### 5. **Backend Selection for Ring Attention**
When using ring attention, select backend based on LOCAL sequence length:
```python
seq_len_hint = max(segment_lengths)
if memory_efficient and dist.is_initialized():
    seq_len_hint = seq_len_hint // dist.get_world_size()
```

### Benchmarking Guidelines

#### 1. **Multi-GPU Benchmarking Setup**
- Always verify `dist.is_initialized()` before using distributed features
- Use proper barriers for synchronization: `dist.barrier()`
- Profile with CUDA events for accurate timing

#### 2. **Memory Profiling**
- Monitor peak memory per GPU, not total
- Account for communication buffers in memory estimates
- Use `torch.cuda.max_memory_allocated()` for accurate measurements

#### 3. **Scaling Validation**
Verified scaling up to 1 billion tokens:
- Linear memory scaling: O(n/k) where k = world_size
- Constant memory per token regardless of total sequence length
- Example: 204,800 tokens with 4 GPUs = 459.2 MB per GPU

### GPU Architecture Considerations

#### 1. **Data Type Selection**
- Use float16/bfloat16 on Ampere+ GPUs (compute capability >= 8.0)
- Fall back to float32 for older GPUs (Pascal and earlier)
- Automatic detection based on `torch.cuda.get_device_capability()`

#### 2. **Flash Attention Backend**
- FA3 supported on H100/H200 GPUs (1.5-2x speedup)
- FA2 for A100/A10/RTX 30xx/40xx
- Automatic backend selection based on hardware

#### 3. **NCCL Optimization**
Environment variables for network optimization:
- `NCCL_SOCKET_IFNAME`: Specify network interface
- `NCCL_IB_DISABLE`: Disable InfiniBand if not available
- `NCCL_P2P_DISABLE`: Disable P2P for compatibility

### Hilbert Curve Optimization

When implementing Hilbert optimization:
1. Apply per-segment for cache efficiency
2. Use GPU-aware backend selection
3. Preserve numerical stability with proper LSE accumulation
4. Benchmark against standard ordering for your use case

### Performance Expectations

Based on extensive benchmarking:
- **Single GPU**: Standard attention up to ~32K tokens
- **Multi-GPU Ring**: Linear scaling to billions of tokens
- **Memory per token**: ~0.009 MB (constant with ring attention)
- **Communication overhead**: ~10-15% with proper implementation

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
ALWAYS follow the Project Structure Rules above when creating new files.