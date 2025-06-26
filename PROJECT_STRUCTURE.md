# Project Structure

This document describes the organization of the Dilated Attention PyTorch repository.

## Directory Layout

```
dilated-attention-pytorch/
├── dilated_attention_pytorch/     # Main package source code
│   ├── core/                      # Core refactored components
│   │   ├── base.py               # Abstract base classes
│   │   ├── config.py             # Configuration dataclasses
│   │   ├── factory.py            # Factory pattern for module creation
│   │   └── memory_pool.py        # Memory management utilities
│   ├── utils/                    # Utility modules
│   │   ├── attention_utils.py    # Attention computation helpers
│   │   ├── sparse_pattern_utils.py # Sparse pattern generation
│   │   └── validation.py         # Input validation utilities
│   ├── __init__.py              # Package exports
│   ├── dilated_attention.py     # Core dilated attention
│   ├── multihead_dilated_attention.py # Multi-head wrapper
│   ├── improved_*.py            # Enhanced implementations
│   ├── ring_*.py                # Ring attention implementations
│   ├── block_sparse_*.py        # Block-sparse implementations
│   ├── transformer.py           # Transformer architectures
│   └── long_net.py              # LongNet implementation
│
├── tests/                        # Test suite
│   ├── test_*.py                # Unit tests
│   ├── compare_implementations.py # Implementation comparisons
│   └── memory_*.py              # Memory analysis tests
│
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark.py             # Main benchmark script
│   ├── benchmark_all.py         # Comprehensive benchmarks
│   ├── benchmark_ring_billion_tokens.py # Billion-token tests
│   └── sequence_benchmark_results.txt # Benchmark results
│
├── analysis/                     # Analysis scripts
│   ├── billion_token_analysis.py # Billion-token scaling analysis
│   ├── ring_attention_analysis.py # Ring attention analysis
│   └── ring_performance_analysis.py # Performance analysis
│
├── scripts/                      # Utility scripts
│   ├── debug/                   # Debugging utilities
│   │   ├── debug_block_sparse.py
│   │   ├── debug_forward_pass.py
│   │   └── debug_unfold.py
│   ├── demo/                    # Demo scripts
│   │   ├── key_findings_demo.py
│   │   └── quick_performance_demo.py
│   ├── optimize_*.py            # Optimization scripts
│   └── profile_*.py             # Profiling scripts
│
├── examples/                     # Usage examples
│   ├── simple_usage.py          # Basic usage
│   ├── factory_pattern_example.py # Factory pattern usage
│   └── distributed_training_example.py # Distributed training
│
├── docs/                        # Documentation
│   ├── README.md                # Documentation index
│   ├── guides/                  # User guides
│   │   ├── ring-attention-guide.md
│   │   ├── block-sparse-attention-guide.md
│   │   ├── distributed-training-guide.md
│   │   └── billion-token-deployment-guide.md
│   └── reports/                 # Technical reports
│       ├── *_REPORT.md          # Analysis reports
│       └── *_SUMMARY.md         # Implementation summaries
│
├── CLAUDE.md                    # Claude Code instructions
├── README.md                    # Project README
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── pyproject.toml              # Modern Python packaging
├── poetry.lock                 # Poetry dependencies
├── requirements.txt            # Pip requirements
└── setup.py                    # Legacy setup script
```

## Key Components

### Core Package (`dilated_attention_pytorch/`)
- **Core modules**: Base implementations of dilated attention mechanisms
- **Improved variants**: Enhanced versions with optimizations
- **Ring attention**: O(n) memory complexity implementations
- **Block-sparse**: Sparse attention patterns for efficiency
- **Utilities**: Shared utilities and helpers

### Tests (`tests/`)
- Unit tests for all implementations
- Memory usage analysis
- Performance comparisons
- Thread safety tests

### Benchmarks (`benchmarks/`)
- Performance benchmarking scripts
- Billion-token scaling tests
- Sequence length limit testing
- Results and analysis

### Documentation (`docs/`)
- User guides for different use cases
- Technical reports and analyses
- API documentation
- Implementation details

### Examples (`examples/`)
- Simple usage examples
- Advanced patterns
- Distributed training setups

### Scripts (`scripts/`)
- Development utilities
- Debugging tools
- Performance profiling
- Optimization experiments

## Development Workflow

1. **Source code**: Make changes in `dilated_attention_pytorch/`
2. **Testing**: Run tests from `tests/` directory
3. **Benchmarking**: Use scripts in `benchmarks/` for performance testing
4. **Documentation**: Update guides in `docs/` as needed
5. **Examples**: Add usage examples in `examples/`

## Import Structure

After reorganization, imports remain unchanged:
```python
from dilated_attention_pytorch import RingDilatedAttention
from dilated_attention_pytorch.core import create_multihead_dilated_attention
```

All scripts moved to subdirectories can be run from the project root:
```bash
python benchmarks/benchmark.py
python tests/test_ring_attention.py
python scripts/demo/quick_performance_demo.py
```