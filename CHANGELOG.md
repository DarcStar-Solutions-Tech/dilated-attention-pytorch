# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **CI/CD Improvements**:
  - Dependabot configuration for automated dependency updates
  - GitHub issue templates (bug report, feature request)
  - Pull request template with checklist
  - Code of Conduct
  - Contributing guidelines
  - `.gitattributes` for consistent line endings
  - `py.typed` file for PEP 561 compliance
- **Development Tooling**:
  - Pre-commit hooks with Ruff and mypy
  - Ruff replaces Black, isort, and flake8
  - Enhanced Ruff rules for better code quality
  - Stricter mypy configuration

### Changed
- **Python Support**: Minimum Python version raised to 3.12
- **Linting**: Switched from Black/isort/flake8 to Ruff
- **Repository URLs**: Updated to DarcStar-Solutions-Tech organization
- **Dependencies**: Removed legacy linting tools from test dependencies
- **Documentation**: Updated installation instructions and development workflow

### Removed
- Python 3.9, 3.10, and 3.11 support
- Poetry configuration files (using Hatch)
- `setup.cfg` (moved all configuration to `pyproject.toml`)

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