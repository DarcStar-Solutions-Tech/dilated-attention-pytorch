# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an unofficial PyTorch implementation of DilatedAttention from the LongNet paper "LongNet: Scaling Transformers to 1,000,000,000 Tokens". The project provides efficient attention mechanisms for handling very long sequences.

## Core Architecture

### Main Components

- **DilatedAttention** (`dilated_attention_pytorch/dilated_attention.py`): Core dilated attention mechanism that supports variable segment lengths and dilation rates
- **MultiheadDilatedAttention** (`dilated_attention_pytorch/multihead_dilated_attention.py`): Drop-in replacement for nn.MultiheadAttention with dilated attention and MAGNETO improvements
- **ImprovedDilatedAttention** (`dilated_attention_pytorch/improved_dilated_attention.py`): Enhanced version with additional optimizations
- **DistributedDilatedAttention** (`dilated_attention_pytorch/distributed_dilated_attention.py`): Distributed/multi-GPU implementation
- **LongNet** (`dilated_attention_pytorch/long_net.py`): Full transformer architecture for language modeling
- **Transformer** (`dilated_attention_pytorch/transformer.py`): General transformer with dilated attention

### Key Parameters

All dilated attention modules require:
- `segment_lengths`: Geometric sequence (e.g., [2048, 4096, 8192])
- `dilation_rates`: Corresponding dilation rates (e.g., [1, 2, 4])
- Sequence length must be divisible by the largest segment length

## Development Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_dilated_attention.py
pytest tests/test_long_net.py

# Run tests with specific parameters
pytest tests/test_dilated_attention.py -v
```

### Dependencies Management
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Install development dependencies (from setup.py)
pip install -e .[dev]
pip install -e .[test]
pip install -e .[all]

# Using Poetry (if available)
poetry install
```

### Code Quality
Based on setup.py, the project supports:
```bash
# Code formatting
black .
isort .

# Linting
flake8 .
mypy .

# Pre-commit hooks (if configured)
pre-commit install
pre-commit run --all-files
```

### Benchmarking
```bash
# Run benchmarks with default settings
python benchmark.py

# Custom benchmark parameters
python benchmark.py --batch_size 2 --total_tokens 26 --heads 8
```

## Implementation Notes

### Device and Memory Considerations
- Uses CUDA when available, falls back to CPU
- Prefers float16/bfloat16 for performance on GPU
- Sequence lengths should be multiples of largest segment length
- Memory usage scales with sequence length and number of segments

### Dependencies
- **torch**: Core PyTorch functionality
- **xformers**: Efficient attention operations
- **einops**: Tensor rearrangement utilities
- **flash-attn**: Flash attention implementation
- **plotly**: Visualization for benchmarks

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

## File Organization

```
dilated_attention_pytorch/
├── __init__.py              # Empty package init
├── dilated_attention.py     # Core dilated attention
├── multihead_dilated_attention.py  # Multi-head wrapper
├── improved_dilated_attention.py   # Enhanced version
├── distributed_dilated_attention.py # Multi-GPU support
├── transformer.py           # Transformer with dilated attention
└── long_net.py             # Full LongNet architecture

tests/
├── test_dilated_attention.py # Core attention tests
└── test_long_net.py          # LongNet architecture tests

benchmark.py                 # Performance benchmarking
requirements.txt            # Python dependencies
setup.py                    # Package configuration
```