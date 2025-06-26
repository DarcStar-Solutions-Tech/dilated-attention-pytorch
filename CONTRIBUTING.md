# Contributing to Dilated Attention PyTorch

Thank you for your interest in contributing to dilated-attention-pytorch! We welcome contributions from the community.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/dilated-attention-pytorch.git
   cd dilated-attention-pytorch
   ```
3. **Set up your development environment**:
   ```bash
   # Create a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install the package in development mode with all dependencies
   pip install -e ".[all]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names like:
- `feature/add-flash-attention-3`
- `fix/memory-leak-in-ring-attention`
- `docs/update-installation-guide`

### 2. Make Your Changes

- Write clean, readable code following the project's style
- Add or update tests as needed
- Update documentation if you're changing functionality
- Add docstrings to new functions and classes

### 3. Run Quality Checks

Before committing, ensure your code passes all checks:

```bash
# Format your code
hatch run format

# Run linting
hatch run lint

# Run type checking
hatch run typecheck

# Run tests
hatch run test

# Or run all checks at once
hatch run all
```

### 4. Commit Your Changes

Write clear, descriptive commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) format:

```bash
git commit -m "feat: add support for Flash Attention 3"
git commit -m "fix: resolve memory leak in ring attention forward pass"
git commit -m "docs: update installation instructions for Python 3.13"
```

Pre-commit hooks will automatically run when you commit. If they fail, fix the issues and try again.

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- A clear title and description
- Reference to any related issues
- Summary of the changes made
- Any additional context that reviewers should know

## Code Style Guidelines

### Python Style

- We use **Ruff** for formatting and linting
- Follow PEP 8 with a line length of 100 characters
- Use type hints for function arguments and return values
- Write descriptive variable names

### Docstring Format

Use Google-style docstrings:

```python
def dilated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_lengths: List[int],
    dilation_rates: List[int],
) -> torch.Tensor:
    """
    Apply dilated attention mechanism.
    
    Args:
        query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, num_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, num_heads, head_dim)
        segment_lengths: List of segment lengths for each attention head
        dilation_rates: List of dilation rates for each segment
        
    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim)
        
    Raises:
        ValueError: If segment_lengths and dilation_rates have different lengths
    """
```

## Testing Guidelines

### Writing Tests

- Add tests for any new functionality
- Use pytest for testing
- Place tests in the `tests/` directory
- Use descriptive test names that explain what is being tested

Example test:

```python
import pytest
import torch
from dilated_attention_pytorch import DilatedAttention

def test_dilated_attention_output_shape():
    """Test that dilated attention produces correct output shape."""
    batch_size, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    
    attention = DilatedAttention(
        segment_lengths=[256, 512],
        dilation_rates=[1, 2]
    )
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    output = attention(q, k, v)
    
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)
```

### Running Tests

```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Run a specific test file
pytest tests/test_dilated_attention.py

# Run tests matching a pattern
pytest -k "test_output_shape"
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. Python version and PyTorch version
2. Operating system
3. GPU type (if applicable)
4. Minimal code example that reproduces the issue
5. Full error message and stack trace
6. What you expected to happen

### Feature Requests

For feature requests, please describe:

1. The problem you're trying to solve
2. Your proposed solution
3. Any alternative solutions you've considered
4. Examples of how the feature would be used

## Getting Help

- Check existing issues and pull requests
- Read the documentation
- Ask questions in GitHub Discussions
- Review the test files for usage examples

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## License

By contributing to dilated-attention-pytorch, you agree that your contributions will be licensed under the MIT License.