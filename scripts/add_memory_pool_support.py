"""
Add memory pool support to modules that currently lack it.

This script provides templates and guidance for adding memory pool support
to attention modules that handle large tensors.
"""

from pathlib import Path
from typing import List, Tuple


# Template for adding memory pool support
MEMORY_POOL_TEMPLATE = '''
# Add to imports
from .core import get_global_memory_pool

# Add to __init__ parameters
def __init__(
    self,
    # ... existing parameters ...
    enable_memory_pool: bool = False,
    enable_profiling: bool = False,
    # ... rest of parameters ...
):
    # ... existing init code ...
    
    # Memory pool setup
    self.enable_memory_pool = enable_memory_pool
    self._memory_pool = None
    if self.enable_memory_pool:
        self._memory_pool = get_global_memory_pool(
            enable_profiling=enable_profiling,
        )

# Add allocation method
def _allocate_tensor(
    self,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    zero_init: bool = True,
) -> torch.Tensor:
    """Allocate tensor using memory pool if enabled."""
    if self._memory_pool is not None:
        # Calculate tensor size
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        bytes_per_element = (
            torch.finfo(dtype).bits // 8
            if dtype.is_floating_point
            else torch.iinfo(dtype).bits // 8
        )
        size_mb = (num_elements * bytes_per_element) / (1024 * 1024)
        
        # Only use pool for tensors >= 1MB
        if size_mb >= 1.0:
            tensor = self._memory_pool.allocate(shape, dtype, device)
            if zero_init:
                tensor.zero_()
            return tensor
    
    # Fallback to regular allocation
    if zero_init:
        return torch.zeros(shape, dtype=dtype, device=device)
    else:
        return torch.empty(shape, dtype=dtype, device=device)

# Add deallocation method
def _deallocate_tensor(self, tensor: torch.Tensor) -> None:
    """Return tensor to memory pool if enabled."""
    if self._memory_pool is not None:
        self._memory_pool.deallocate(tensor)
'''


# Modules that need memory pool support
MODULES_NEEDING_SUPPORT = [
    {
        "file": "multihead_dilated_attention.py",
        "class": "MultiheadDilatedAttention",
        "large_tensors": ["qkv_proj output", "attention scores", "output projection"],
        "priority": "HIGH",
    },
    {
        "file": "improved_multihead_dilated_attention.py",
        "class": "ImprovedMultiheadDilatedAttention",
        "large_tensors": [
            "qkv_proj output",
            "attention scores",
            "output projection",
            "relative position bias",
        ],
        "priority": "HIGH",
    },
    {
        "file": "distributed_dilated_attention.py",
        "class": "DistributedMultiheadDilatedAttention",
        "large_tensors": ["distributed attention buffers", "gradient accumulation"],
        "priority": "HIGH",
    },
    {
        "file": "improved_distributed_dilated_attention.py",
        "classes": [
            "DistributedImprovedDilatedAttention",
            "DistributedImprovedMultiheadDilatedAttention",
        ],
        "large_tensors": ["communication buffers", "gradient buffers"],
        "priority": "HIGH",
    },
    {
        "file": "transformer.py",
        "classes": ["DilatedTransformerEncoderLayer", "DilatedTransformerDecoderLayer"],
        "large_tensors": [
            "self-attention output",
            "cross-attention output",
            "feedforward intermediate",
        ],
        "priority": "MEDIUM",
    },
    {
        "file": "long_net.py",
        "class": "LongNet",
        "large_tensors": ["embeddings", "transformer outputs", "language model head"],
        "priority": "MEDIUM",
    },
]


def generate_integration_guide(module_info: dict) -> str:
    """Generate integration guide for a specific module."""
    guide = f"""
Memory Pool Integration Guide for {module_info["file"]}
{"=" * 60}

Module: {module_info.get("class") or ", ".join(module_info.get("classes", []))}
Priority: {module_info["priority"]}

Large tensors that would benefit from pooling:
"""
    for tensor in module_info["large_tensors"]:
        guide += f"  - {tensor}\n"

    guide += """
Integration Steps:
1. Add memory pool imports at the top of the file
2. Add enable_memory_pool parameter to __init__ (default=False)
3. Initialize self._memory_pool in __init__ when enabled
4. Add _allocate_tensor and _deallocate_tensor methods
5. Replace large tensor allocations with _allocate_tensor calls
6. Add corresponding _deallocate_tensor calls in cleanup/del methods

Example locations to update:
"""

    if "multihead" in module_info["file"]:
        guide += """
  - In forward(): Replace torch.zeros/empty for attention scores
  - In _scaled_dot_product_attention(): Use for intermediate tensors
  - In projection layers: Use for output tensors
"""
    elif "distributed" in module_info["file"]:
        guide += """
  - In __init__(): Pre-allocate communication buffers
  - In forward(): Use for gradient accumulation buffers
  - In all_reduce operations: Use pooled buffers
"""
    elif "transformer" in module_info["file"]:
        guide += """
  - In forward(): Use for attention outputs
  - In feedforward layers: Use for intermediate activations
  - Consider reusing buffers across layers
"""

    return guide


def analyze_tensor_allocations(filepath: Path) -> List[Tuple[int, str]]:
    """Find potential tensor allocations in a file."""
    allocations = []

    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Look for tensor creation patterns
        if any(
            pattern in line
            for pattern in [
                "torch.zeros(",
                "torch.empty(",
                "torch.ones(",
                "torch.randn(",
                ".new_zeros(",
                ".new_empty(",
                ".new_ones(",
            ]
        ):
            allocations.append((i + 1, line.strip()))

    return allocations


def main():
    """Generate memory pool integration guidance."""
    project_root = Path(__file__).parent.parent
    attention_dir = project_root / "dilated_attention_pytorch"

    print("Memory Pool Integration Analysis")
    print("=" * 60)

    # Generate detailed guides
    guides_dir = project_root / "docs" / "memory_pool_integration"
    guides_dir.mkdir(parents=True, exist_ok=True)

    for module_info in MODULES_NEEDING_SUPPORT:
        filepath = attention_dir / module_info["file"]

        if not filepath.exists():
            print(f"Warning: {module_info['file']} not found")
            continue

        print(f"\nAnalyzing {module_info['file']}...")

        # Find tensor allocations
        allocations = analyze_tensor_allocations(filepath)

        # Generate guide
        guide = generate_integration_guide(module_info)
        guide += f"\nFound {len(allocations)} potential tensor allocations:\n"

        for line_num, line in allocations[:10]:  # Show first 10
            guide += f"  Line {line_num}: {line}\n"

        if len(allocations) > 10:
            guide += f"  ... and {len(allocations) - 10} more\n"

        # Save guide
        guide_path = guides_dir / f"{module_info['file']}.md"
        with open(guide_path, "w") as f:
            f.write(guide)

        print(f"  - Found {len(allocations)} tensor allocations")
        print(f"  - Guide saved to: {guide_path.relative_to(project_root)}")

    # Generate summary
    summary = f"""# Memory Pool Integration Summary

Generated: {Path(__file__).stat().st_mtime}

## Modules Requiring Memory Pool Support

Total modules: {len(MODULES_NEEDING_SUPPORT)}
High priority: {sum(1 for m in MODULES_NEEDING_SUPPORT if m["priority"] == "HIGH")}
Medium priority: {sum(1 for m in MODULES_NEEDING_SUPPORT if m["priority"] == "MEDIUM")}

## Integration Template

```python
{MEMORY_POOL_TEMPLATE}
```

## Module-Specific Guides

See individual files in this directory for detailed integration instructions.

## Best Practices

1. **Enable by default**: Keep memory pool disabled by default for backward compatibility
2. **Size threshold**: Only use pool for tensors >= 1MB to avoid overhead
3. **Cleanup**: Always implement proper deallocation in cleanup methods
4. **Reuse**: Consider reusing buffers across forward passes when possible
5. **Profiling**: Enable profiling during development to measure impact

## Testing

After integration, add tests to verify:
- Memory pool is used for large allocations
- Deallocation is properly handled
- No memory leaks occur
- Performance improves for large sequences
"""

    summary_path = guides_dir / "README.md"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nSummary saved to: {summary_path.relative_to(project_root)}")
    print("\nNext steps:")
    print("1. Review the generated guides in docs/memory_pool_integration/")
    print("2. Implement memory pool support in high-priority modules")
    print("3. Add tests using test_memory_pool_integration.py as a template")
    print("4. Benchmark performance impact with large sequences")


if __name__ == "__main__":
    main()
