#!/usr/bin/env python
"""Debug script to check factory registration issues."""

import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

# Import the factory module
try:
    from dilated_attention_pytorch.core.factory import (
        _ATTENTION_REGISTRY,
        _MULTIHEAD_REGISTRY,
        _register_implementations,
    )

    print("Successfully imported factory module")

    # Call registration function
    _register_implementations()

    print(f"\nAttention Registry: {list(_ATTENTION_REGISTRY.keys())}")
    print(f"Multihead Registry: {list(_MULTIHEAD_REGISTRY.keys())}")

except Exception as e:
    print(f"Error during import: {e}")
    import traceback

    traceback.print_exc()

# Also test individual imports
print("\n\nTesting individual imports:")
modules_to_test = [
    "dilated_attention_pytorch.dilated_attention",
    "dilated_attention_pytorch.multihead_dilated_attention",
    "dilated_attention_pytorch.improved_dilated_attention",
    "dilated_attention_pytorch.improved_multihead_dilated_attention",
    "dilated_attention_pytorch.ring_dilated_attention",
    "dilated_attention_pytorch.ring_multihead_dilated_attention",
    "dilated_attention_pytorch.ring_distributed_refactored",
    "dilated_attention_pytorch.improved_distributed_dilated_attention",
    "dilated_attention_pytorch.block_sparse_ring_dilated_attention",
    "dilated_attention_pytorch.block_sparse_ring_multihead_dilated_attention",
]

for module_name in modules_to_test:
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
    except Exception as e:
        print(f"✗ {module_name}: {type(e).__name__}: {e}")
