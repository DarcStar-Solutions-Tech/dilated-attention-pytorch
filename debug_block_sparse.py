"""
Debug script to find where block sparse implementation hangs
"""
import torch
import sys

print("Starting debug script...")

# Add debug prints to understand where the hang occurs
print("1. Importing module...")
sys.stdout.flush()

try:
    from dilated_attention_pytorch.block_sparse_ring_dilated_attention import (
        BlockSparseRingDilatedAttention,
        SparsePatternConfig,
    )
    print("2. Import successful")
    sys.stdout.flush()
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("3. Creating sparse config...")
sys.stdout.flush()

try:
    sparse_config = SparsePatternConfig(
        pattern_type='dilated_sparse',
        sparsity_ratio=0.1,
        block_size=32,
    )
    print("4. Sparse config created")
    sys.stdout.flush()
except Exception as e:
    print(f"Config creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("5. Creating module...")
sys.stdout.flush()

try:
    module = BlockSparseRingDilatedAttention(
        segment_lengths=[1024, 2048],
        dilation_rates=[1, 2],
        sparse_config=sparse_config,
        dropout=0.0,
        ring_size=1,  # Explicitly set ring size
    )
    print("6. Module created successfully!")
    sys.stdout.flush()
except Exception as e:
    print(f"Module creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Debug complete - no hang in module creation")