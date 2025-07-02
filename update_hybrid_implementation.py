#!/usr/bin/env python3
"""
Script to update the hybrid implementation with the fixed dilated attention logic.
This will backup the original and apply the necessary changes.
"""

import os
import shutil
from datetime import datetime


def update_hybrid_implementation():
    """Update the hybrid implementation with the fix."""

    # Paths
    original_file = "dilated_attention_pytorch/ring_dilated_attention_hybrid.py"
    backup_file = f"dilated_attention_pytorch/ring_dilated_attention_hybrid.py.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    fixed_file = "dilated_attention_pytorch/ring_dilated_attention_hybrid_fixed.py"

    print("UPDATING HYBRID IMPLEMENTATION")
    print("=" * 60)

    # Check if files exist
    if not os.path.exists(original_file):
        print(f"Error: Original file not found: {original_file}")
        return False

    if not os.path.exists(fixed_file):
        print(f"Error: Fixed implementation not found: {fixed_file}")
        return False

    # Create backup
    print(f"1. Creating backup: {backup_file}")
    try:
        shutil.copy2(original_file, backup_file)
        print("   ✓ Backup created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create backup: {e}")
        return False

    # Read the fixed implementation
    print(f"2. Reading fixed implementation from: {fixed_file}")
    try:
        with open(fixed_file, "r") as f:
            fixed_content = f.read()

        # Update imports in the fixed content to match the original
        # (The fixed version might have different import paths)
        fixed_content = fixed_content.replace(
            "from .ring_utils_v3 import", "from .ring_attention_utils import"
        )

        print("   ✓ Fixed implementation loaded")
    except Exception as e:
        print(f"   ✗ Failed to read fixed implementation: {e}")
        return False

    # Write the updated implementation
    print(f"3. Updating original file: {original_file}")
    try:
        with open(original_file, "w") as f:
            f.write(fixed_content)
        print("   ✓ Implementation updated successfully")
    except Exception as e:
        print(f"   ✗ Failed to update implementation: {e}")
        # Restore backup
        print("   Restoring backup...")
        shutil.copy2(backup_file, original_file)
        return False

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    print(f"Original backed up to: {backup_file}")
    print("Implementation updated with proper dilated attention semantics")
    print("\nTo verify the update:")
    print("1. Single GPU: python tests/test_hybrid_fixed_correctness.py")
    print(
        "2. Multi-GPU:  torchrun --nproc_per_node=2 tests/test_hybrid_fixed_multi_gpu.py"
    )
    print("\nTo revert if needed:")
    print(f"   cp {backup_file} {original_file}")

    return True


def create_minimal_test():
    """Create a minimal test to verify the update worked."""

    test_code = '''#!/usr/bin/env python3
"""Minimal test to verify hybrid implementation update."""

import torch
from dilated_attention_pytorch.ring_dilated_attention_hybrid import (
    RingDilatedAttentionHybrid
)

# Test configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RingDilatedAttentionHybrid(
    segment_lengths=[8],
    dilation_rates=[2],
    device=device,
    dtype=torch.float32,
)

# Create test input with distinct segment values
seq_len = 16
q = torch.zeros(1, seq_len, 4, 32, device=device)
k = torch.zeros(1, seq_len, 4, 32, device=device)
v = torch.zeros(1, seq_len, 4, 32, device=device)

# Segment 1: value 1.0, Segment 2: value 10.0
q[:, :8] = 1.0
k[:, :8] = 1.0
v[:, :8] = 1.0
q[:, 8:] = 10.0
k[:, 8:] = 10.0
v[:, 8:] = 10.0

# Run model
with torch.no_grad():
    output = model(q, k, v, is_causal=False)

# Check segment means
seg1_mean = output[:, :8].mean().item()
seg2_mean = output[:, 8:].mean().item()

print("HYBRID IMPLEMENTATION TEST")
print("=" * 40)
print(f"Segment 1 mean: {seg1_mean:.4f} (expected ~1.0)")
print(f"Segment 2 mean: {seg2_mean:.4f} (expected ~10.0)")

# Check locality
locality_preserved = abs(seg1_mean - 1.0) < 0.1 and abs(seg2_mean - 10.0) < 0.1
print(f"\\nLocality preserved: {locality_preserved}")

if locality_preserved:
    print("✓ Update successful - dilated attention working correctly!")
else:
    print("✗ Update may have issues - segments are mixing!")
'''

    test_file = "test_hybrid_update.py"
    with open(test_file, "w") as f:
        f.write(test_code)

    print(f"\nCreated minimal test: {test_file}")
    print("Run it with: python test_hybrid_update.py")


if __name__ == "__main__":
    success = update_hybrid_implementation()
    if success:
        create_minimal_test()
