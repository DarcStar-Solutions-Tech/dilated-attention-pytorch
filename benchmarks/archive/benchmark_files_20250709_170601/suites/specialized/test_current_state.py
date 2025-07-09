#!/usr/bin/env python3
"""Test current state of implementations after RingDilatedAttentionProduction removal."""

implementations = [
    # Core
    "DilatedAttention",
    "ImprovedDilatedAttention",
    # Multihead
    "MultiheadDilatedAttention",
    "ImprovedMultiheadDilatedAttention",
    # Ring (removed RingDilatedAttentionProduction)
    "RingDilatedAttentionHilbertOptimizedFixed",
    "RingDistributedDilatedAttention",
    # Block-sparse
    "BlockSparseRingDilatedAttention",
    "BlockSparseRingDilatedAttentionFixed",
    "BlockSparseRingMultiheadDilatedAttention",
    "BlockSparseRingDistributedDilatedAttention",
    "BlockSparseAdaptive",
    "BlockSparseAdaptiveFixed",
    "BlockSparseRingDilatedAttentionHilbertPostPattern",
    # Distributed
    "DistributedMultiheadDilatedAttention",
    # Kernels
    "HilbertAttentionTritonFixed",
    "HilbertDilatedAttention",
    # Head-parallel
    "HeadParallelDilatedAttentionOptimized",
]

print(f"Testing {len(implementations)} implementations...\n")

working = []
failed = []

for impl in implementations:
    try:
        module = __import__("dilated_attention_pytorch", fromlist=[impl])
        if hasattr(module, impl):
            cls = getattr(module, impl)
            print(f"✅ {impl}: Successfully imported")
            working.append(impl)
        else:
            print(f"❌ {impl}: Not found in module")
            failed.append((impl, "Not found in module"))
    except Exception as e:
        print(f"❌ {impl}: {e}")
        failed.append((impl, str(e)))

print(f"\n{'=' * 60}")
print(f"Summary: {len(working)}/{len(implementations)} implementations work")
print(f"Success rate: {len(working) / len(implementations) * 100:.1f}%")

if failed:
    print("\nFailed implementations:")
    for name, error in failed:
        print(f"  - {name}: {error}")
