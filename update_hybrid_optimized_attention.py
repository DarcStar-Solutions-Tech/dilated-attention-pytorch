#!/usr/bin/env python3
"""
Update hybrid implementation to use optimized attention computation.
"""


def update_hybrid_to_use_optimized_attention():
    """Update the hybrid implementation to use the optimized attention fallback chain."""

    hybrid_file = "dilated_attention_pytorch/ring_dilated_attention_hybrid.py"

    print("Updating hybrid implementation to use optimized attention...")

    # Read the file
    with open(hybrid_file, "r") as f:
        content = f.read()

    # Add import for optimized LSE
    import_section = """# Import LSE utilities from V3
from .ring_attention_lse import (
    StableRingAccumulator,
    compute_attention_with_lse,
)"""

    new_import_section = """# Import LSE utilities from V3
from .ring_attention_lse import (
    StableRingAccumulator,
    compute_attention_with_lse,
)
# Import optimized LSE with backend fallbacks
try:
    from .ring_attention_lse_optimized import compute_attention_with_lse_optimized
    HAS_OPTIMIZED_LSE = True
except ImportError:
    HAS_OPTIMIZED_LSE = False
    compute_attention_with_lse_optimized = compute_attention_with_lse"""

    content = content.replace(import_section, new_import_section)

    # Update _compute_segment_attention to use optimized version
    old_fallback = """        # Standard attention computation with LSE
        return compute_attention_with_lse(
            q_seg,
            k_seg,
            v_seg,
            scale=1.0 / math.sqrt(q_seg.shape[-1]),
            mask=self._get_segment_causal_mask(
                q_seg.shape[2], k_seg.shape[2], seg_start, overlap_start, is_causal
            ),
            dropout=self.dropout,
            training=self.training,
        )"""

    new_fallback = """        # Use optimized attention computation with backend fallbacks
        if HAS_OPTIMIZED_LSE:
            return compute_attention_with_lse_optimized(
                q_seg,
                k_seg,
                v_seg,
                scale=1.0 / math.sqrt(q_seg.shape[-1]),
                mask=self._get_segment_causal_mask(
                    q_seg.shape[2], k_seg.shape[2], seg_start, overlap_start, is_causal
                ),
                dropout=self.dropout,
                training=self.training,
                is_causal=is_causal and seg_start == 0,  # Only first segment needs causal
            )
        else:
            # Fallback to standard computation
            return compute_attention_with_lse(
                q_seg,
                k_seg,
                v_seg,
                scale=1.0 / math.sqrt(q_seg.shape[-1]),
                mask=self._get_segment_causal_mask(
                    q_seg.shape[2], k_seg.shape[2], seg_start, overlap_start, is_causal
                ),
                dropout=self.dropout,
                training=self.training,
            )"""

    content = content.replace(old_fallback, new_fallback)

    # Update single device computation too
    old_single_device = """            # Compute attention for segment
            seg_output, _ = compute_attention_with_lse(
                q_seg,
                k_seg,
                v_seg,
                scale=1.0 / math.sqrt(d),
                mask=self._get_segment_causal_mask(
                    q_seg.shape[2], k_seg.shape[2], seg_start, seg_start, is_causal
                ),
                dropout=self.dropout,
                training=self.training,
            )"""

    new_single_device = """            # Compute attention for segment with optimized backend
            if HAS_OPTIMIZED_LSE:
                seg_output, _ = compute_attention_with_lse_optimized(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=1.0 / math.sqrt(d),
                    mask=self._get_segment_causal_mask(
                        q_seg.shape[2], k_seg.shape[2], seg_start, seg_start, is_causal
                    ),
                    dropout=self.dropout,
                    training=self.training,
                    is_causal=is_causal and seg_start == 0,
                )
            else:
                seg_output, _ = compute_attention_with_lse(
                    q_seg,
                    k_seg,
                    v_seg,
                    scale=1.0 / math.sqrt(d),
                    mask=self._get_segment_causal_mask(
                        q_seg.shape[2], k_seg.shape[2], seg_start, seg_start, is_causal
                    ),
                    dropout=self.dropout,
                    training=self.training,
                )"""

    content = content.replace(old_single_device, new_single_device)

    # Write back
    with open(hybrid_file, "w") as f:
        f.write(content)

    print("✓ Updated hybrid implementation to use optimized attention")
    print("  - Added import for compute_attention_with_lse_optimized")
    print("  - Updated _compute_segment_attention to use optimized backend")
    print("  - Updated single device computation")
    print("\nThe implementation now uses the fallback chain:")
    print("  1. Flash Attention (if available)")
    print("  2. PyTorch SDPA (if available)")
    print("  3. xFormers (if available)")
    print("  4. Standard einsum implementation")


if __name__ == "__main__":
    update_hybrid_to_use_optimized_attention()
