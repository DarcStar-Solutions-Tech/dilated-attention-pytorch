#!/usr/bin/env python3
"""
Demonstrate SDPA (Scaled Dot Product Attention) with dilated attention masks.

This script shows how to create and apply dilated attention patterns using
PyTorch's native SDPA functionality with custom attention masks.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time


def create_dilated_mask(
    seq_len: int,
    dilation_rate: int,
    offset: int = 0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a dilated attention mask for SDPA.

    Args:
        seq_len: Sequence length
        dilation_rate: Dilation rate (attend every nth position)
        offset: Offset for the dilation pattern
        device: Device to create mask on
        dtype: Data type for mask

    Returns:
        Attention mask of shape [seq_len, seq_len]
    """
    # Create base mask (all -inf)
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

    # Fill dilated positions with 0
    for i in range(seq_len):
        # Each position attends to positions at dilated intervals
        attend_positions = torch.arange(offset, seq_len, dilation_rate, device=device)

        # Only attend to positions that match our dilation pattern
        if i % dilation_rate == offset:
            valid_positions = attend_positions[
                attend_positions % dilation_rate == offset
            ]
            mask[i, valid_positions] = 0.0

    return mask


def create_segmented_dilated_mask(
    seq_len: int,
    segment_lengths: list,
    dilation_rates: list,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create attention mask for segmented dilated attention.

    Each segment has its own dilation rate.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)

    position = 0
    for seg_len, dil_rate in zip(segment_lengths, dilation_rates):
        seg_end = min(position + seg_len, seq_len)
        _ = seg_end - position

        if dil_rate == 1:
            # Full attention within segment
            mask[position:seg_end, position:seg_end] = 0.0
        else:
            # Dilated attention within segment
            for offset in range(dil_rate):
                indices = torch.arange(
                    position + offset, seg_end, dil_rate, device=device
                )
                if len(indices) > 0:
                    # Each dilated group attends to itself
                    for idx in indices:
                        mask[idx, indices] = 0.0

        position = seg_end
        if position >= seq_len:
            break

    return mask


def visualize_attention_mask(mask: torch.Tensor, title: str = "Attention Mask"):
    """Visualize attention mask pattern."""
    # Convert to numpy and handle -inf values
    mask_np = mask.cpu().numpy()
    mask_vis = np.where(mask_np == float("-inf"), 0, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(mask_vis, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Attention Weight (0=masked, 1=allowed)")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def benchmark_sdpa_with_masks(
    batch_size: int = 2,
    seq_len: int = 1024,
    num_heads: int = 8,
    head_dim: int = 64,
    device: torch.device = torch.device("cpu"),
):
    """Benchmark SDPA with different mask patterns."""
    print(f"\nBenchmarking SDPA with masks on {device}")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Heads: {num_heads}, Dim: {head_dim}")

    # Create test tensors
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Test configurations
    configs = [
        ("No mask", None),
        ("Causal mask", "causal"),
        (
            "Dilated (rate=2)",
            create_dilated_mask(seq_len, 2, device=device, dtype=dtype),
        ),
        (
            "Dilated (rate=4)",
            create_dilated_mask(seq_len, 4, device=device, dtype=dtype),
        ),
        (
            "Segmented dilated",
            create_segmented_dilated_mask(
                seq_len, [256, 512, 256], [1, 2, 4], device=device, dtype=dtype
            ),
        ),
    ]

    results = {}

    for name, mask in configs:
        # Handle causal mask specially
        if mask == "causal":
            is_causal = True
            attn_mask = None
        else:
            is_causal = False
            attn_mask = mask

        # Warmup
        for _ in range(10):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                _ = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        num_iters = 100

        for _ in range(num_iters):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                _ = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        avg_time = (end - start) / num_iters * 1000  # ms
        results[name] = avg_time

    # Print results
    print("\nResults (ms per forward pass):")
    baseline = results.get("No mask", 1.0)
    for name, time_ms in results.items():
        slowdown = time_ms / baseline
        print(f"  {name:20s}: {time_ms:8.2f} ms (slowdown: {slowdown:.2f}x)")

    return results


def test_dilated_attention_correctness():
    """Test that dilated attention with SDPA produces correct results."""
    print("\nTesting dilated attention correctness...")

    # Small example for verification
    seq_len = 16
    batch_size = 1
    num_heads = 1
    head_dim = 8
    dilation_rate = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create simple test data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn_like(q)
    v = (
        torch.arange(seq_len, device=device, dtype=q.dtype)
        .view(1, 1, seq_len, 1)
        .expand_as(v)
    )

    # Method 1: SDPA with dilated mask
    mask = create_dilated_mask(seq_len, dilation_rate, device=device, dtype=q.dtype)
    output_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Method 2: Manual dilated attention
    output_manual = torch.zeros_like(v)
    scale = 1.0 / (head_dim**0.5)

    for offset in range(dilation_rate):
        indices = torch.arange(offset, seq_len, dilation_rate, device=device)
        if len(indices) == 0:
            continue

        q_dilated = q[:, :, indices, :]
        k_dilated = k[:, :, indices, :]
        v_dilated = v[:, :, indices, :]

        scores = torch.matmul(q_dilated, k_dilated.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        dilated_output = torch.matmul(attn_weights, v_dilated)

        output_manual[:, :, indices, :] = dilated_output

    # Compare results
    diff = torch.abs(output_sdpa - output_manual).max().item()
    print(f"Max difference between SDPA and manual: {diff:.6f}")
    print(f"Results match: {diff < 1e-5}")

    # Visualize one head's output
    print("\nOutput values (first 8 positions):")
    print("SDPA:  ", output_sdpa[0, 0, :8, 0].cpu().numpy())
    print("Manual:", output_manual[0, 0, :8, 0].cpu().numpy())


def demo_hilbert_with_sdpa():
    """Demonstrate Hilbert curve ordering with SDPA."""
    print("\nDemonstrating Hilbert ordering with SDPA...")

    try:
        from dilated_attention_pytorch.kernels.hilbert_attention_core import (
            create_hilbert_mapping,
        )
    except ImportError:
        print("Could not import Hilbert utilities, skipping demo")
        return

    seq_len = 64  # Small for visualization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create Hilbert indices
    hilbert_indices = create_hilbert_mapping(seq_len).to(device)

    # Create a simple attention pattern
    q = torch.eye(seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    k = q.clone()
    v = torch.arange(seq_len, device=device).float().view(1, 1, seq_len, 1)

    # Apply Hilbert ordering
    q_hilbert = q.index_select(2, hilbert_indices).index_select(3, hilbert_indices)
    k_hilbert = k.index_select(2, hilbert_indices).index_select(3, hilbert_indices)
    v_hilbert = v.index_select(2, hilbert_indices)

    # Compute attention
    output_hilbert = F.scaled_dot_product_attention(q_hilbert, k_hilbert, v_hilbert)

    # Reverse Hilbert ordering
    inverse_indices = torch.zeros_like(hilbert_indices)
    inverse_indices[hilbert_indices] = torch.arange(seq_len, device=device)
    output = output_hilbert.index_select(2, inverse_indices)

    print(f"Hilbert indices (first 16): {hilbert_indices[:16].cpu().numpy()}")
    print(f"Output shape: {output.shape}")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Visualize different mask patterns
    print("\nVisualizing attention mask patterns...")

    seq_len = 64
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Standard dilated mask
    mask1 = create_dilated_mask(seq_len, 2, device=device)
    ax = axes[0, 0]
    _ = ax.imshow(torch.where(mask1 == float("-inf"), 0, 1).cpu(), cmap="Blues")
    ax.set_title("Dilated Mask (rate=2)")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # Higher dilation rate
    mask2 = create_dilated_mask(seq_len, 4, device=device)
    ax = axes[0, 1]
    ax.imshow(torch.where(mask2 == float("-inf"), 0, 1).cpu(), cmap="Blues")
    ax.set_title("Dilated Mask (rate=4)")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # Segmented dilated mask
    mask3 = create_segmented_dilated_mask(
        seq_len, [16, 32, 16], [1, 2, 4], device=device
    )
    ax = axes[1, 0]
    ax.imshow(torch.where(mask3 == float("-inf"), 0, 1).cpu(), cmap="Blues")
    ax.set_title("Segmented Dilated Mask")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # Causal dilated mask
    mask4 = create_dilated_mask(seq_len, 2, device=device)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask4 = torch.where(causal_mask > 0, float("-inf"), mask4)
    ax = axes[1, 1]
    ax.imshow(torch.where(mask4 == float("-inf"), 0, 1).cpu(), cmap="Blues")
    ax.set_title("Causal Dilated Mask")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    plt.tight_layout()
    plt.savefig("sdpa_dilated_masks.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to sdpa_dilated_masks.png")

    # Run tests
    test_dilated_attention_correctness()

    # Benchmark if on GPU
    if device.type == "cuda":
        benchmark_sdpa_with_masks(device=device)

    # Hilbert demo
    demo_hilbert_with_sdpa()

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
