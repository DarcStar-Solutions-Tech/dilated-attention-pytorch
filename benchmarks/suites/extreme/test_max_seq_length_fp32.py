#!/usr/bin/env python3
"""
Test maximum sequence length for each implementation with FP32.
Binary search to find the longest sequence that fits in memory.
"""

import torch
import gc
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

from dilated_attention_pytorch import (
    DilatedAttention,
    MultiheadDilatedAttention,
    ImprovedMultiheadDilatedAttention,
    create_multihead_dilated_attention,
)


class MaxSequenceTester:
    def __init__(self, device: torch.device):
        self.device = device
        self.dtype = torch.float32

    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory stats in MB."""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**2,
                "reserved": torch.cuda.memory_reserved() / 1024**2,
                "free": (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_allocated()
                )
                / 1024**2,
            }
        return {"allocated": 0, "reserved": 0, "free": 0}

    def test_sequence_length(
        self,
        model_fn,
        seq_len: int,
        batch_size: int = 1,
        num_heads: int = 8,
        head_dim: int = 64,
        is_multihead: bool = False,
    ) -> Tuple[bool, Optional[Dict]]:
        """Test if a sequence length works."""

        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()

        try:
            # Get initial memory
            mem_before = self.get_memory_info()

            # Create model
            model = model_fn(seq_len)

            # Create inputs
            embed_dim = num_heads * head_dim
            if is_multihead:
                x = torch.randn(
                    batch_size, seq_len, embed_dim, device=self.device, dtype=self.dtype
                )
                inputs = (x, x, x)
            else:
                q = torch.randn(
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                inputs = (q, k, v)

            # Forward pass
            with torch.no_grad():
                output = model(*inputs)
                if isinstance(output, tuple):
                    output = output[0]

            # Get memory after
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mem_after = self.get_memory_info()

            # Calculate memory used
            memory_used = mem_after["allocated"] - mem_before["allocated"]

            # Cleanup
            del model, output, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            return True, {
                "seq_len": seq_len,
                "memory_used_mb": memory_used,
                "memory_free_mb": mem_after["free"],
            }

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            return False, None

    def binary_search_max_length(
        self,
        model_name: str,
        model_fn,
        min_len: int = 1024,
        max_len: int = 1024 * 1024,  # 1M tokens
        step: int = 1024,
        **kwargs,
    ) -> Dict:
        """Binary search for maximum sequence length."""

        print(f"\n{'=' * 60}")
        print(f"Testing {model_name}")
        print(f"{'=' * 60}")

        # First, find a working minimum
        current = min_len
        while current <= max_len:
            success, info = self.test_sequence_length(model_fn, current, **kwargs)
            if success:
                print(
                    f"  ✓ {current:,} tokens - Memory: {info['memory_used_mb']:.1f} MB"
                )
                last_working = current
                last_info = info
                break
            else:
                print(f"  ✗ {current:,} tokens - OOM")
                return {
                    "model": model_name,
                    "max_seq_len": 0,
                    "error": f"Cannot even run {current} tokens",
                }
            current += step

        # Binary search for maximum
        low = last_working
        high = max_len
        best_length = last_working
        best_info = last_info

        while low <= high:
            mid = ((low + high) // (2 * step)) * step  # Round to nearest step

            if mid == best_length:  # Avoid infinite loop
                mid += step

            if mid > high:
                break

            success, info = self.test_sequence_length(model_fn, mid, **kwargs)

            if success:
                print(
                    f"  ✓ {mid:,} tokens - Memory: {info['memory_used_mb']:.1f} MB, Free: {info['memory_free_mb']:.1f} MB"
                )
                best_length = mid
                best_info = info
                low = mid + step
            else:
                print(f"  ✗ {mid:,} tokens - OOM")
                high = mid - step

        # Try to squeeze out a bit more
        print("\nFine-tuning search...")
        current = best_length + step
        while current <= best_length + 10 * step:
            success, info = self.test_sequence_length(model_fn, current, **kwargs)
            if success:
                print(
                    f"  ✓ {current:,} tokens - Memory: {info['memory_used_mb']:.1f} MB"
                )
                best_length = current
                best_info = info
                current += step
            else:
                print(f"  ✗ {current:,} tokens - OOM (limit reached)")
                break

        return {
            "model": model_name,
            "max_seq_len": best_length,
            "memory_used_mb": best_info["memory_used_mb"],
            "memory_free_mb": best_info["memory_free_mb"],
            "parameters": {
                "batch_size": kwargs.get("batch_size", 1),
                "num_heads": kwargs.get("num_heads", 8),
                "head_dim": kwargs.get("head_dim", 64),
            },
        }


def create_model_functions(device):
    """Create model factory functions."""

    def create_dilated_attention(seq_len):
        # Adaptive segments based on sequence length
        if seq_len <= 4096:
            segments = [min(1024, seq_len // 4), min(2048, seq_len // 2)]
            dilations = [1, 2]
        elif seq_len <= 16384:
            segments = [2048, 4096, 8192]
            dilations = [1, 2, 4]
        else:
            segments = [4096, 8192, 16384, 32768]
            dilations = [1, 2, 4, 8]

        # Ensure sequence length is divisible by largest segment
        max_segment = max(segments)
        if seq_len % max_segment != 0:
            segments = [s for s in segments if seq_len % s == 0]
            if not segments:
                segments = [1024]  # Fallback
            dilations = dilations[: len(segments)]

        return DilatedAttention(
            segment_lengths=segments, dilation_rates=dilations, attention_dropout=0.0
        ).to(device)

    def create_multihead_dilated(seq_len):
        if seq_len <= 4096:
            segments = [min(1024, seq_len // 4), min(2048, seq_len // 2)]
            dilations = [1, 2]
        elif seq_len <= 16384:
            segments = [2048, 4096, 8192]
            dilations = [1, 2, 4]
        else:
            segments = [4096, 8192, 16384, 32768]
            dilations = [1, 2, 4, 8]

        max_segment = max(segments)
        if seq_len % max_segment != 0:
            segments = [s for s in segments if seq_len % s == 0]
            if not segments:
                segments = [1024]
            dilations = dilations[: len(segments)]

        return MultiheadDilatedAttention(
            embed_dim=512,  # 8 heads * 64 dim
            num_heads=8,
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
        ).to(device)

    def create_improved_multihead(seq_len):
        if seq_len <= 4096:
            segments = [min(1024, seq_len // 4), min(2048, seq_len // 2)]
            dilations = [1, 2]
        elif seq_len <= 16384:
            segments = [2048, 4096, 8192]
            dilations = [1, 2, 4]
        else:
            segments = [4096, 8192, 16384, 32768]
            dilations = [1, 2, 4, 8]

        max_segment = max(segments)
        if seq_len % max_segment != 0:
            segments = [s for s in segments if seq_len % s == 0]
            if not segments:
                segments = [1024]
            dilations = dilations[: len(segments)]

        return ImprovedMultiheadDilatedAttention(
            embed_dim=512,
            num_heads=8,
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
        ).to(device)

    def create_factory_auto(seq_len):
        if seq_len <= 4096:
            segments = [min(1024, seq_len // 4), min(2048, seq_len // 2)]
            dilations = [1, 2]
        elif seq_len <= 16384:
            segments = [2048, 4096, 8192]
            dilations = [1, 2, 4]
        else:
            segments = [4096, 8192, 16384, 32768]
            dilations = [1, 2, 4, 8]

        max_segment = max(segments)
        if seq_len % max_segment != 0:
            segments = [s for s in segments if seq_len % s == 0]
            if not segments:
                segments = [1024]
            dilations = dilations[: len(segments)]

        return create_multihead_dilated_attention(
            "auto",
            embed_dim=512,
            num_heads=8,
            segment_lengths=segments,
            dilation_rates=dilations,
            dropout=0.0,
        ).to(device)

    return {
        "DilatedAttention": (create_dilated_attention, False),
        "MultiheadDilatedAttention": (create_multihead_dilated, True),
        "ImprovedMultiheadDilatedAttention": (create_improved_multihead, True),
        "Factory-Auto": (create_factory_auto, True),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("CUDA not available. This test requires GPU.")
        return

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Clear any existing allocations
    torch.cuda.empty_cache()
    gc.collect()

    mem_info = MaxSequenceTester(device).get_memory_info()
    print(f"Available Memory: {mem_info['free']:.1f} MB")

    tester = MaxSequenceTester(device)
    model_fns = create_model_functions(device)

    results = []

    # Test each implementation
    for model_name, (model_fn, is_multihead) in model_fns.items():
        result = tester.binary_search_max_length(
            model_name,
            model_fn,
            min_len=2048,
            max_len=256 * 1024,  # 256K max
            step=2048,
            batch_size=1,
            num_heads=8,
            head_dim=64,
            is_multihead=is_multihead,
        )
        results.append(result)

        # Clear memory between tests
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H%M-UTC")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            "dtype": "float32",
        },
        "results": results,
    }

    filename = f"benchmarks/max_seq_length_fp32_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("MAXIMUM SEQUENCE LENGTH SUMMARY (FP32)")
    print("=" * 60)
    print(f"{'Implementation':<35} {'Max Seq Length':>15} {'Memory Used':>12}")
    print("-" * 65)

    for result in results:
        if result["max_seq_len"] > 0:
            print(
                f"{result['model']:<35} {result['max_seq_len']:>15,} "
                f"{result['memory_used_mb']:>11.1f} MB"
            )
        else:
            print(f"{result['model']:<35} {'Failed':>15} {'N/A':>12}")

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
