#!/usr/bin/env python3
"""
Test script to verify Triton Hilbert attention backward pass performance issue.
"""

import torch
import time
import gc
from contextlib import contextmanager


@contextmanager
def timer(name):
    """Simple timer context manager."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{name}: {elapsed:.2f}ms")


def test_backward_performance():
    """Test backward pass performance of different implementations."""

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires GPU.")
        return

    device = torch.device("cuda")
    print(f"Testing on: {torch.cuda.get_device_name()}")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    seq_len = 4096
    num_heads = 12
    head_dim = 64

    # Test configurations
    configs = [
        {
            "name": "ImprovedDilatedAttention",
            "module": "dilated_attention_pytorch",
            "class": "ImprovedDilatedAttention",
            "kwargs": {
                "segment_lengths": [1024, 2048, 4096],
                "dilation_rates": [1, 2, 4],
                "dropout": 0.0,
            },
            "input_format": "4d",
        },
        {
            "name": "HilbertAttentionTritonFixed",
            "module": "dilated_attention_pytorch.kernels",
            "class": "HilbertAttentionTritonFixed",
            "kwargs": {
                "segment_lengths": [1024, 2048, 4096],
                "dilation_rates": [1, 2, 4],
                "dropout": 0.0,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
            "input_format": "4d",
        },
    ]

    print(
        f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}"
    )
    print("-" * 60)

    for config in configs:
        print(f"\nTesting {config['name']}...")

        try:
            # Import and create model
            module = __import__(config["module"], fromlist=[config["class"]])
            model_class = getattr(module, config["class"])
            model = model_class(**config["kwargs"]).to(device)

            # Create inputs
            q = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(q, k, v)

            # Test forward pass
            with timer("  Forward pass"):
                output = model(q, k, v)

            # Test backward pass
            loss = output.sum()

            # Clear gradients
            if q.grad is not None:
                q.grad.zero_()
                k.grad.zero_()
                v.grad.zero_()

            with timer("  Backward pass"):
                loss.backward()

            # Calculate ratio
            # Re-run to get accurate timing
            torch.cuda.synchronize()
            fwd_start = time.perf_counter()
            output = model(q, k, v)
            torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - fwd_start) * 1000

            loss = output.sum()
            if q.grad is not None:
                q.grad.zero_()
                k.grad.zero_()
                v.grad.zero_()

            torch.cuda.synchronize()
            bwd_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            bwd_time = (time.perf_counter() - bwd_start) * 1000

            ratio = bwd_time / fwd_time
            print(f"  Backward/Forward ratio: {ratio:.2f}x")

            # Memory usage
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  Peak memory: {peak_memory:.1f} MB")

        except Exception as e:
            print(f"  Failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Analysis complete!")

    # Additional analysis for Triton implementation
    print("\nTriton-specific analysis:")
    print("- Triton kernels lack custom backward pass implementation")
    print("- Hilbert reordering creates non-contiguous memory access")
    print("- PyTorch autograd must trace through all operations")
    print("- No gradient checkpointing or Flash Attention integration")


if __name__ == "__main__":
    test_backward_performance()
