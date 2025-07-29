#!/usr/bin/env python3
"""Test fused kernels for longer sequences to see if they provide benefits."""

import torch
import time
import sys
import gc
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dilated_attention_pytorch.kernels import UnifiedHilbertAttention


def benchmark_long_sequences():
    """Benchmark fused kernels on longer sequences."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test longer sequences
    sequence_lengths = [8192, 16384, 32768]

    print("Fused Kernel Performance on Long Sequences")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {torch.cuda.get_device_capability()}")
    print()

    for seq_len in sequence_lengths:
        print(f"\nSequence Length: {seq_len:,}")
        print("-" * 60)

        try:
            # Create module
            module = UnifiedHilbertAttention(
                hidden_dim=768,
                num_heads=12,
                segment_size=128,
                dilation_rate=1,
                dropout=0.0,
                hilbert_threshold=1024,
            ).to(device)

            # Create input
            x = torch.randn(1, seq_len, 768, device=device, dtype=torch.float16)

            # Force different backends
            configs = [
                ("PyTorch Baseline", False, "pytorch"),
                ("Triton Standard", True, "triton"),
                ("Fused Kernel", True, "fused"),
            ]

            results = {}

            for config_name, use_hilbert, backend in configs:
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Configure backend
                if backend == "pytorch":
                    module._triton_available = False
                    module._fused_kernels_available = False
                elif backend == "triton":
                    module._triton_available = True
                    module._fused_kernels_available = False
                elif backend == "fused":
                    module._triton_available = True
                    module._fused_kernels_available = True
                    # Temporarily adjust sequence range for fused kernels
                    original_forward = module.forward

                    def patched_forward(x, use_hilbert=True, is_causal=False):
                        # Modify the check to allow longer sequences
                        M = x.size(1)
                        original_check = "2048 <= M_padded <= 8192"
                        _ = "2048 <= M_padded <= 65536"  # Allow up to 64K

                        # Temporarily patch the forward method
                        import inspect

                        source = inspect.getsource(original_forward)
                        if original_check in source:
                            # This is a hack for testing - in production we'd modify the actual code
                            pass

                        # For now, directly call fused forward for testing
                        B, M, D = x.shape
                        M_padded = M if M % 128 == 0 else M + (128 - M % 128)

                        # Project to Q, K, V
                        qkv = module.qkv_proj(x)
                        qkv = qkv.view(B, M, 3, module.num_heads, module.head_dim)
                        qkv = qkv.permute(2, 0, 3, 1, 4)

                        # Call fused kernel directly
                        out = module._fused_forward(
                            qkv, M_padded, M, B, use_hilbert, is_causal
                        )

                        # Reshape and project output
                        out = out.transpose(1, 2).contiguous()
                        out = out.view(B, M_padded, D)
                        if M != M_padded:
                            out = out[:, :M, :]
                        out = module.out_proj(out)
                        return out

                    module.forward = patched_forward

                try:
                    with torch.no_grad():
                        # Warmup
                        for _ in range(2):
                            _ = module(x, use_hilbert=use_hilbert)

                        # Time
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        num_runs = 5
                        for _ in range(num_runs):
                            _ = module(x, use_hilbert=use_hilbert)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) / num_runs * 1000

                        # Memory usage
                        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

                        results[config_name] = (elapsed, memory_mb)

                except Exception as e:
                    results[config_name] = (None, None)
                    print(f"  {config_name:<20} Error: {str(e)}")
                    continue

                # Restore original settings
                if backend == "fused":
                    module.forward = original_forward
                module._triton_available = True
                module._fused_kernels_available = True

            # Print results
            baseline_time = results.get("PyTorch Baseline", (None, None))[0]

            for config_name, (time_ms, mem_mb) in results.items():
                if time_ms is not None:
                    if baseline_time is not None:
                        speedup = baseline_time / time_ms
                        print(
                            f"  {config_name:<20} {time_ms:>8.2f}ms ({speedup:>5.2f}x) - Memory: {mem_mb:>6.1f}MB"
                        )
                    else:
                        print(
                            f"  {config_name:<20} {time_ms:>8.2f}ms - Memory: {mem_mb:>6.1f}MB"
                        )

        except Exception as e:
            print(f"  Error testing sequence length {seq_len}: {str(e)}")

    # Theoretical analysis
    print("\n\nTheoretical Analysis:")
    print("=" * 80)
    print("\nFused Kernel Benefits:")
    print("- Reduced kernel launch overhead (significant for 2K-8K sequences)")
    print("- Better memory access patterns through larger tiles")
    print("- Potential for better occupancy with optimized configurations")

    print("\nFused Kernel Limitations for Very Long Sequences:")
    print("- Shared memory constraints limit tile sizes")
    print("- Diminishing returns as compute dominates over launch overhead")
    print("- May need different strategies (e.g., Flash Attention style tiling)")

    print("\nRecommendations:")
    print("- 2K-16K: Fused kernels likely beneficial")
    print("- 16K-64K: Depends on GPU architecture and memory")
    print("- 64K+: Consider specialized algorithms (Ring Attention, Flash Attention)")


if __name__ == "__main__":
    benchmark_long_sequences()
