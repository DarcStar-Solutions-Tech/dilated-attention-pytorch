"""
Test the unfold-based RingDilatedAttention implementation
"""
import torch
import time
from dilated_attention_pytorch import RingDilatedAttention, UnfoldRingDilatedAttention
from dilated_attention_pytorch.ring_dilated_attention_unfold_optimized import OptimizedUnfoldRingDilatedAttention

def test_correctness():
    """Verify that unfold implementation produces same results as original."""
    print("Testing Correctness of Unfold Implementation")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32  # Use float32 for better numerical comparison
    
    # Test parameters
    test_configs = [
        # (batch_size, seq_len, num_heads, head_dim, segments, dilations)
        (1, 1024, 8, 64, [256, 512, 1024], [1, 2, 4]),
        (2, 2048, 12, 64, [512, 1024, 2048], [1, 2, 4]),
        (1, 4096, 16, 64, [1024, 2048, 4096], [1, 2, 4]),
    ]
    
    for batch_size, seq_len, num_heads, head_dim, segments, dilations in test_configs:
        print(f"\nTesting: batch={batch_size}, seq_len={seq_len}, heads={num_heads}")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # Create modules
        orig_module = RingDilatedAttention(segments, dilations, 0.0, ring_size=1).to(device, dtype)
        unfold_module = UnfoldRingDilatedAttention(segments, dilations, 0.0, ring_size=1).to(device, dtype)
        
        # Set to eval mode for consistent results
        orig_module.eval()
        unfold_module.eval()
        
        # Forward pass
        with torch.no_grad():
            orig_out = orig_module(q, k, v)
            unfold_out = unfold_module(q, k, v)
        
        # Compare results
        diff = (orig_out - unfold_out).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        # Check if results match within tolerance
        tolerance = 1e-5
        if torch.allclose(orig_out, unfold_out, rtol=tolerance, atol=tolerance):
            print("  ✓ Results match!")
        else:
            print("  ✗ Results differ beyond tolerance!")
            # Debug: show where differences occur
            large_diffs = (diff > tolerance).sum().item()
            print(f"  Number of elements with large differences: {large_diffs}")
        
        # Cleanup
        del q, k, v, orig_out, unfold_out
        torch.cuda.empty_cache()


def benchmark_performance():
    """Benchmark the performance improvement of unfold implementation."""
    print("\n\nBenchmarking Performance")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    # Benchmark configurations
    benchmark_configs = [
        (1, 2048, 8, 64, [512, 1024, 2048], [1, 2, 4]),
        (1, 8192, 12, 64, [2048, 4096, 8192], [1, 2, 4]),
        (1, 16384, 16, 64, [4096, 8192, 16384], [1, 2, 4]),
    ]
    
    for batch_size, seq_len, num_heads, head_dim, segments, dilations in benchmark_configs:
        print(f"\nSequence length: {seq_len:,} tokens")
        
        # Create inputs
        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        
        # Create modules
        orig_module = RingDilatedAttention(segments, dilations, 0.0, ring_size=1).to(device, dtype)
        unfold_module = UnfoldRingDilatedAttention(segments, dilations, 0.0, ring_size=1).to(device, dtype)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = orig_module(q, k, v)
                _ = unfold_module(q, k, v)
        
        # Benchmark original
        torch.cuda.synchronize()
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            with torch.no_grad():
                _ = orig_module(q, k, v)
        torch.cuda.synchronize()
        orig_time = (time.time() - start) / iterations * 1000
        
        # Benchmark unfold
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iterations):
            with torch.no_grad():
                _ = unfold_module(q, k, v)
        torch.cuda.synchronize()
        unfold_time = (time.time() - start) / iterations * 1000
        
        # Results
        speedup = orig_time / unfold_time
        print(f"  Original (index_select): {orig_time:.2f}ms")
        print(f"  Unfold implementation: {unfold_time:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Memory usage comparison
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = orig_module(q, k, v)
            orig_memory = torch.cuda.max_memory_allocated() / (1024**2)
            
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = unfold_module(q, k, v)
            unfold_memory = torch.cuda.max_memory_allocated() / (1024**2)
            
            print(f"  Memory - Original: {orig_memory:.1f}MB, Unfold: {unfold_memory:.1f}MB")
            print(f"  Memory reduction: {(1 - unfold_memory/orig_memory)*100:.1f}%")
        
        # Cleanup
        del q, k, v, orig_module, unfold_module
        torch.cuda.empty_cache()


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\n\nTesting Edge Cases")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    
    # Edge case 1: Non-divisible sequence lengths
    print("\n1. Non-divisible sequence lengths:")
    segments = [256, 512, 1024]
    dilations = [1, 2, 4]
    
    for seq_len in [1000, 1500, 3000]:  # Not divisible by largest segment
        print(f"  Testing seq_len={seq_len}")
        
        q = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype)
        k = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype)
        v = torch.randn(1, seq_len, 8, 64, device=device, dtype=dtype)
        
        orig = RingDilatedAttention(segments, dilations, 0.0).to(device, dtype)
        unfold = UnfoldRingDilatedAttention(segments, dilations, 0.0).to(device, dtype)
        
        with torch.no_grad():
            orig_out = orig(q, k, v)
            unfold_out = unfold(q, k, v)
        
        if torch.allclose(orig_out, unfold_out, rtol=1e-5):
            print("    ✓ Pass")
        else:
            print(f"    ✗ Fail - max diff: {(orig_out - unfold_out).abs().max().item():.2e}")
    
    # Edge case 2: Different dilation offsets
    print("\n2. Testing different dilation offsets:")
    # The offset depends on head group index
    # We'll test with different numbers of heads to trigger different offsets
    
    for num_heads in [3, 6, 9, 12]:  # Different head counts
        print(f"  Testing num_heads={num_heads}")
        
        q = torch.randn(1, 1024, num_heads, 64, device=device, dtype=dtype)
        k = torch.randn(1, 1024, num_heads, 64, device=device, dtype=dtype)
        v = torch.randn(1, 1024, num_heads, 64, device=device, dtype=dtype)
        
        orig = RingDilatedAttention(segments, dilations, 0.0).to(device, dtype)
        unfold = UnfoldRingDilatedAttention(segments, dilations, 0.0).to(device, dtype)
        
        with torch.no_grad():
            orig_out = orig(q, k, v)
            unfold_out = unfold(q, k, v)
        
        if torch.allclose(orig_out, unfold_out, rtol=1e-5):
            print("    ✓ Pass")
        else:
            print(f"    ✗ Fail - max diff: {(orig_out - unfold_out).abs().max().item():.2e}")


if __name__ == "__main__":
    test_correctness()
    benchmark_performance()
    test_edge_cases()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("The unfold-based implementation should provide significant speedups")
    print("while maintaining numerical accuracy with the original implementation.")