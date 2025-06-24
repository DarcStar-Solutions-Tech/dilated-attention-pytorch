#!/usr/bin/env python3
"""
Simple comparison of DilatedAttention vs ImprovedDilatedAttention functionality
"""
import torch
import torch.nn.functional as F
from einops import rearrange

# Create simplified versions without xformers dependency
class SimpleDilatedAttention(torch.nn.Module):
    """Simplified version of DilatedAttention without xformers"""
    def __init__(self, segment_lengths, dilation_rates, dropout=0.0):
        super().__init__()
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError("segment_lengths and dilation_rates must have the same length")
        
        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.dropout_p = dropout
        self.num_groups = len(self.dilation_rates)

    def forward(self, query, key, value, is_causal=False):
        b, n, h, d = query.shape
        out = torch.zeros_like(query)
        
        # Distribute heads across groups
        group_sizes = [h // self.num_groups] * self.num_groups
        for i in range(h % self.num_groups):
            group_sizes[i] += 1

        for i, (g, r, s) in enumerate(zip(group_sizes, self.dilation_rates, self.segment_lengths)):
            # Split sequences into segments
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            
            # Apply dilation and segment offset
            offset = i % r
            hmin = sum(group_sizes[:i])
            hmax = hmin + g
            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            
            # Fold segments into batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")
            
            # Apply attention using PyTorch's scaled_dot_product_attention
            x = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal
            )
            
            # Unfold segments back
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)
            
            # Normalize attention outputs
            x = x / (x.sum(dim=(1, 2), keepdim=True) + 1e-8)
            
            # Gather attention outputs
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d")
        
        return out / self.num_groups

class SimpleImprovedDilatedAttention(torch.nn.Module):
    """Simplified version of ImprovedDilatedAttention"""
    def __init__(self, segment_lengths, dilation_rates, dropout=0.0, use_tf32=True):
        super().__init__()
        assert len(segment_lengths) == len(dilation_rates)
        self.seg = segment_lengths
        self.dil = dilation_rates
        self.drop = dropout
        self.num_groups = len(self.seg)
        if use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, q, k, v, is_causal=False):
        b, n, h, d = q.shape
        device = q.device
        out = torch.zeros_like(q)

        # Distribute heads
        gs = [h // self.num_groups] * self.num_groups
        for i in range(h % self.num_groups): 
            gs[i] += 1

        for i, (g, r, s) in enumerate(zip(gs, self.dil, self.seg)):
            if n < s: 
                continue  # skip large segments if too small

            offset = i % r
            hmin = sum(gs[:i])
            hmax = hmin + g

            # Segment + dilation slicing
            q_seg = rearrange(q[..., hmin:hmax, :], 'b (n s) g d -> (b n) s g d', s=s)
            k_seg = rearrange(k[..., hmin:hmax, :], 'b (n s) g d -> (b n) s g d', s=s)
            v_seg = rearrange(v[..., hmin:hmax, :], 'b (n s) g d -> (b n) s g d', s=s)

            if r > 1 or offset:
                idx = torch.arange(offset, s, r, device=device)
                q_seg = q_seg[:, idx]
                k_seg = k_seg[:, idx]
                v_seg = v_seg[:, idx]

            # Attention
            x = F.scaled_dot_product_attention(
                q_seg, k_seg, v_seg,
                attn_mask=None,
                dropout_p=self.drop if self.training else 0.0,
                is_causal=is_causal
            )

            # Normalize by local attention norm
            denom = x.sum(dim=(1, 2), keepdim=True) + 1e-8
            x = x / denom

            # Scatter results back
            x = rearrange(x, '(b n) s g d -> b (n s) g d', b=b, s=x.shape[1])
            out[..., hmin:hmax, :] += rearrange(x, 'b (n s) g d -> b (n s) g d', s=s)

        return out / self.num_groups

def test_functionality():
    """Test if both implementations produce similar outputs"""
    print("Testing functional equivalence...")
    
    # Test parameters
    batch_size = 2
    seq_len = 4096
    num_heads = 8
    embed_dim = 64
    segment_lengths = [1024, 2048]
    dilation_rates = [1, 2]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # Create test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, embed_dim, device=device, dtype=dtype)
    
    # Test both implementations
    model1 = SimpleDilatedAttention(segment_lengths, dilation_rates).to(device)
    model2 = SimpleImprovedDilatedAttention(segment_lengths, dilation_rates).to(device)
    
    with torch.no_grad():
        out1 = model1(q, k, v)
        out2 = model2(q, k, v)
    
    # Compare outputs
    print(f"Output 1 shape: {out1.shape}")
    print(f"Output 2 shape: {out2.shape}")
    print(f"Output 1 mean: {out1.mean().item():.6f}")
    print(f"Output 2 mean: {out2.mean().item():.6f}")
    print(f"Output 1 std: {out1.std().item():.6f}")
    print(f"Output 2 std: {out2.std().item():.6f}")
    
    # Check if outputs are similar
    mse = F.mse_loss(out1, out2).item()
    print(f"MSE between outputs: {mse:.8f}")
    
    if mse < 1e-4:
        print("✓ Outputs are functionally equivalent")
    else:
        print("✗ Outputs differ significantly")
    
    return mse < 1e-4

def analyze_code_differences():
    """Analyze key differences between implementations"""
    print("\n" + "="*80)
    print("CODE STRUCTURE ANALYSIS")
    print("="*80)
    
    differences = {
        "Initialization": {
            "DilatedAttention": [
                "Uses ValueError for validation",
                "Stores softmax_scale and op parameters",
                "More parameter validation"
            ],
            "ImprovedDilatedAttention": [
                "Uses assert for validation",
                "Enables TF32 optimization",
                "Simpler parameter storage"
            ]
        },
        "Forward Pass": {
            "DilatedAttention": [
                "Explicit head group size calculation with remainder handling",
                "Step-by-step tensor operations",
                "Manual attention normalization",
                "More explicit memory management"
            ],
            "ImprovedDilatedAttention": [
                "Streamlined head distribution",
                "Early exit for oversized segments",
                "Optimized tensor indexing",
                "More concise operations"
            ]
        },
        "Performance Features": {
            "DilatedAttention": [
                "Relies on xformers for optimization",
                "Manual softmax scaling",
                "Explicit dropout handling"
            ],
            "ImprovedDilatedAttention": [
                "TF32 optimization",
                "Torch.compile support",
                "Automatic backend selection"
            ]
        }
    }
    
    for category, items in differences.items():
        print(f"\n{category}:")
        print("-" * 40)
        for impl, features in items.items():
            print(f"{impl}:")
            for feature in features:
                print(f"  • {feature}")

def memory_complexity_analysis():
    """Analyze memory complexity of both implementations"""
    print("\n" + "="*80)
    print("MEMORY COMPLEXITY ANALYSIS")
    print("="*80)
    
    print("DilatedAttention:")
    print("• Memory allocation: O(b * n * h * d) for output tensor")
    print("• Intermediate tensors: Multiple rearrange operations create temporary tensors")
    print("• Segment processing: Sequential processing with explicit cleanup")
    print("• Peak memory: Higher due to multiple intermediate tensors")
    
    print("\nImprovedDilatedAttention:")
    print("• Memory allocation: O(b * n * h * d) for output tensor")  
    print("• Intermediate tensors: More efficient tensor operations")
    print("• Segment processing: Early exit reduces unnecessary computations")
    print("• Peak memory: Lower due to optimized operations")
    
    print("\nKey Memory Differences:")
    print("• ImprovedDilatedAttention uses more efficient tensor slicing")
    print("• Early exit in ImprovedDilatedAttention saves memory for large segments")
    print("• Both have similar theoretical complexity but different practical usage")

def runtime_complexity_analysis():
    """Analyze runtime complexity of both implementations"""
    print("\n" + "="*80)
    print("RUNTIME COMPLEXITY ANALYSIS")
    print("="*80)
    
    print("Both implementations have similar theoretical complexity:")
    print("• Time complexity: O(b * n * h * d * num_segments)")
    print("• Attention computation: O(s^2) per segment where s is segment length")
    print("• Total attention cost: O(sum(s_i^2) * num_heads)")
    
    print("\nPractical Performance Differences:")
    print("DilatedAttention:")
    print("• Relies on xformers optimization")
    print("• More tensor operations and memory transfers")
    print("• Explicit normalization steps")
    
    print("\nImprovedDilatedAttention:")
    print("• TF32 acceleration (when available)")
    print("• Torch.compile optimization")
    print("• More efficient tensor operations")
    print("• Early exit for oversized segments")
    print("• Automatic backend selection for attention")

if __name__ == "__main__":
    print("Comparing DilatedAttention vs ImprovedDilatedAttention")
    print("="*60)
    
    # Test functionality
    try:
        is_equivalent = test_functionality()
    except Exception as e:
        print(f"Error in functionality test: {e}")
        is_equivalent = False
    
    # Analyze differences
    analyze_code_differences()
    memory_complexity_analysis()
    runtime_complexity_analysis()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Functional Equivalence: {'Yes' if is_equivalent else 'No'}")
    print("Key Improvements in ImprovedDilatedAttention:")
    print("• TF32 optimization for faster computation")
    print("• Torch.compile support for graph optimization")
    print("• More efficient tensor operations")
    print("• Early exit for oversized segments")
    print("• Cleaner, more maintainable code")
    print("• Better memory efficiency")