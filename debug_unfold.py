"""Debug unfold implementation"""
import torch

# Test tensor unfold behavior
print("Testing unfold behavior:")
print("=" * 40)

# Create test tensor
b, n, h, d = 1, 8, 2, 4
x = torch.arange(b * n * h * d).reshape(b, n, h, d).float()
print(f"Original shape: {x.shape}")
print(f"Original:\n{x[0, :, 0, 0]}")  # Show first head, first dim

# Test unfold with different parameters
# unfold(dimension, size, step)
dilation_rate = 2

# Method 1: Direct unfold on sequence dimension
x_unfold = x.unfold(1, 1, dilation_rate)
print(f"\nUnfold shape: {x_unfold.shape}")
print(f"After unfold, squeeze: {x_unfold.squeeze(-1).shape}")

# Test segment + unfold
segment_size = 4
num_segments = n // segment_size
x_seg = x.view(b, num_segments, segment_size, h, d)
print(f"\nSegmented shape: {x_seg.shape}")

# Apply unfold to segments
x_seg_unfold = x_seg.unfold(2, 1, dilation_rate).squeeze(-1)
print(f"Segment unfold shape: {x_seg_unfold.shape}")
print(f"Values: {x_seg_unfold[0, :, :, 0, 0]}")

# What we actually need:
# From (b, num_segments, segment_size, h, d) with dilation
# To (b, num_segments, segment_size//dilation, h, d)