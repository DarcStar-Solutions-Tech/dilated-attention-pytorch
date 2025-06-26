# Flash Attention 3 Setup Guide

Flash Attention 3 provides significant performance improvements over Flash Attention 2, especially on H100/H800 GPUs. This guide covers installation and setup.

## Performance Benefits

- **1.5-2.0x faster** than Flash Attention 2 with FP16
- **Up to 740 TFLOPS** (75% utilization of H100 theoretical max)
- **Up to 1.2 PFLOPS** with FP8 precision
- **2.6x smaller quantization error** than baseline FP8 attention
- **Asynchronous computation** with warp-specialization
- **Enhanced memory efficiency** through improved kernels

## Hardware Requirements

### Minimum Requirements
- **GPU**: H100 or H800 (Hopper architecture)
- **CUDA**: >= 12.3
- **PyTorch**: >= 1.12
- **Python**: >= 3.9

### Optimal Configuration
- **GPU**: H100 SXM (80GB) or H800
- **CUDA**: 12.4+
- **PyTorch**: 2.0+
- **System**: Linux with adequate cooling

## Installation Methods

### Method 1: Latest Stable Flash Attention 2.8+ (Recommended)

```bash
# Install latest stable version with FA3-like optimizations
uv pip install flash-attn>=2.8.0 --no-build-isolation

# Or with pip
pip install flash-attn>=2.8.0 --no-build-isolation
```

### Method 2: Flash Attention 3 Beta (Experimental)

⚠️ **Beta Release**: Only for testing/benchmarking

```bash
# Clone and install FA3 beta
git clone https://github.com/togethercomputer/flash-attention-3.git
cd flash-attention-3/hopper
python setup.py install

# Verify installation
python -c "import flash_attn_3; print('FA3 Beta installed successfully')"
```

### Method 3: Development Installation

```bash
# Install from source with latest optimizations
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation

# For CUDA 12.3+ with H100 optimizations
MAX_JOBS=4 pip install . --no-build-isolation
```

## Project Integration

The dilated-attention-pytorch project automatically detects and uses Flash Attention 3:

```bash
# Install project with FA3 support
uv pip install -e .[all]

# Test FA3 integration
python test_flash_attention_3.py
```

## Configuration Options

### Environment Variables

```bash
# Enable FA3 optimizations
export FLASH_ATTENTION_VERSION=3
export CUDA_VISIBLE_DEVICES=0

# For H100 systems
export FLASH_ATTENTION_FP8=1  # Enable FP8 (experimental)
export FLASH_ATTENTION_ASYNC=1  # Enable async optimizations
```

### PyTorch Configuration

```python
import torch
import torch.nn.functional as F

# Enable FA3 through SDPA backend
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    # Your attention computation
    output = F.scaled_dot_product_attention(q, k, v)
```

## Verification & Testing

### Check Installation

```python
from dilated_attention_pytorch.ring_dilated_attention import (
    get_flash_attention_version,
    is_flash_attention_3_available
)

print(f"Flash Attention Version: {get_flash_attention_version()}")
print(f"FA3 Available: {is_flash_attention_3_available()}")

# Check hardware optimization
import torch
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_properties(0).name
    print(f"GPU: {device_name}")
    print(f"H100 Optimized: {'H100' in device_name or 'H800' in device_name}")
```

### Performance Testing

```bash
# Run FA3 performance test
python test_flash_attention_3.py

# Benchmark comparison
hatch run benchmark:run --batch_size 2 --total_tokens 26 --heads 8
```

## Troubleshooting

### Common Issues

#### CUDA Version Mismatch
```bash
# Check CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# Ensure CUDA >= 12.3 for FA3
```

#### Compilation Errors
```bash
# Install with specific CUDA architecture
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install flash-attn --no-build-isolation

# For H100 specifically
TORCH_CUDA_ARCH_LIST="9.0" pip install flash-attn --no-build-isolation
```

#### Memory Issues
```bash
# Reduce compilation memory usage
MAX_JOBS=1 pip install flash-attn --no-build-isolation

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Hardware-Specific Notes

#### H100 Systems
- Enable Tensor Core utilization: `export CUDA_DEVICE_MAX_CONNECTIONS=1`
- Use mixed precision: FP16 or BF16
- Consider FP8 for maximum throughput (experimental)

#### Non-H100 Systems
- FA3 features limited on non-Hopper architectures
- FA2.8+ still provides significant improvements
- Falls back to optimized FA2 kernels

## Performance Tuning

### Optimal Settings for H100

```python
# Ring Attention with FA3 optimizations
ring_attention = RingDilatedAttention(
    segment_lengths=[2048, 4096, 8192],
    dilation_rates=[1, 2, 4],
    dropout=0.0,
    use_tf32=True,           # Enable TF32
    block_size=512,          # Optimized for H100
    use_flash_attention=True,
    device='cuda'
)
```

### Memory Optimization

```python
# Enable gradient checkpointing
ring_attention = RingDilatedAttention(
    # ... other params
    use_checkpointing=True,
    memory_efficient=True
)

# Use mixed precision
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = ring_attention(q, k, v)
```

## Integration with Training

### DeepSpeed Integration

```python
# deepspeed_config.json
{
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    },
    "flash_attention": {
        "enabled": true,
        "version": 3
    }
}
```

### Distributed Training

```bash
# Multi-GPU training with FA3
torchrun --nproc_per_node=8 train.py \
    --use_flash_attention_3 \
    --fp16 \
    --gradient_checkpointing
```

## Monitoring & Profiling

### GPU Utilization

```bash
# Monitor H100 utilization
nvidia-smi -l 1

# Detailed profiling
nsys profile python test_flash_attention_3.py
```

### Memory Usage

```python
# Monitor memory usage
import torch

def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Call before/after attention computation
print_memory_stats()
```

## Future Roadmap

### Upcoming Features (FA3)
- **BF16 support**: Better numerical stability
- **Variable length support**: Dynamic sequence lengths
- **FP8 forward pass**: Maximum throughput
- **Multi-query attention**: Optimized for inference
- **Distributed attention**: Seamless multi-GPU support

### Integration Timeline
- **Q1 2025**: Stable FA3 release
- **Q2 2025**: PyTorch native integration
- **Q3 2025**: Framework-agnostic support

## Resources

- [Flash Attention 3 Repository](https://github.com/togethercomputer/flash-attention-3)
- [Flash Attention 2 Repository](https://github.com/Dao-AILab/flash-attention)
- [PyTorch Blog: FlashAttention-3](https://pytorch.org/blog/flashattention-3/)
- [H100 Architecture Guide](https://docs.nvidia.com/cuda/hopper-architecture/)

## Support

For Flash Attention 3 specific issues:
- [FA3 GitHub Issues](https://github.com/togethercomputer/flash-attention-3/issues)
- [FA2 GitHub Issues](https://github.com/Dao-AILab/flash-attention/issues)

For integration with this project:
- Open an issue in this repository
- Include GPU model, CUDA version, and error logs