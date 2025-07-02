#!/bin/bash
# Run multi-GPU benchmark for V2 Collective

echo "Running V2 Collective Multi-GPU Benchmark..."
echo "=========================================="

# Check if we have multiple GPUs
n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $n_gpus GPU(s)"

if [ $n_gpus -lt 2 ]; then
    echo "Warning: This benchmark requires at least 2 GPUs for multi-GPU testing"
    echo "Running single-GPU benchmark only..."
fi

# Run the benchmark
cd /home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch
python benchmarks/benchmark_v2_collective_multi_gpu.py