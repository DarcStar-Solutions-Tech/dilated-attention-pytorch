#!/bin/bash
# Run comprehensive dilated vs standard ring attention benchmarks

echo "=================================================="
echo "Dilated vs Standard Ring Attention Benchmarks"
echo "=================================================="

# Single GPU benchmark
echo -e "\n1. Running single GPU comparison..."
python dilated_vs_standard_ring_benchmark.py \
    --seq-lengths 1024 2048 4096 8192 \
    --segment-lengths 256 512 1024 \
    --dilation-rates 1 2 4

# Multi-GPU benchmark (if available)
if command -v torchrun &> /dev/null && [ $(nvidia-smi -L | wc -l) -ge 2 ]; then
    echo -e "\n2. Running multi-GPU comparison (2 GPUs)..."
    torchrun --nproc_per_node=2 dilated_vs_standard_ring_benchmark.py \
        --seq-lengths 2048 4096 8192 16384 \
        --segment-lengths 512 1024 2048 \
        --dilation-rates 1 2 4
fi

# Quick memory test
echo -e "\n3. Running memory scaling test..."
python dilated_vs_standard_ring_benchmark.py \
    --seq-lengths 1024 2048 4096 \
    --batch-size 1 \
    --output-dir benchmarks/results/ring/memory_test

echo -e "\n=================================================="
echo "Benchmark complete! Check results in:"
echo "  benchmarks/results/ring/dilated_vs_standard/"
echo "=================================================="