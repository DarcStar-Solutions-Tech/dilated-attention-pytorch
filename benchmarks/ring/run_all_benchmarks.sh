#!/bin/bash
# Run all ring attention benchmarks

echo "Starting Ring Attention Benchmark Suite"
echo "======================================"

# Create results directory
mkdir -p benchmarks/results/ring

# 1. Single GPU comparison
echo -e "\n1. Running single GPU comparison..."
python benchmarks/ring/single_gpu_comparison.py

# 2. Memory scaling analysis
echo -e "\n2. Running memory scaling analysis..."
python benchmarks/ring/memory_scaling_analysis.py

# 3. Comprehensive benchmark (single GPU)
echo -e "\n3. Running comprehensive benchmark (1 GPU)..."
python benchmarks/ring/comprehensive_ring_benchmark.py \
    --seq-lengths 1024 2048 4096 8192 16384 \
    --batch-sizes 1 2 \
    --output-dir benchmarks/results/ring/single

# 4. Multi-GPU benchmarks (if available)
if command -v torchrun &> /dev/null; then
    # Check if we have multiple GPUs
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    
    if [ "$GPU_COUNT" -gt 1 ]; then
        echo -e "\n4. Running multi-GPU benchmarks ($GPU_COUNT GPUs)..."
        
        # 2 GPU benchmark
        if [ "$GPU_COUNT" -ge 2 ]; then
            echo -e "\n   Running 2 GPU benchmark..."
            torchrun --nproc_per_node=2 \
                benchmarks/ring/comprehensive_ring_benchmark.py \
                --seq-lengths 4096 8192 16384 32768 \
                --output-dir benchmarks/results/ring/2gpu
        fi
        
        # 4 GPU benchmark
        if [ "$GPU_COUNT" -ge 4 ]; then
            echo -e "\n   Running 4 GPU benchmark..."
            torchrun --nproc_per_node=4 \
                benchmarks/ring/comprehensive_ring_benchmark.py \
                --seq-lengths 8192 16384 32768 65536 \
                --output-dir benchmarks/results/ring/4gpu
        fi
    else
        echo -e "\n4. Skipping multi-GPU benchmarks (only $GPU_COUNT GPU available)"
    fi
else
    echo -e "\n4. Skipping multi-GPU benchmarks (torchrun not available)"
fi

# 5. Extreme sequence benchmark (conservative)
echo -e "\n5. Running extreme sequence benchmark (up to 100K tokens)..."
python benchmarks/ring/extreme_sequence_benchmark.py \
    --max-length 100000 \
    --output-dir benchmarks/results/ring/extreme_100k

echo -e "\nAll benchmarks completed!"
echo "Results saved in: benchmarks/results/ring/"

# Generate final report
echo -e "\nGenerating final report..."
python -c "
import json
import glob
from datetime import datetime

# Find all result files
json_files = glob.glob('benchmarks/results/ring/**/*.json', recursive=True)
csv_files = glob.glob('benchmarks/results/ring/**/*.csv', recursive=True)
png_files = glob.glob('benchmarks/results/ring/**/*.png', recursive=True)
md_files = glob.glob('benchmarks/results/ring/**/*.md', recursive=True)

print('\\nBenchmark Results Summary')
print('========================')
print(f'Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'\\nFiles created:')
print(f'  - JSON results: {len(json_files)}')
print(f'  - CSV results: {len(csv_files)}')
print(f'  - Visualizations: {len(png_files)}')
print(f'  - Reports: {len(md_files)}')
print(f'\\nTotal files: {len(json_files) + len(csv_files) + len(png_files) + len(md_files)}')
"