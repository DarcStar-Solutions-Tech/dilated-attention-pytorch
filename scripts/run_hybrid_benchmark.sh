#!/bin/bash
# Run hybrid vs V3 vs V2 benchmark on multiple GPUs

echo "Running Ring Attention Hybrid Benchmark"
echo "======================================"
echo ""

# Test configurations
GPU_COUNTS=(2 4)

for gpus in "${GPU_COUNTS[@]}"; do
    if [ "$gpus" -le $(nvidia-smi -L | wc -l) ]; then
        echo "Testing with $gpus GPUs..."
        echo "------------------------"
        
        # Set environment for better performance
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((gpus-1)))
        export NCCL_DEBUG=WARN
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
        
        # Run the benchmark
        torchrun --nproc_per_node=$gpus benchmarks/compare_hybrid_v3_v2_multi_gpu.py
        
        echo ""
        echo "Completed $gpus GPU test"
        echo ""
        sleep 2
    else
        echo "Skipping $gpus GPU test (not enough GPUs available)"
    fi
done

echo "All benchmarks completed!"