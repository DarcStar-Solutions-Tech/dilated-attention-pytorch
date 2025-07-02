#!/bin/bash
# Launch script for RingDilatedAttentionV2Collective multi-GPU benchmark

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RingDilatedAttentionV2Collective Multi-GPU Benchmark${NC}"
echo "=================================================="

# Check if benchmark script exists
BENCHMARK_SCRIPT="benchmarks/specialized/benchmark_ring_v2_collective_distributed.py"
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo -e "${RED}Error: Benchmark script not found at $BENCHMARK_SCRIPT${NC}"
    exit 1
fi

# Default parameters
BATCH_SIZE=${BATCH_SIZE:-2}
SEQ_LENGTHS=${SEQ_LENGTHS:-"16384 32768 65536"}
NUM_HEADS=${NUM_HEADS:-12}
EMBED_DIM=${EMBED_DIM:-768}
WARMUP_RUNS=${WARMUP_RUNS:-3}
BENCHMARK_RUNS=${BENCHMARK_RUNS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-"benchmark_results/ring_v2_collective"}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run benchmark
run_benchmark() {
    local num_gpus=$1
    local extra_args="${@:2}"
    
    echo -e "\n${YELLOW}Running with $num_gpus GPU(s)...${NC}"
    
    if [ $num_gpus -eq 1 ]; then
        # Single GPU - run directly
        python "$BENCHMARK_SCRIPT" \
            --batch_size $BATCH_SIZE \
            --seq_lengths $SEQ_LENGTHS \
            --num_heads $NUM_HEADS \
            --embed_dim $EMBED_DIM \
            --warmup_runs $WARMUP_RUNS \
            --benchmark_runs $BENCHMARK_RUNS \
            --output_dir "$OUTPUT_DIR" \
            $extra_args
    else
        # Multi-GPU - use torchrun
        torchrun --nproc_per_node=$num_gpus \
            "$BENCHMARK_SCRIPT" \
            --batch_size $BATCH_SIZE \
            --seq_lengths $SEQ_LENGTHS \
            --num_heads $NUM_HEADS \
            --embed_dim $EMBED_DIM \
            --warmup_runs $WARMUP_RUNS \
            --benchmark_runs $BENCHMARK_RUNS \
            --output_dir "$OUTPUT_DIR" \
            $extra_args
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Benchmark with $num_gpus GPU(s) completed successfully${NC}"
    else
        echo -e "${RED}✗ Benchmark with $num_gpus GPU(s) failed${NC}"
    fi
}

# Check available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Parse command line arguments
if [ "$1" == "--all" ]; then
    # Run all configurations
    echo -e "${GREEN}Running all GPU configurations...${NC}"
    
    # Single GPU baseline
    run_benchmark 1
    
    # Multi-GPU configurations
    for gpus in 2 4 8; do
        if [ $gpus -le $NUM_GPUS ]; then
            run_benchmark $gpus
        else
            echo -e "${YELLOW}Skipping $gpus GPU configuration (not enough GPUs)${NC}"
        fi
    done
    
elif [ "$1" == "--profile" ]; then
    # Run with profiling
    echo -e "${GREEN}Running with profiling enabled...${NC}"
    NUM_GPUS_TO_USE=${2:-1}
    run_benchmark $NUM_GPUS_TO_USE --profile
    
elif [ "$1" == "--quick" ]; then
    # Quick test with smaller sequences
    echo -e "${GREEN}Running quick test...${NC}"
    SEQ_LENGTHS="8192 16384" WARMUP_RUNS=1 BENCHMARK_RUNS=3 run_benchmark ${2:-1}
    
elif [ "$1" == "--large" ]; then
    # Test with very large sequences
    echo -e "${GREEN}Running large sequence test...${NC}"
    SEQ_LENGTHS="131072 262144" BATCH_SIZE=1 run_benchmark ${2:-4}
    
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS] [NUM_GPUS]"
    echo ""
    echo "Options:"
    echo "  --all          Run benchmarks on all GPU configurations (1, 2, 4, 8)"
    echo "  --profile      Run with profiling enabled"
    echo "  --quick        Run quick test with smaller sequences"
    echo "  --large        Run test with very large sequences"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  BATCH_SIZE     Batch size (default: 2)"
    echo "  SEQ_LENGTHS    Space-separated sequence lengths (default: '16384 32768 65536')"
    echo "  NUM_HEADS      Number of attention heads (default: 12)"
    echo "  EMBED_DIM      Embedding dimension (default: 768)"
    echo "  WARMUP_RUNS    Number of warmup runs (default: 3)"
    echo "  BENCHMARK_RUNS Number of benchmark runs (default: 10)"
    echo "  OUTPUT_DIR     Output directory (default: 'benchmark_results/ring_v2_collective')"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with 1 GPU"
    echo "  $0 4                  # Run with 4 GPUs"
    echo "  $0 --all              # Run all configurations"
    echo "  $0 --profile 2        # Run with 2 GPUs and profiling"
    echo "  $0 --quick            # Quick test"
    echo "  BATCH_SIZE=4 $0 2     # Run with batch size 4 on 2 GPUs"
    
else
    # Run with specified number of GPUs
    NUM_GPUS_TO_USE=${1:-1}
    
    if [ $NUM_GPUS_TO_USE -gt $NUM_GPUS ]; then
        echo -e "${RED}Error: Requested $NUM_GPUS_TO_USE GPUs but only $NUM_GPUS available${NC}"
        exit 1
    fi
    
    run_benchmark $NUM_GPUS_TO_USE
fi

echo -e "\n${GREEN}All benchmarks completed!${NC}"
echo "Results saved to: $OUTPUT_DIR"