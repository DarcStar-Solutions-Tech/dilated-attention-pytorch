# Single GPU Ring Hilbert Attention - Extreme Sequence Lengths

Generated: 2025-07-08 21:35:21 UTC

## Configuration

- GPU: NVIDIA GeForce GTX 1080 (8GB)
- Architecture: Pascal
- Data Type: torch.float32
- Ring Size: 1 (single GPU)

## Performance Results

| Seq Length | Hilbert | Forward (ms) | Backward (ms) | Total (ms) | Memory (MB) | Peak (MB) | Throughput (tok/s) |
|------------|---------|--------------|---------------|------------|-------------|-----------|-------------------|
| 8,192 | Yes | 71.95±14.25 | 171.05±28.92 | 243.00 | 0.0 | 106.9 | 113,856 |
| 8,192 | No | 82.74±26.03 | 243.02±102.10 | 325.77 | 0.0 | 106.9 | 99,006 |

## Maximum Sequence Length Achieved

- **With Hilbert**: 8,192 tokens
- **Without Hilbert**: 8,192 tokens
- **Overall Maximum**: 8,192 tokens

## Failed Attempts

- 16,384 tokens (Hilbert): OOM - CUDA out of memory
- 16,384 tokens (No Hilbert): skipped - OOM with Hilbert
- 32,768 tokens (No Hilbert): skipped - Would exceed GPU memory limit: 221.1% > 85.0%
- 65,536 tokens (No Hilbert): skipped - Would exceed GPU memory limit: 788.1% > 85.0%
- 131,072 tokens (No Hilbert): skipped - Would exceed GPU memory limit: 1544.1% > 85.0%
- 262,144 tokens (No Hilbert): skipped - Would exceed GPU memory limit: 3056.2% > 85.0%
- 524,288 tokens (No Hilbert): skipped - Would exceed GPU memory limit: 6080.2% > 85.0%

## Hilbert Impact Analysis

