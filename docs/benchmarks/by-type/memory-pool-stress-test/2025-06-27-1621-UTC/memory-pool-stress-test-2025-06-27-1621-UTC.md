# Memory Pool Stress Test Benchmark Results

**Date**: 2025-06-27-1621-UTC  
**Git Commit**: fbade2bf  
**Hardware**: NVIDIA GeForce GTX 1080  

## Parameters

- **device**: cuda
- **iterations**: 1000
- **allocation_pattern**: attention_simulation

## Summary

- **Baseline Avg Ms**: 0.09
- **Basic Improvement Pct**: 49.55
- **Enhanced Improvement Pct**: 52.60
- **Reuse Rate**: 0.74
- **Memory Efficiency Ratio**: 142.86

## Detailed Results

| Implementation | active_buffers | avg_allocation_ms | avg_fragmentation | bucket_stats | bucket_utilization | config | fragmentation_stats | hot_cache_hits | hot_cache_size | max_allocation_ms | memory_by_pool | min_allocation_ms | numa_stats | pool_sizes | reuse_rate | total_allocated_bytes | total_buffers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | N/A | 0.09 | 0 | N/A | 0 | N/A | N/A | 0 | N/A | 63.52 | N/A | 0.00 | N/A | N/A | 0 | N/A | 0 |
| basic_pool | N/A | 0.05 | 0.00 | N/A | 2 | N/A | N/A | 0 | N/A | 0.74 | N/A | 0.02 | N/A | N/A | 0.00 | N/A | 2 |
| enhanced_pool | N/A | 0.04 | 0.01 | N/A | 3 | N/A | N/A | 5 | N/A | 0.64 | N/A | 0.01 | N/A | N/A | 0.74 | N/A | 5 |
| final_pool_stats | 7 | N/A | N/A | {'bucket_16': 2, 'bucket_14': 0, 'bucket_15': 0, 'bucket_17': 0, 'bucket_18': 0, 'bucket_23': 1, 'bucket_21': 0, 'bucket_22': 0, 'bucket_24': 2, 'bucket_25': 0, 'bucket_19': 0, 'bucket_20': 0, 'bucket_26': 1, 'bucket_13': 0, 'bucket_27': 0, 'bucket_28': 1, 'bucket_29': 0, 'bucket_30': 0} | N/A | {'fragmentation_threshold': 0.3, 'aggressive_threshold': 0.1, 'conservative_threshold': 0.5} | {'cuda': {'fragmentation_score': 0.0, 'fragments_count': 7, 'needs_defrag': False}} | N/A | 5 | N/A | {'default': 386702336, 'ring': 0, 'sparse': 0, 'distributed': 0} | N/A | {'numa_node_0': {'devices': ['cuda:0'], 'buffers': 0}, 'numa_node_1': {'devices': ['cuda:1'], 'buffers': 0}} | {'default': 7, 'ring': 0, 'sparse': 0, 'distributed': 0} | N/A | 386702336 | 7 |