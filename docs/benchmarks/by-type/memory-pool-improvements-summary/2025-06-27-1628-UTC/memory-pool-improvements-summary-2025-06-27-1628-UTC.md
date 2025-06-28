# Memory Pool Improvements Summary Benchmark Results

**Date**: 2025-06-27-1628-UTC  
**Git Commit**: fbade2bf  
**Hardware**: NVIDIA GeForce GTX 1080  

## Parameters

- **batch_size**: 2
- **seq_len**: 8192
- **embed_dim**: 768
- **num_heads**: 12
- **device**: cuda

## Summary

- **Implementations Tested**: 5.00
- **Memory Efficiency**: 0.04
- **Avg Allocation Time Ms**: 0.73

## Detailed Results

| Implementation | ImprovedDilatedAttention | ImprovedMultiheadDilatedAttention | MultiheadDilatedAttention | RingDilatedAttention | RingMultiheadDilatedAttention | active_buffers | avg_allocation_ms | bucket_stats | config | fragmentation_stats | hot_cache_size | memory_by_pool | numa_stats | pool_sizes | total_allocated_bytes | total_buffers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stress_test | N/A | N/A | N/A | N/A | N/A | N/A | 0.73 | N/A | N/A | N/A | 4 | N/A | N/A | N/A | N/A | 4 |
| implementation_results | {'avg_time_ms': 37.057368736714125, 'min_time_ms': 20.111924037337303, 'memory_gb': 0.113677978515625} | {'avg_time_ms': 192.329047806561, 'min_time_ms': 74.59526136517525, 'memory_gb': 0.12248992919921875} | {'avg_time_ms': 60.85331812500954, 'min_time_ms': 34.925004467368126, 'memory_gb': 0.12245941162109375} | {'avg_time_ms': 19.57311201840639, 'min_time_ms': 17.518398351967335, 'memory_gb': 0.113677978515625} | {'avg_time_ms': 288.34546115249395, 'min_time_ms': 98.67981821298599, 'memory_gb': 0.26311492919921875} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| final_pool_stats | N/A | N/A | N/A | N/A | N/A | 4 | N/A | {'bucket_18': 1, 'bucket_16': 0, 'bucket_17': 0, 'bucket_19': 0, 'bucket_20': 1, 'bucket_21': 1, 'bucket_22': 0, 'bucket_23': 1, 'bucket_24': 0, 'bucket_25': 0} | {'fragmentation_threshold': 0.3, 'aggressive_threshold': 0.1, 'conservative_threshold': 0.5} | {'cuda': {'fragmentation_score': 0.0, 'fragments_count': 4, 'needs_defrag': False}} | 4 | {'default': 12845056, 'ring': 0, 'sparse': 0, 'distributed': 0} | {'numa_node_0': {'devices': ['cuda:0'], 'buffers': 0}, 'numa_node_1': {'devices': ['cuda:1'], 'buffers': 0}} | {'default': 4, 'ring': 0, 'sparse': 0, 'distributed': 0} | 12845056 | 4 |