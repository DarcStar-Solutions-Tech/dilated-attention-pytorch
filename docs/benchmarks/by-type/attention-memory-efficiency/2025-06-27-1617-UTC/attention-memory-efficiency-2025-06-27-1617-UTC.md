# Attention Memory Efficiency Benchmark Results

**Date**: 2025-06-27-1617-UTC  
**Git Commit**: fbade2bf  
**Hardware**: NVIDIA GeForce GTX 1080  

## Parameters

- **device**: cuda
- **test_configs**: 4
- **implementations**: ['improved', 'improved_multihead']

## Summary

- **Total Tests**: 8.00
- **Avg Memory Gb**: 0.22
- **Total Pool Buffers**: 0.00
- **Memory Pool Efficiency**: 0.00

## Detailed Results

| Implementation | active_buffers | bucket_stats | config | fragmentation_stats | hot_cache_size | improved | improved_multihead | memory_by_pool | numa_stats | pool_sizes | total_allocated_bytes | total_buffers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| implementations | N/A | N/A | N/A | N/A | N/A | [{'allocated_gb': 0.140655517578125, 'reserved_gb': 0.333984375, 'init_overhead_gb': 0.093780517578125, 'avg_forward_ms': 16.49340456351638, 'min_forward_ms': 16.06209296733141, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'medium_sequence', 'implementation': 'improved'}, {'allocated_gb': 0.28131103515625, 'reserved_gb': 0.642578125, 'init_overhead_gb': 0.18756103515625, 'avg_forward_ms': 30.74215278029442, 'min_forward_ms': 28.910615481436253, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'long_sequence', 'implementation': 'improved'}, {'allocated_gb': 0.1406402587890625, 'reserved_gb': 0.333984375, 'init_overhead_gb': 0.0937652587890625, 'avg_forward_ms': 26.550248079001904, 'min_forward_ms': 15.575956553220749, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'larger_batch', 'implementation': 'improved'}, {'allocated_gb': 0.2813720703125, 'reserved_gb': 0.642578125, 'init_overhead_gb': 0.1876220703125, 'avg_forward_ms': 38.100244756788015, 'min_forward_ms': 33.30265358090401, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'very_long_sequence', 'implementation': 'improved'}] | [{'allocated_gb': 0.15740203857421875, 'reserved_gb': 0.494140625, 'init_overhead_gb': 0.11052703857421875, 'avg_forward_ms': 158.73701684176922, 'min_forward_ms': 69.00413427501917, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'medium_sequence', 'implementation': 'improved_multihead'}, {'allocated_gb': 0.29012298583984375, 'reserved_gb': 0.923828125, 'init_overhead_gb': 0.19637298583984375, 'avg_forward_ms': 356.77130902186036, 'min_forward_ms': 121.79895956069231, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'long_sequence', 'implementation': 'improved_multihead'}, {'allocated_gb': 0.14945220947265625, 'reserved_gb': 0.474609375, 'init_overhead_gb': 0.10257720947265625, 'avg_forward_ms': 92.7386911585927, 'min_forward_ms': 70.54311595857143, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'larger_batch', 'implementation': 'improved_multihead'}, {'allocated_gb': 0.29018402099609375, 'reserved_gb': 0.923828125, 'init_overhead_gb': 0.19643402099609375, 'avg_forward_ms': 386.408400349319, 'min_forward_ms': 161.58580593764782, 'pool_buffers': 0, 'pool_hot_cache': 0, 'bucket_count': 0, 'fragmentation_score': 0.0, 'test_name': 'very_long_sequence', 'implementation': 'improved_multihead'}] | N/A | N/A | N/A | N/A | N/A |
| pool_stats | 0 | {} | {'fragmentation_threshold': 0.3, 'aggressive_threshold': 0.1, 'conservative_threshold': 0.5} | {} | 0 | N/A | N/A | {'default': 0, 'ring': 0, 'sparse': 0, 'distributed': 0} | {'numa_node_0': {'devices': ['cuda:0'], 'buffers': 0}, 'numa_node_1': {'devices': ['cuda:1'], 'buffers': 0}} | {'default': 0, 'ring': 0, 'sparse': 0, 'distributed': 0} | 0 | 0 |