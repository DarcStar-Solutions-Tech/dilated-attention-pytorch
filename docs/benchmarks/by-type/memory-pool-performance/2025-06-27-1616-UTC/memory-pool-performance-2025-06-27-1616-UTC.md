# Memory Pool Performance Benchmark Results

**Date**: 2025-06-27-1616-UTC  
**Git Commit**: fbade2bf  
**Hardware**: NVIDIA GeForce GTX 1080  

## Parameters

- **device**: cuda
- **test_iterations**: 100
- **fragmentation_threshold**: 0.3

## Summary

- **Allocation Improvement Pct**: 0.59
- **Reuse Improvement Pct**: 8.84
- **Memory Improvement Pct**: 0.00
- **Avg Bucket Efficiency**: 0.38
- **Fragmentation Handled**: 1.00

## Detailed Results

- **basic_results**: [{'test_name': 'basic_test_0', 'allocation_time_ms': 1.1036759242415428, 'reuse_time_ms': 1.011882908642292, 'fragmentation_score': 0.1, 'memory_used_mb': 0.0146484375, 'cache_hits': 4, 'bucket_efficiency': 0.5}, {'test_name': 'basic_test_1', 'allocation_time_ms': 2.6593683287501335, 'reuse_time_ms': 3.2502543181180954, 'fragmentation_score': 0.1, 'memory_used_mb': 4.0146484375, 'cache_hits': 1, 'bucket_efficiency': 0.35714285714285715}, {'test_name': 'basic_test_2', 'allocation_time_ms': 2.185555174946785, 'reuse_time_ms': 2.2420231252908707, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 1, 'bucket_efficiency': 0.3333333333333333}, {'test_name': 'basic_test_3', 'allocation_time_ms': 1.9601844251155853, 'reuse_time_ms': 1.949322409927845, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 3, 'bucket_efficiency': 0.3157894736842105}]
- **enhanced_results**: [{'test_name': 'enhanced_test_0', 'allocation_time_ms': 1.0636979714035988, 'reuse_time_ms': 1.017903909087181, 'fragmentation_score': 0.1, 'memory_used_mb': 0.0146484375, 'cache_hits': 4, 'bucket_efficiency': 0.5}, {'test_name': 'enhanced_test_1', 'allocation_time_ms': 2.5384463369846344, 'reuse_time_ms': 2.5028344243764877, 'fragmentation_score': 0.1, 'memory_used_mb': 4.0146484375, 'cache_hits': 1, 'bucket_efficiency': 0.35714285714285715}, {'test_name': 'enhanced_test_2', 'allocation_time_ms': 2.1562669426202774, 'reuse_time_ms': 2.1326718851923943, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 1, 'bucket_efficiency': 0.3333333333333333}, {'test_name': 'enhanced_test_3', 'allocation_time_ms': 2.103523351252079, 'reuse_time_ms': 2.052750438451767, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 3, 'bucket_efficiency': 0.3157894736842105}]
- **fragmentation_test**: {'test_name': 'fragmented_allocation', 'allocation_time_ms': 1.362355425953865, 'reuse_time_ms': 1.31931621581316, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 0, 'bucket_efficiency': 0.3157894736842105}
- **numa_results**: [{'test_name': 'numa_node_0_device_cuda:0', 'allocation_time_ms': 1.8287207931280136, 'reuse_time_ms': 1.8206676468253136, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 0, 'bucket_efficiency': 0.3157894736842105}, {'test_name': 'numa_node_1_device_cuda:1', 'allocation_time_ms': 1.7689960077404976, 'reuse_time_ms': 1.8444424495100975, 'fragmentation_score': 0.0, 'memory_used_mb': 68.0146484375, 'cache_hits': 0, 'bucket_efficiency': 0.3157894736842105}]