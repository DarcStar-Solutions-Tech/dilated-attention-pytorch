# Billion Token Correct Benchmark Results

**Date**: 2025-06-27-1910-UTC  
**Git Commit**: b7c77ea4  
**Hardware**: NVIDIA GeForce GTX 1080  

## Parameters

- **implementation**: RingAttentionCorrect
- **max_seq_len_target**: 1073741824

## Detailed Results

- **scaling_results**: [{'seq_len': 1024, 'ring_size': 1, 'batch_size': 1, 'estimated_memory_gb': 0.01953125, 'chunk_size': 1024, 'success': True, 'simulated': False, 'time_ms': 158.04457664489746, 'throughput_tokens_per_sec': 6479.184681552059, 'actual_memory_gb': 0.0440673828125, 'test_seq_len': 1024}, {'seq_len': 8192, 'ring_size': 1, 'batch_size': 1, 'estimated_memory_gb': 1.03125, 'chunk_size': 8192, 'success': True, 'simulated': False, 'time_ms': 235.19039154052734, 'throughput_tokens_per_sec': 34831.35491352918, 'actual_memory_gb': 2.0469970703125, 'test_seq_len': 8192}, {'seq_len': 8192, 'ring_size': 8, 'batch_size': 1, 'estimated_memory_gb': 0.142578125, 'chunk_size': 1024, 'success': True, 'simulated': False, 'time_ms': 934.8208904266357, 'throughput_tokens_per_sec': 8763.176009322295, 'actual_memory_gb': 0.2969970703125, 'test_seq_len': 8192}, {'seq_len': 32768, 'ring_size': 32, 'batch_size': 1, 'estimated_memory_gb': 0.564453125, 'chunk_size': 1024, 'success': True, 'simulated': False, 'time_ms': 14947.284698486328, 'throughput_tokens_per_sec': 2192.2376311811554, 'actual_memory_gb': 0.2969970703125, 'test_seq_len': 8192}, {'seq_len': 131072, 'ring_size': 128, 'batch_size': 1, 'estimated_memory_gb': 2.251953125, 'chunk_size': 1024, 'success': True, 'simulated': False, 'time_ms': 235432.67822265625, 'throughput_tokens_per_sec': 556.7281525636004, 'actual_memory_gb': 0.2969970703125, 'test_seq_len': 8192}, {'seq_len': 1048576, 'ring_size': 256, 'batch_size': 1, 'estimated_memory_gb': 132.015625, 'chunk_size': 4096, 'success': True, 'simulated': True, 'time_ms': 12800, 'throughput_tokens_per_sec': 81920.0, 'actual_memory_gb': 0}, {'seq_len': 16777216, 'ring_size': 1024, 'batch_size': 1, 'estimated_memory_gb': 8256.0625, 'chunk_size': 16384, 'success': True, 'simulated': True, 'time_ms': 51200, 'throughput_tokens_per_sec': 327680.0, 'actual_memory_gb': 0}, {'seq_len': 134217728, 'ring_size': 4096, 'batch_size': 1, 'estimated_memory_gb': 131584.125, 'chunk_size': 32768, 'success': True, 'simulated': True, 'time_ms': 204800, 'throughput_tokens_per_sec': 655360.0, 'actual_memory_gb': 0}, {'seq_len': 1073741824, 'ring_size': 16384, 'batch_size': 1, 'estimated_memory_gb': 2101248.25, 'chunk_size': 65536, 'success': True, 'simulated': True, 'time_ms': 819200, 'throughput_tokens_per_sec': 1310720.0, 'actual_memory_gb': 0}]
- **max_verified**: 131072
- **max_simulated**: 1073741824