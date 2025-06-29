# Ring Attention Pattern Caching Integration Plan

Generated: 2025-06-29T14:30:00Z

## Background

Ring Dilated Attention modules currently maintain their own in-memory caches for
dilated indices. While this avoids recomputation, it duplicates patterns across
modules and does not take advantage of the unified `PatternCache` utilities that
already power other attention variants. A single global cache would allow pattern
reuse across all ring implementations and reduce memory usage.

## Objectives

1. **Integrate the global pattern cache** into all ring attention modules.
2. **Benchmark performance** with and without caching enabled.
3. **Compare results** against the previous local caching approach.
4. **Document usage** and publish the benchmark findings.

## Implementation Steps

1. **Audit Ring Implementations**
   - `ring_dilated_attention_v2.py`
   - `ring_dilated_attention_production.py`
   - `ring_distributed_dilated_attention.py`
   - `true_ring_dilated_attention.py`
   - Block‑sparse ring variants

2. **Add a `use_pattern_cache` flag** to each class constructor. When enabled,
   obtain the cache via `get_global_pattern_cache()`.

3. **Replace per-instance caches** in `_apply_dilation` with cache lookups:
   - Generate a key using sequence length, `segment_lengths`, `dilation_rates`,
     and the current offset.
   - Retrieve indices with `cache.get_dilated_indices()`.
   - If missing, compute the indices, then store them using
     `cache.put_dilated_indices()`.

4. **Provide helper functions** (`clear_pattern_cache()`) to clear the global
   cache during tests and benchmarks.

5. **Ensure thread safety** by relying on the locking already implemented in the
   `PatternCache` utilities.

6. **Update unit tests** so multiple forward passes confirm that the same
   pattern object is reused.

## Benchmarking Plan

1. **Create `benchmark_ring_pattern_cache.py`** under `benchmarks/`.
   - Measure cold and warm forward-pass times for RingDilatedAttentionV2 with and
     without caching.
   - Test sequences of 16K–64K tokens with `ring_size` ≥ 4.
   - Record peak memory usage and pattern generation time.
   - Save results to
     `benchmark_ring_pattern_cache_results-<timestamp>.json`.

2. **Run baseline benchmarks** using the existing local-cache implementation for
   comparison. Preserve these results for the report.

3. **Compare performance** by calculating speedup and memory reduction. Expect at
   least a 2–3× improvement in pattern generation time and a 10–15% reduction in
   overall forward-pass latency.

4. **Publish findings** in `docs/benchmarks/` and update `docs/benchmarks/README.md`
   with a link to the new results file.

## Documentation Updates

- Add a section to `docs/guides/RING_ATTENTION_EXPLANATION.md` describing how to
  enable the global pattern cache using `use_pattern_cache=True`.
- Summarize benchmark results in a short report placed in
  `docs/benchmarks/`. Include a comparison table of old vs. new timings.
- Mention the new flag in the project README under the Ring Attention section.

## Timeline

- **Day 1** – Implement caching hooks in all ring modules and update unit tests.
- **Day 2** – Run benchmarks on both CPU and GPU; collect JSON results.
- **Day 3** – Finalize documentation and integrate benchmark plots.

## Next Steps

- Investigate persistent on-disk caching for extremely long sequences.
- Evaluate cache sharing across distributed workers to minimize communication
  overhead when patterns are identical.

