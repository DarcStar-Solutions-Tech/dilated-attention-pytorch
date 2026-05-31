"""Regression tests for `HierarchicalSparsePatternGenerator` load balancing.

Guards the bug where `_apply_load_balancing` compared `avg_time` to a multiple of
*itself* (`avg_time > (1 + thr) * avg_time`), so the over/under-load branches were
dead code and the balancer was a silent no-op. The fix compares the recent load
window against the rank's longer-term baseline; these tests assert each branch fires
and produces the right direction of sparsity adjustment.
"""

import torch

from dilated_attention_pytorch.sparse.distributed_sparse_config import (
    DistributedSparseConfig,
)
from dilated_attention_pytorch.sparse.sparse_pattern_generator import (
    HierarchicalSparsePatternGenerator,
)


def _generator() -> HierarchicalSparsePatternGenerator:
    # All-default config; node-size detection falls back to min(8, world_size)
    # with no distributed init required.
    return HierarchicalSparsePatternGenerator(
        DistributedSparseConfig(), world_size=8, rank=0
    )


def _half_dense_patterns() -> dict[str, torch.Tensor]:
    # ~50% density so there is room to both add and remove connections.
    torch.manual_seed(0)
    pat = torch.rand(2, 8, 8) < 0.5
    return {"local": pat.clone()}


def _density(patterns: dict[str, torch.Tensor]) -> float:
    return float(patterns["local"].float().mean())


def _push(gen: HierarchicalSparsePatternGenerator, times: list[float]) -> None:
    for t in times:
        gen.update_load_stats(
            computation_time=t, communication_volume=0, memory_usage=0
        )


def test_overloaded_rank_sheds_work() -> None:
    """Recent load >> baseline → overloaded → sparsity increases (density drops)."""
    gen = _generator()
    _push(gen, [1.0] * 20 + [3.0] * 10)  # recent 10 high vs baseline ~1.67
    patterns = _half_dense_patterns()
    before = _density(patterns)
    after = _density(gen._apply_load_balancing(patterns, num_blocks=8))
    assert after < before, f"overloaded rank should shed work: {after} !< {before}"


def test_underloaded_rank_takes_more_work() -> None:
    """Recent load << baseline → underloaded → sparsity decreases (density rises)."""
    gen = _generator()
    _push(gen, [3.0] * 20 + [1.0] * 10)  # recent 10 low vs baseline ~2.33
    patterns = _half_dense_patterns()
    before = _density(patterns)
    after = _density(gen._apply_load_balancing(patterns, num_blocks=8))
    assert after > before, (
        f"underloaded rank should take more work: {after} !> {before}"
    )


def test_balanced_rank_is_unchanged() -> None:
    """Recent load == baseline → neither branch fires → pattern untouched."""
    gen = _generator()
    _push(gen, [1.0] * 30)
    patterns = _half_dense_patterns()
    before = _density(patterns)
    after = _density(gen._apply_load_balancing(patterns, num_blocks=8))
    assert after == before, f"balanced rank should be unchanged: {after} != {before}"


def test_no_history_is_noop() -> None:
    """Too little history (< 2 samples) → no adjustment, no crash."""
    gen = _generator()
    patterns = _half_dense_patterns()
    before = _density(patterns)
    assert _density(gen._apply_load_balancing(patterns, num_blocks=8)) == before
    _push(gen, [1.0])  # single sample still below the 2-sample floor
    assert _density(gen._apply_load_balancing(patterns, num_blocks=8)) == before
