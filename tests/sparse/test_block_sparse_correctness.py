"""Correctness tests for BlockSparseAttention against a dense masked-softmax reference.

These guard the audit findings in ``block_sparse_attention.py``:
- the per-block-softmax outputs must be combined under ONE joint softmax denominator
  (not summed), and
- causal masking must neither leak future tokens (sequential path) nor produce NaN
  from fully-masked rows (batched path).

The reference builds the exact position-level mask implied by the model's selected block
pattern (intersected with the causal triangle when requested) and runs a single dense
softmax over it; the block-sparse output must match.
"""

import math

import pytest
import torch

from dilated_attention_pytorch.sparse.block_sparse_attention import (
    BlockSparseAttention,
    SparsePatternConfig,
    _merge_block_attention,
)


def _dense_reference(q, k, v, model, is_causal):
    """Dense attention restricted to the model's selected blocks (+ causal), one softmax."""
    b, s, h, d = q.shape
    bs = model.block_size
    num_blocks = s // bs
    row_idx, col_idx = model._get_sparse_block_indices(num_blocks, h, q.device)

    allowed = torch.zeros(s, s, dtype=torch.bool, device=q.device)
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        allowed[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs] = True
    if is_causal:
        allowed &= torch.tril(torch.ones(s, s, dtype=torch.bool, device=q.device))

    qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(~allowed, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    # Rows with no allowed key -> all -inf -> NaN; such query positions get zero output,
    # matching the block-sparse path (which leaves unattended query blocks at zero).
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.matmul(attn, vh)
    return out.transpose(1, 2)


@pytest.mark.parametrize("pattern", ["local_window", "dilated_sparse", "global_local"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("enable_batched_ops", [True, False])
def test_matches_dense_reference(pattern, is_causal, enable_batched_ops):
    torch.manual_seed(0)
    batch, seq_len, heads, head_dim, block_size = 2, 64, 2, 16, 8
    cfg = SparsePatternConfig(
        pattern_type=pattern,
        block_size=block_size,
        window_size=24,
        global_tokens=1,
        dilation_rates=[1, 2],
    )
    model = BlockSparseAttention(
        sparse_config=cfg, enable_batched_ops=enable_batched_ops
    )

    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    out = model(q, k, v, is_causal=is_causal)
    ref = _dense_reference(q, k, v, model, is_causal)

    assert not torch.isnan(out).any(), "output contains NaN"
    assert torch.allclose(out, ref, atol=1e-10, rtol=1e-6), (
        f"max abs diff {(out - ref).abs().max().item():.3e}"
    )


def test_old_sum_bug_would_fail():
    """A query block attending to >1 key block must not inflate magnitude (the old += bug)."""
    torch.manual_seed(1)
    batch, seq_len, heads, head_dim, block_size = 1, 32, 1, 8, 8  # 4 blocks
    cfg = SparsePatternConfig(
        pattern_type="local_window", block_size=block_size, window_size=32
    )
    model = BlockSparseAttention(sparse_config=cfg, enable_batched_ops=False)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    out = model(q, k, v, is_causal=False)
    ref = _dense_reference(q, k, v, model, is_causal=False)
    # Joint softmax is a convex combination of v: output norm per position <= max |v| row norm.
    assert torch.allclose(out, ref, atol=1e-10)
    # Sanity: summed-softmax bug produced norms ~k x too large; assert we're in v's range.
    assert out.abs().max() <= v.abs().max() + 1e-6


def test_returned_weights_are_jointly_normalized():
    """return_attention_weights must give true joint weights (per query row sum to ~1)."""
    torch.manual_seed(2)
    batch, seq_len, heads, head_dim, block_size = 1, 32, 1, 8, 8
    cfg = SparsePatternConfig(
        pattern_type="local_window", block_size=block_size, window_size=24
    )
    model = BlockSparseAttention(sparse_config=cfg, enable_batched_ops=False)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    _, weights = model(q, k, v, is_causal=False, return_attention_weights=True)

    # Sum the per-block weights for each absolute query row across all its key blocks.
    seq = seq_len
    row_sum = torch.zeros(batch, heads, seq, dtype=torch.float64)
    for w, rows, cols in zip(
        weights["values"], weights["row_indices"], weights["col_indices"]
    ):
        # w: [b, h, bs, bs]; rows/cols index the flattened (query,key) within the block pair
        qrows = rows.view(model.block_size, model.block_size)[:, 0]  # query positions
        row_sum[:, :, qrows] += w.sum(dim=-1)

    attended = row_sum > 0
    assert torch.allclose(
        row_sum[attended], torch.ones_like(row_sum[attended]), atol=1e-9
    ), (
        f"weights not normalized: range [{row_sum[attended].min()}, {row_sum[attended].max()}]"
    )


def test_merge_primitive_matches_joint_softmax():
    """The LSE merge of two key-block partials equals one softmax over the concatenation."""
    torch.manual_seed(3)
    b, h, qn, kn, d = 2, 3, 5, 7, 4
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, qn, d, dtype=torch.float64)
    ka, va = (
        torch.randn(b, h, kn, d, dtype=torch.float64),
        torch.randn(b, h, kn, d, dtype=torch.float64),
    )
    kb, vb = (
        torch.randn(b, h, kn, d, dtype=torch.float64),
        torch.randn(b, h, kn, d, dtype=torch.float64),
    )

    def part(kk, vv):
        s = torch.matmul(q, kk.transpose(-2, -1)) * scale
        return torch.matmul(torch.softmax(s, dim=-1), vv), torch.logsumexp(s, dim=-1)

    oa, la = part(ka, va)
    ob, lb = part(kb, vb)
    merged, _ = _merge_block_attention(
        *_merge_block_attention(None, None, oa, la), ob, lb
    )

    # Reference: single softmax over concatenated keys.
    kc, vc = torch.cat([ka, kb], dim=2), torch.cat([va, vb], dim=2)
    s = torch.matmul(q, kc.transpose(-2, -1)) * scale
    joint = torch.matmul(torch.softmax(s, dim=-1), vc)
    assert torch.allclose(merged, joint, atol=1e-12)
