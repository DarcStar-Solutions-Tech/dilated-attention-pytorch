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
def test_matches_dense_reference(pattern, is_causal):
    torch.manual_seed(0)
    batch, seq_len, heads, head_dim, block_size = 2, 64, 2, 16, 8
    cfg = SparsePatternConfig(
        pattern_type=pattern,
        block_size=block_size,
        window_size=24,
        global_tokens=1,
        dilation_rates=[1, 2],
    )
    model = BlockSparseAttention(sparse_config=cfg)

    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    out = model(q, k, v, is_causal=is_causal)
    ref = _dense_reference(q, k, v, model, is_causal)

    assert not torch.isnan(out).any(), "output contains NaN"
    assert torch.allclose(out, ref, atol=1e-10, rtol=1e-6), (
        f"max abs diff {(out - ref).abs().max().item():.3e}"
    )


def test_compute_aliases_match_reference():
    """The deprecated _batched/_sequential aliases must both equal the dense reference.

    ``enable_batched_ops`` no longer selects a distinct code path, so pin the aliases
    directly to catch any future divergence from the grouped implementation.
    """
    torch.manual_seed(7)
    batch, seq_len, heads, head_dim, block_size = 2, 48, 2, 16, 8
    cfg = SparsePatternConfig(
        pattern_type="local_window", block_size=block_size, window_size=24
    )
    model = BlockSparseAttention(sparse_config=cfg)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    nb = seq_len // block_size
    idx = model._get_sparse_block_indices(nb, heads, q.device)

    for is_causal in (False, True):
        ref = _dense_reference(q, k, v, model, is_causal)
        for method in (
            model._compute_sparse_attention_batched,
            model._compute_sparse_attention_sequential,
            model._compute_sparse_attention_grouped,
        ):
            out = torch.zeros_like(q)
            method(q, k, v, out, idx, is_causal)
            assert torch.allclose(out, ref, atol=1e-10), (
                f"{method.__name__} causal={is_causal} diff "
                f"{(out - ref).abs().max().item():.3e}"
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


def _expected_attended_rows(model, seq_len, heads, device, is_causal):
    """Query positions the pattern selects (>=1 allowed key), computed independently."""
    bs = model.block_size
    nb = seq_len // bs
    row_idx, col_idx = model._get_sparse_block_indices(nb, heads, device)
    allowed = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        allowed[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs] = True
    if is_causal:
        allowed &= torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
    return allowed.any(
        dim=-1
    )  # [seq_len] bool: True where the query attends to >=1 key


@pytest.mark.parametrize("is_causal", [False, True])
def test_returned_weights_are_jointly_normalized(is_causal):
    """Every attended query row's joint weights must sum to ~1; unattended rows to ~0."""
    torch.manual_seed(2)
    batch, seq_len, heads, head_dim, block_size = 1, 32, 2, 8, 8
    cfg = SparsePatternConfig(
        pattern_type="local_window", block_size=block_size, window_size=24
    )
    model = BlockSparseAttention(sparse_config=cfg)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    _, weights = model(q, k, v, is_causal=is_causal, return_attention_weights=True)

    bs = model.block_size
    # Accumulate weight mass per ABSOLUTE query position, indexing every row distinctly
    # (the COO row indices give the true query position for each flattened entry).
    row_sum = torch.zeros(batch, heads, seq_len, dtype=torch.float64)
    for w, rows in zip(weights["values"], weights["row_indices"]):
        # w: [b, h, q, k]; rows: flattened query position per (q, k) entry.
        qpos = rows.view(bs, bs)[:, 0]  # query positions for this block (one per q row)
        row_sum[:, :, qpos] += w.sum(dim=-1)

    expected = _expected_attended_rows(model, seq_len, heads, q.device, is_causal)
    # Attended rows sum to 1; unattended rows must be exactly 0 (not silently skipped).
    assert torch.allclose(
        row_sum[:, :, expected], torch.ones_like(row_sum[:, :, expected]), atol=1e-9
    ), (
        f"attended rows not normalized: range [{row_sum[:, :, expected].min()}, {row_sum[:, :, expected].max()}]"
    )
    if (~expected).any():
        assert torch.allclose(
            row_sum[:, :, ~expected],
            torch.zeros_like(row_sum[:, :, ~expected]),
            atol=1e-12,
        )


def test_returned_weights_coo_reconstruction_matches_dense():
    """Reconstructing a dense [seq,seq] weight matrix from (values,row,col) must match a
    dense joint-softmax reference -- guards the COO row/col orientation."""
    torch.manual_seed(5)
    batch, seq_len, heads, head_dim, block_size = 1, 32, 1, 8, 8
    cfg = SparsePatternConfig(
        pattern_type="local_window", block_size=block_size, window_size=24
    )
    model = BlockSparseAttention(sparse_config=cfg)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    _, weights = model(q, k, v, is_causal=False, return_attention_weights=True)

    dense = torch.zeros(batch, heads, seq_len, seq_len, dtype=torch.float64)
    for w, rows, cols in zip(
        weights["values"], weights["row_indices"], weights["col_indices"]
    ):
        flat = w.reshape(batch, heads, -1)  # [b, h, q*k]
        dense[:, :, rows, cols] += flat

    # Reference: dense joint softmax over the same allowed mask.
    bs = model.block_size
    nb = seq_len // bs
    ri, ci = model._get_sparse_block_indices(nb, heads, q.device)
    allowed = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for r, c in zip(ri.tolist(), ci.tolist()):
        allowed[r * bs : (r + 1) * bs, c * bs : (c + 1) * bs] = True
    qh, kh = q.transpose(1, 2), k.transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(head_dim)
    ref = torch.nan_to_num(
        torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1), nan=0.0
    )
    assert torch.allclose(dense, ref, atol=1e-10), (
        f"COO reconstruction max diff {(dense - ref).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_edge_case_patterns(monkeypatch, is_causal):
    """Diagonal-excluding and future-only block selections still match a dense reference,
    exercising the causal-skip and the 'query block with no attended key blocks' paths."""
    torch.manual_seed(9)
    batch, seq_len, heads, head_dim, block_size = 1, 32, 1, 8, 8  # 4 blocks
    cfg = SparsePatternConfig(pattern_type="local_window", block_size=block_size)
    model = BlockSparseAttention(sparse_config=cfg)

    # block 1 -> key block 0 only (excludes its own diagonal block);
    # block 2 -> key block 3 only (future-only: skipped under causal -> zero output).
    rows = torch.tensor([0, 1, 2, 3])
    cols = torch.tensor([0, 0, 3, 3])
    monkeypatch.setattr(
        model, "_get_sparse_block_indices", lambda *a, **kw: (rows, cols)
    )

    q = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    k = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)
    v = torch.randn(batch, seq_len, heads, head_dim, dtype=torch.float64)

    out = model(q, k, v, is_causal=is_causal)
    ref = _dense_reference(q, k, v, model, is_causal)
    assert not torch.isnan(out).any()
    assert torch.allclose(out, ref, atol=1e-10), (
        f"edge-case diff {(out - ref).abs().max().item():.3e}"
    )


def test_merge_robust_to_fully_masked_partial():
    """_merge_block_attention must not propagate NaN from a fully-masked (-inf) partial."""
    torch.manual_seed(11)
    b, h, qn, kn, d = 1, 1, 3, 4, 4
    scale = 1.0 / math.sqrt(d)

    def part(kk, vv, mask=False):
        s = torch.matmul(q, kk.transpose(-2, -1)) * scale
        if mask:
            s = s.masked_fill(torch.ones_like(s, dtype=torch.bool), float("-inf"))
        return torch.matmul(torch.softmax(s, dim=-1), vv), torch.logsumexp(s, dim=-1)

    q = torch.randn(b, h, qn, d, dtype=torch.float64)
    kf, vf = (
        torch.randn(b, h, kn, d, dtype=torch.float64),
        torch.randn(b, h, kn, d, dtype=torch.float64),
    )
    km, vm = (
        torch.randn(b, h, kn, d, dtype=torch.float64),
        torch.randn(b, h, kn, d, dtype=torch.float64),
    )
    of, lf = part(kf, vf)
    om, lm = part(km, vm, mask=True)  # fully-masked: lse=-inf, out=NaN

    # finite-then-masked and masked-then-finite must both yield the finite result, no NaN.
    for a, b2 in (((of, lf), (om, lm)), ((om, lm), (of, lf))):
        o1, l1 = _merge_block_attention(None, None, *a)
        o2, l2 = _merge_block_attention(o1, l1, *b2)
        assert not torch.isnan(o2).any(), "merge propagated NaN from masked partial"
        assert torch.allclose(o2, of, atol=1e-12), (
            "masked partial changed the finite result"
        )

    # both-masked: defined as zero output, -inf lse, no NaN.
    o, lse = _merge_block_attention(*_merge_block_attention(None, None, om, lm), om, lm)
    assert not torch.isnan(o).any()
    assert torch.allclose(o, torch.zeros_like(o))
    assert torch.isneginf(lse).all()


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
