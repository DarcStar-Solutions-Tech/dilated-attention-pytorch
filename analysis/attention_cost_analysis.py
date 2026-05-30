#!/usr/bin/env python3
"""Attention cost model: a-priori compute/memory bounds for dense vs. sparse attention.

The cost model behind ``docs/guides/unified-attention-architecture.md``. It quantifies, before
any training run, the two quantities that are mathematically determinable a priori, refined by
the 2026-05-30 prior-art research (NSA / DSA / MoBA-FlashMoBA / FlexAttention / MLA):

  * COMPUTE  -- core attention is O(n^2) dense vs O(n*W) sparse (W = per-query attended-key
               budget). Content-adaptive selection is NOT free: scoring which blocks/tokens to
               keep adds a term that is O(n^2 / b) for BLOCK-level routing (MoBA / our lane) but
               O(n^2) for TOKEN-level routing (DeepSeek DSA lightning indexer). That selection
               term caps the achievable speedup, and is ~b x cheaper block-level than token-level.
  * MEMORY   -- non-Flash scores are O(n^2) per layer (Flash removes); the KV working set is O(n),
               sharded to O(n/p) by ring/sequence parallelism, and can be shrunk further by an
               MLA-style latent KV compression ratio (a SECOND, multiplicative lever on the term
               ring shards -- it lowers the ring degree p needed to fit a GPU).

Realized (not theoretical) speed also depends on MFU, which differs forward vs backward: block-
dense sparse kernels (FlashMoBA / FlexAttention) keep high forward MFU (~90% of dense) but weaker
backward MFU (~85%), so a training step (fwd + 2x bwd) is modeled with split MFU.

Quality (the third quantity) is NOT modeled here: it is not predeterminable as a tight a-priori
number (see the design doc) -- it is empirically frontier-validated (NSA/DSA match dense) and
gated by loss parity + a runtime dropped-mass certificate, not bounded in advance.

Examples:
    python analysis/attention_cost_analysis.py                      # block selection, no KV compression
    python analysis/attention_cost_analysis.py --selection token    # DeepSeek DSA-style (O(n^2) indexer)
    python analysis/attention_cost_analysis.py --kv-compression 8    # MLA-style: drops the ring degree p
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# --- units -----------------------------------------------------------------
def human_flops(x: float) -> str:
    for unit, scale in (
        ("EFLOP", 1e18),
        ("PFLOP", 1e15),
        ("TFLOP", 1e12),
        ("GFLOP", 1e9),
    ):
        if x >= scale:
            return f"{x / scale:.2f} {unit}"
    return f"{x:.0f} FLOP"


def human_bytes(x: float) -> str:
    for unit, scale in (
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
    ):
        if x >= scale:
            return f"{x / scale:.1f} {unit}"
    return f"{x:.0f} B"


# --- configuration ---------------------------------------------------------
@dataclass
class Config:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # informational; d_model already captures the head FLOPs
    params: float = 7e9
    bytes_per_elem: int = 2  # bf16/fp16
    pattern_budget: int = 4096  # W: attended keys/query in the CORE attention
    block_size: int = 64  # b: block granularity for content-adaptive selection
    selection_mode: str = (
        "block"  # "block" (MoBA / our lane) | "token" (DSA) | "none" (static)
    )
    index_dim: int = 128  # dim of centroid / lightning-indexer scoring vectors
    kv_compression_ratio: float = 1.0  # MLA-style latent KV compression (1.0 = none)
    # hardware (defaults ~ one H100, bf16 tensor cores)
    gpu_peak_flops: float = 989e12
    gpu_mem_bytes: float = 80 * 1024**3
    mfu_dense: float = 0.50  # dense fwd & bwd
    mfu_sparse_fwd: float = 0.45  # block-dense kernels ~90% of dense fwd
    mfu_sparse_bwd: float = (
        0.38  # backward weaker (FlexAttention ~85%; bwd kernel-limited)
    )


# --- compute (FORWARD FLOPs; backward ~= 2x forward) -----------------------
def dense_attn_flops(n: int, c: Config) -> float:
    """QK^T + A.V matmuls, all heads, all layers: 4 * n^2 * d_model * L (forward)."""
    return 4.0 * n * n * c.d_model * c.n_layers


def sparse_core_flops(n: int, c: Config) -> float:
    """Core attention over W selected keys/query: 4 * n * W * d_model * L (forward)."""
    return 4.0 * n * c.pattern_budget * c.d_model * c.n_layers


def selection_flops(n: int, c: Config) -> float:
    """Content-adaptive SELECTION (routing/indexer) cost -- not free.

    block: per-query x per-key-block centroid scoring (MoBA / FlashMoBA): O(n^2/b).
    token: per-query x per-token lightning indexer (DeepSeek DSA): O(n^2), b x more.
    none:  static precomputed pattern -> 0.
    """
    if c.selection_mode == "none":
        return 0.0
    if c.selection_mode == "block":
        n_key_units = max(1, n // c.block_size)
        return 2.0 * n * n_key_units * c.index_dim * c.n_layers
    if c.selection_mode == "token":
        return 2.0 * n * n * c.index_dim * c.n_layers
    raise ValueError(f"unknown selection_mode: {c.selection_mode}")


def sparse_attn_flops(n: int, c: Config) -> float:
    """Total sparse attention forward = core attention + selection overhead."""
    return sparse_core_flops(n, c) + selection_flops(n, c)


def linear_flops(n: int, c: Config) -> float:
    """QKVO projections + FFN -- the 2N rule. Grows LINEARLY with context."""
    return 2.0 * c.params * n


def crossover_n(c: Config) -> float:
    """Context where dense attention FLOPs == the entire linear term: n = P / (2 d L)."""
    return c.params / (2.0 * c.d_model * c.n_layers)


# --- memory (bytes) --------------------------------------------------------
def kv_bytes(n: int, c: Config) -> float:
    """K and V across all layers, after optional MLA-style latent compression.

    2 * n * d_model * bytes * L / compression_ratio. (Block/token SELECTION does NOT shrink this;
    only representation compression does.)
    """
    return 2.0 * n * c.d_model * c.bytes_per_elem * c.n_layers / c.kv_compression_ratio


def dense_scores_bytes_per_layer(n: int, c: Config) -> float:
    """The n^2 score matrix per layer that Flash avoids materializing."""
    return float(n) * n * c.n_heads * c.bytes_per_elem


def weight_bytes(c: Config) -> float:
    return c.params * c.bytes_per_elem


def ring_p_to_fit(n: int, c: Config) -> int:
    """Smallest power-of-two ring/sequence-parallel degree p so KV/p + weights fit one GPU."""
    budget = c.gpu_mem_bytes - weight_bytes(c)
    if budget <= 0:
        return -1
    kv = kv_bytes(n, c)
    p = 1
    while kv / p > budget and p < 4096:
        p *= 2
    return p if kv / p <= budget else -1


# --- wall-clock (training step = fwd + 2x bwd, split MFU) ------------------
def train_step_seconds(n: int, c: Config, sparse: bool) -> float:
    """Approx wall-clock for one training step (fwd + 2x bwd) of the WHOLE model."""
    fwd = linear_flops(n, c) + (
        sparse_attn_flops(n, c) if sparse else dense_attn_flops(n, c)
    )
    bwd = 2.0 * fwd
    if sparse:
        return fwd / (c.gpu_peak_flops * c.mfu_sparse_fwd) + bwd / (
            c.gpu_peak_flops * c.mfu_sparse_bwd
        )
    return (fwd + bwd) / (c.gpu_peak_flops * c.mfu_dense)


# --- report ----------------------------------------------------------------
def report(c: Config, contexts: list[int]) -> None:
    print("=" * 108)
    print(
        "Attention cost model (v2 — selection cost + fwd/bwd MFU + MLA KV compression)"
    )
    print(
        f"  model: d_model={c.d_model} layers={c.n_layers} heads={c.n_heads} "
        f"params={c.params:.2g} dtype={c.bytes_per_elem}B"
    )
    print(
        f"  sparse: W={c.pattern_budget} keys/query, selection='{c.selection_mode}' "
        f"(block_size={c.block_size}, index_dim={c.index_dim}), kv_compression={c.kv_compression_ratio:g}x"
    )
    print(
        f"  hardware: peak={human_flops(c.gpu_peak_flops)}/s mem={human_bytes(c.gpu_mem_bytes)} "
        f"MFU dense={c.mfu_dense:.0%} sparse_fwd={c.mfu_sparse_fwd:.0%} sparse_bwd={c.mfu_sparse_bwd:.0%}"
    )
    print(
        f"  crossover (dense attn = whole linear term): n ~= {crossover_n(c):,.0f} tokens; "
        f"weights resident = {human_bytes(weight_bytes(c))}"
    )
    print("=" * 108)
    hdr = (
        f"{'context n':>11} | {'dense attn':>11} | {'sparse core':>11} | {'selection':>11} | "
        f"{'eff attn x':>10} | {'model x':>8} | {'KV(all L)':>10} | {'ring p':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in contexts:
        da = dense_attn_flops(n, c)
        core = sparse_core_flops(n, c)
        sel = selection_flops(n, c)
        lin = linear_flops(n, c)
        eff_attn_x = da / (core + sel)
        model_x = (lin + da) / (lin + core + sel)
        p = ring_p_to_fit(n, c)
        p_str = "OOM" if p < 0 else str(p)
        print(
            f"{n:>11,} | {human_flops(da):>11} | {human_flops(core):>11} | {human_flops(sel):>11} | "
            f"{eff_attn_x:>9.0f}x | {model_x:>7.1f}x | {human_bytes(kv_bytes(n, c)):>10} | {p_str:>6}"
        )
    print("-" * len(hdr))
    print(
        "  eff attn x = dense / (sparse core + selection) — selection overhead INCLUDED."
    )
    print(
        "  model x = whole-model forward speedup; ring p = min seq-parallel degree to fit one GPU."
    )

    n = max(contexts)
    print("=" * 108)
    print(
        f"Training-step wall-clock (fwd + 2x bwd, whole model) at n={n:,} on the configured GPU:"
    )
    ds, ss = train_step_seconds(n, c, False), train_step_seconds(n, c, True)
    print(
        f"  dense:  {ds:>9.1f} s   sparse: {ss:>9.1f} s   ->  {ds / ss:.0f}x faster per step  "
        f"(+ dense must shard {human_bytes(kv_bytes(n, c))} KV)"
    )
    print("=" * 108)


def parse_args() -> tuple[Config, list[int]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--params", type=float, default=7e9)
    p.add_argument("--bytes", type=int, default=2, dest="bytes_per_elem")
    p.add_argument(
        "--pattern-budget", type=int, default=4096, help="W: attended keys/query"
    )
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--selection", choices=["block", "token", "none"], default="block")
    p.add_argument("--index-dim", type=int, default=128)
    p.add_argument(
        "--kv-compression",
        type=float,
        default=1.0,
        help="MLA latent KV compression ratio",
    )
    p.add_argument("--gpu-peak-flops", type=float, default=989e12)
    p.add_argument("--gpu-mem-gb", type=float, default=80.0)
    p.add_argument("--mfu-dense", type=float, default=0.50)
    p.add_argument("--mfu-sparse-fwd", type=float, default=0.45)
    p.add_argument("--mfu-sparse-bwd", type=float, default=0.38)
    p.add_argument("--contexts", type=str, default="8192,32768,131072,1048576")
    a = p.parse_args()
    cfg = Config(
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        params=a.params,
        bytes_per_elem=a.bytes_per_elem,
        pattern_budget=a.pattern_budget,
        block_size=a.block_size,
        selection_mode=a.selection,
        index_dim=a.index_dim,
        kv_compression_ratio=a.kv_compression,
        gpu_peak_flops=a.gpu_peak_flops,
        gpu_mem_bytes=a.gpu_mem_gb * 1024**3,
        mfu_dense=a.mfu_dense,
        mfu_sparse_fwd=a.mfu_sparse_fwd,
        mfu_sparse_bwd=a.mfu_sparse_bwd,
    )
    return cfg, [int(x) for x in a.contexts.split(",")]


if __name__ == "__main__":
    cfg, contexts = parse_args()
    report(cfg, contexts)
