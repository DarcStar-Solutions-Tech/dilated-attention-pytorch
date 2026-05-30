#!/usr/bin/env python3
"""Attention cost model: a-priori compute/memory bounds for dense vs. sparse attention.

This is the cost model behind ``docs/guides/unified-attention-architecture.md``. It
quantifies, *before* any training run, the two quantities that are mathematically
determinable a priori:

  * COMPUTE  -- the attention-matrix term scales O(n^2) dense vs O(n * W) sparse, where
               W is the (context-independent) per-query attended-key budget. The exact
               attention speedup is n / W; the whole-model speedup folds in the linear
               (QKVO + FFN) term that does NOT grow with context.
  * MEMORY   -- non-Flash scores are O(n^2) per layer (the wall Flash removes); the KV
               working set is O(n) and is sharded to O(n / p) by ring/sequence parallelism.

Quality (the third quantity) is deliberately NOT modeled here: it is not predeterminable
as a tight a-priori number (see the design doc). Treat compute/memory as bankable bounds
and gate quality empirically (loss parity + the runtime dropped-mass certificate).

Run with no arguments to reproduce the worked example in the design doc, or pass flags to
size your own config:

    python analysis/attention_cost_analysis.py
    python analysis/attention_cost_analysis.py --d-model 8192 --layers 80 --params 70e9 \
        --pattern-budget 8192 --contexts 8192,65536,524288
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
    pattern_budget: int = (
        4096  # W: attended keys per query (local + dilated + adaptive top-k)
    )
    # hardware (defaults ~ one H100, bf16 tensor cores, realistic MFU)
    gpu_peak_flops: float = 989e12
    gpu_mem_bytes: float = 80 * 1024**3
    dense_mfu: float = 0.50
    sparse_mfu: float = 0.40
    train_multiplier: float = 3.0  # fwd + 2x bwd for a full training step


# --- compute (FLOPs per forward pass) --------------------------------------
def dense_attn_flops(n: int, c: Config) -> float:
    """QK^T + A.V matmuls, all heads, all layers: 4 * n^2 * d_model * L."""
    return 4.0 * n * n * c.d_model * c.n_layers


def sparse_attn_flops(n: int, c: Config) -> float:
    """Same, but each query attends to W keys instead of n: 4 * n * W * d_model * L."""
    return 4.0 * n * c.pattern_budget * c.d_model * c.n_layers


def linear_flops(n: int, c: Config) -> float:
    """QKVO projections + FFN -- the 2N rule. Grows LINEARLY with context."""
    return 2.0 * c.params * n


def crossover_n(c: Config) -> float:
    """Context length where dense attention FLOPs == the entire linear term.

    4 n^2 d L = 2 P n  ->  n = P / (2 d L).
    Beyond this, dense attention costs more than the rest of the model combined.
    """
    return c.params / (2.0 * c.d_model * c.n_layers)


# --- memory (bytes) --------------------------------------------------------
def kv_bytes(n: int, c: Config) -> float:
    """K and V across all layers: 2 * n * d_model * bytes * L. Sparse does NOT reduce this."""
    return 2.0 * n * c.d_model * c.bytes_per_elem * c.n_layers


def dense_scores_bytes_per_layer(n: int, c: Config) -> float:
    """The n^2 score matrix per layer that Flash avoids materializing."""
    return float(n) * n * c.n_heads * c.bytes_per_elem


def weight_bytes(c: Config) -> float:
    return c.params * c.bytes_per_elem


def ring_p_to_fit(n: int, c: Config) -> int:
    """Smallest power-of-two ring/sequence-parallel degree p so KV/p + weights fit one GPU."""
    budget = c.gpu_mem_bytes - weight_bytes(c)
    if budget <= 0:
        return -1  # weights alone exceed the GPU
    kv = kv_bytes(n, c)
    p = 1
    while kv / p > budget and p < 4096:
        p *= 2
    return p if kv / p <= budget else -1


def wallclock_s(flops: float, peak: float, mfu: float) -> float:
    return flops / (peak * mfu)


# --- report ----------------------------------------------------------------
def report(c: Config, contexts: list[int]) -> None:
    print("=" * 100)
    print("Attention cost model")
    print(
        f"  model: d_model={c.d_model}  layers={c.n_layers}  heads={c.n_heads}  "
        f"params={c.params:.2g}  dtype={c.bytes_per_elem}B"
    )
    print(
        f"  sparse pattern: W={c.pattern_budget} attended keys/query (context-independent)"
    )
    print(
        f"  hardware: peak={human_flops(c.gpu_peak_flops)}/s  mem={human_bytes(c.gpu_mem_bytes)}  "
        f"MFU dense={c.dense_mfu:.0%} sparse={c.sparse_mfu:.0%}"
    )
    xover = crossover_n(c)
    print(
        f"  crossover: dense attention exceeds the whole linear term beyond "
        f"n ~= {xover:,.0f} tokens"
    )
    print(f"  weights resident: {human_bytes(weight_bytes(c))}")
    print("=" * 100)

    hdr = (
        f"{'context n':>12} | {'dense attn':>11} | {'sparse attn':>11} | {'attn x':>7} | "
        f"{'model x':>8} | {'KV (all L)':>11} | {'scores/L*':>11} | {'ring p':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in contexts:
        da, sa, lin = (
            dense_attn_flops(n, c),
            sparse_attn_flops(n, c),
            linear_flops(n, c),
        )
        attn_x = da / sa
        model_x = (lin + da) / (lin + sa)
        kv = kv_bytes(n, c)
        scores = dense_scores_bytes_per_layer(n, c)
        p = ring_p_to_fit(n, c)
        p_str = "OOM" if p < 0 else (f"{p}" if p > 1 else "1")
        print(
            f"{n:>12,} | {human_flops(da):>11} | {human_flops(sa):>11} | {attn_x:>6.0f}x | "
            f"{model_x:>7.1f}x | {human_bytes(kv):>11} | {human_bytes(scores):>11} | {p_str:>7}"
        )
    print("-" * len(hdr))
    print("  * scores/L = non-Flash score matrix per layer (the wall Flash removes).")
    print(
        "  attn x = attention-only speedup (= n/W); model x = whole-model forward speedup."
    )
    print("  ring p = min sequence-parallel degree so KV/p + weights fit one GPU.")

    # Wall-clock illustration on the largest context.
    n = max(contexts)
    da, sa = dense_attn_flops(n, c), sparse_attn_flops(n, c)
    print("=" * 100)
    print(f"Wall-clock (attention only, one forward) at n={n:,} on the configured GPU:")
    print(
        f"  dense:  {wallclock_s(da, c.gpu_peak_flops, c.dense_mfu):>8.1f} s   "
        f"(+ must shard {human_bytes(kv_bytes(n, c))} of KV first)"
    )
    print(f"  sparse: {wallclock_s(sa, c.gpu_peak_flops, c.sparse_mfu):>8.1f} s")
    print(
        f"  a full training step is ~{c.train_multiplier:g}x a forward pass "
        f"(fwd + 2x bwd), over many steps."
    )
    print("=" * 100)


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
    p.add_argument("--gpu-peak-flops", type=float, default=989e12)
    p.add_argument("--gpu-mem-gb", type=float, default=80.0)
    p.add_argument("--dense-mfu", type=float, default=0.50)
    p.add_argument("--sparse-mfu", type=float, default=0.40)
    p.add_argument(
        "--contexts",
        type=str,
        default="8192,32768,131072,1048576",
        help="comma-separated context lengths",
    )
    a = p.parse_args()
    cfg = Config(
        d_model=a.d_model,
        n_layers=a.layers,
        n_heads=a.heads,
        params=a.params,
        bytes_per_elem=a.bytes_per_elem,
        pattern_budget=a.pattern_budget,
        gpu_peak_flops=a.gpu_peak_flops,
        gpu_mem_bytes=a.gpu_mem_gb * 1024**3,
        dense_mfu=a.dense_mfu,
        sparse_mfu=a.sparse_mfu,
    )
    contexts = [int(x) for x in a.contexts.split(",")]
    return cfg, contexts


if __name__ == "__main__":
    cfg, contexts = parse_args()
    report(cfg, contexts)
