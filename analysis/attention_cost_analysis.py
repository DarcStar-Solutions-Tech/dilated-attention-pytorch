#!/usr/bin/env python3
"""Attention cost model: a-priori compute / memory / communication for dense vs. sparse attention.

The cost model behind ``docs/guides/unified-attention-architecture.md``. It quantifies, before any
training run, what is determinable a priori, and now reflects BOTH hierarchical extensions in the
architecture (§5.1 hierarchical ring, §5.2 hierarchical routing):

  * COMPUTE  -- core attention is O(n^2) dense vs O(n*W) sparse. Content-adaptive SELECTION is not
               free; its cost depends on routing granularity:
                 token       -> O(n^2)        (DeepSeek DSA lightning indexer)
                 block        -> O(n^2 / b)    (MoBA / FlashMoBA, flat block-centroid top-k)
                 hierarchical -> O(n^2/(b*S)) coarse + O(n) fine  (§5.2 multi-level routing)
               Hierarchical routing keeps SELECTION sub-quadratic at extreme context.
  * MEMORY   -- non-Flash scores O(n^2)/layer (Flash removes); KV O(n) -> O(n/p) via ring ->
               O(n/(p*r)) with MLA-style latent compression ratio r.
  * COMM     -- ring attention rotates ~the whole KV past each device per forward. A FLAT ring pushes
               that over the slow inter-node link; a HIERARCHICAL (topology-aware) ring keeps the
               dense rotation on fast intra-node links and sends only the sparse inter-node fraction
               over slow links (§5.1). At long context COMM, not compute, is the bottleneck.

Quality is NOT modeled (not a tight a-priori number; empirically frontier-validated, gated by loss
parity + a runtime dropped-mass certificate -- see the design doc).

Examples:
    python analysis/attention_cost_analysis.py                       # block routing, flat-vs-hier comm
    python analysis/attention_cost_analysis.py --selection token     # DSA-style O(n^2) indexer
    python analysis/attention_cost_analysis.py --selection hierarchical --contexts 1073741824
    python analysis/attention_cost_analysis.py --kv-compression 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def human_flops(x: float) -> str:
    for unit, scale in (
        ("ZFLOP", 1e21),
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
        ("PB", 1024**5),
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
    ):
        if x >= scale:
            return f"{x / scale:.1f} {unit}"
    return f"{x:.0f} B"


@dataclass
class Config:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    params: float = 7e9
    bytes_per_elem: int = 2
    pattern_budget: int = 4096  # W: attended keys/query in the CORE attention
    block_size: int = 64  # b: block granularity for selection
    selection_mode: str = "block"  # "block" | "token" | "hierarchical" | "none"
    index_dim: int = 128  # dim of centroid / indexer scoring vectors
    # hierarchical-routing (§5.2) knobs
    super_block_blocks: int = 16  # S: blocks per super-block
    num_super_selected: int = 8  # top super-blocks kept per query in the coarse pass
    kv_compression_ratio: float = 1.0  # MLA-style latent KV compression (1.0 = none)
    # hardware
    gpu_peak_flops: float = 989e12
    gpu_mem_bytes: float = 80 * 1024**3
    mfu_dense: float = 0.50
    mfu_sparse_fwd: float = 0.45
    mfu_sparse_bwd: float = 0.38
    # communication (§5.1) — per-GPU effective bandwidths and topology
    node_size: int = 8  # GPUs per node
    bw_intra_gbps: float = 900.0  # NVLink-class intra-node (GB/s)
    bw_inter_gbps: float = 100.0  # InfiniBand-class inter-node (GB/s)
    inter_node_attn_frac: float = 0.05  # sparse cross-node attention fraction (HierarchicalSparsePatternGenerator default)


# --- compute (FORWARD FLOPs; backward ~= 2x forward) -----------------------
def dense_attn_flops(n: int, c: Config) -> float:
    return 4.0 * n * n * c.d_model * c.n_layers


def sparse_core_flops(n: int, c: Config) -> float:
    return 4.0 * n * c.pattern_budget * c.d_model * c.n_layers


def selection_flops(n: int, c: Config) -> float:
    """Content-adaptive SELECTION (routing) cost — depends on granularity (§5.2).

    token        : per-query x per-token indexer            -> O(n^2)
    block        : per-query x per-key-block centroid (flat) -> O(n^2 / b)
    hierarchical : coarse per-query x per-super-block + fine within selected -> O(n^2/(b*S)) + O(n)
    none         : static pattern -> 0
    """
    L, d = c.n_layers, c.index_dim
    if c.selection_mode == "none":
        return 0.0
    if c.selection_mode == "token":
        return 2.0 * n * n * d * L
    if c.selection_mode == "block":
        return 2.0 * n * (n / c.block_size) * d * L
    if c.selection_mode == "hierarchical":
        n_super = n / (c.block_size * c.super_block_blocks)  # super-blocks
        coarse = 2.0 * n * n_super * d * L  # score super-block centroids: O(n^2/(b*S))
        fine = (
            2.0 * n * (c.num_super_selected * c.super_block_blocks) * d * L
        )  # blocks in kept super-blocks: O(n)
        return coarse + fine
    raise ValueError(f"unknown selection_mode: {c.selection_mode}")


def sparse_attn_flops(n: int, c: Config) -> float:
    return sparse_core_flops(n, c) + selection_flops(n, c)


def linear_flops(n: int, c: Config) -> float:
    return 2.0 * c.params * n


def crossover_n(c: Config) -> float:
    return c.params / (2.0 * c.d_model * c.n_layers)


# --- memory (bytes) --------------------------------------------------------
def kv_bytes(n: int, c: Config) -> float:
    return 2.0 * n * c.d_model * c.bytes_per_elem * c.n_layers / c.kv_compression_ratio


def dense_scores_bytes_per_layer(n: int, c: Config) -> float:
    return float(n) * n * c.n_heads * c.bytes_per_elem


def weight_bytes(c: Config) -> float:
    return c.params * c.bytes_per_elem


def ring_p_to_fit(n: int, c: Config) -> int:
    budget = c.gpu_mem_bytes - weight_bytes(c)
    if budget <= 0:
        return -1
    kv = kv_bytes(n, c)
    p = 1
    while kv / p > budget and p < 1_000_000:
        p *= 2
    return p if kv / p <= budget else -1


# --- communication (ring KV rotation per forward, §5.1) --------------------
def ring_recv_bytes(n: int, c: Config, p: int) -> float:
    """Bytes each device receives over a full ring forward ~= (p-1)/p of the whole KV (all layers)."""
    if p <= 1:
        return 0.0
    return kv_bytes(n, c) * (p - 1) / p


def comm_time_flat(n: int, c: Config, p: int) -> float:
    """Flat ring: the whole rotation crosses the slow inter-node link (worst case when ring spans nodes)."""
    return ring_recv_bytes(n, c, p) / (c.bw_inter_gbps * 1e9)


def comm_time_hier(n: int, c: Config, p: int) -> float:
    """Hierarchical ring: dense rotation on fast intra-node links; only the sparse inter-node
    fraction crosses slow links. If the ring fits in one node (p <= node_size) it is all intra-node."""
    b = ring_recv_bytes(n, c, p)
    if p <= c.node_size:
        return b / (c.bw_intra_gbps * 1e9)
    frac = c.inter_node_attn_frac
    return b * (1 - frac) / (c.bw_intra_gbps * 1e9) + b * frac / (c.bw_inter_gbps * 1e9)


def fwd_attn_seconds(n: int, c: Config) -> float:
    """Forward attention compute time (sparse), for comm-vs-compute comparison."""
    return sparse_attn_flops(n, c) / (c.gpu_peak_flops * c.mfu_sparse_fwd)


def train_step_seconds(n: int, c: Config, sparse: bool) -> float:
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
    print("=" * 110)
    print(
        "Attention cost model (v3 — selection granularity + fwd/bwd MFU + MLA + ring communication)"
    )
    print(
        f"  model: d_model={c.d_model} layers={c.n_layers} params={c.params:.2g} dtype={c.bytes_per_elem}B"
    )
    print(
        f"  sparse: W={c.pattern_budget}, selection='{c.selection_mode}' (b={c.block_size}, "
        f"S={c.super_block_blocks}, index_dim={c.index_dim}), kv_compression={c.kv_compression_ratio:g}x"
    )
    print(
        f"  hw: peak={human_flops(c.gpu_peak_flops)}/s mem={human_bytes(c.gpu_mem_bytes)} "
        f"MFU d={c.mfu_dense:.0%}/s_fwd={c.mfu_sparse_fwd:.0%}/s_bwd={c.mfu_sparse_bwd:.0%} | "
        f"BW intra={c.bw_intra_gbps:g}/inter={c.bw_inter_gbps:g} GB/s, node={c.node_size}, inter_frac={c.inter_node_attn_frac:g}"
    )
    print(
        f"  crossover n ~= {crossover_n(c):,.0f} tokens; weights = {human_bytes(weight_bytes(c))}"
    )
    print("=" * 110)
    hdr = (
        f"{'context n':>13} | {'dense attn':>11} | {'sparse core':>11} | {'selection':>11} | "
        f"{'eff attn x':>10} | {'KV(all L)':>10} | {'ring p':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in contexts:
        da = dense_attn_flops(n, c)
        core = sparse_core_flops(n, c)
        sel = selection_flops(n, c)
        p = ring_p_to_fit(n, c)
        print(
            f"{n:>13,} | {human_flops(da):>11} | {human_flops(core):>11} | {human_flops(sel):>11} | "
            f"{da / (core + sel):>9.0f}x | {human_bytes(kv_bytes(n, c)):>10} | {('OOM' if p < 0 else str(p)):>7}"
        )
    print("-" * len(hdr))
    print(
        "  eff attn x = dense / (sparse core + selection); selection cost INCLUDED (§5.2)."
    )

    n = max(contexts)
    p = ring_p_to_fit(n, c)
    print("=" * 110)
    print(f"At n={n:,}:")
    ds, ss = train_step_seconds(n, c, False), train_step_seconds(n, c, True)
    print(
        f"  compute: training step (fwd+2bwd) dense {ds:,.0f}s vs sparse {ss:,.0f}s  -> {ds / ss:.0f}x/step"
    )
    if p > 1:
        recv = ring_recv_bytes(n, c, p)
        tf, th = comm_time_flat(n, c, p), comm_time_hier(n, c, p)
        spans = "spans nodes" if p > c.node_size else "fits 1 node"
        print(
            f"  ring comm (p={p}, {spans}): {human_bytes(recv)}/GPU/fwd  ->  "
            f"flat {tf:,.1f}s vs hierarchical {th:,.1f}s  ({tf / th:.0f}x less)"
        )
        print(
            f"  comm vs compute: fwd-attn compute ~{fwd_attn_seconds(n, c):,.1f}s; "
            f"{'COMM-BOUND' if th > fwd_attn_seconds(n, c) else 'compute-bound'} even with the hierarchical ring"
            if p > c.node_size
            else "  (single-node ring: all intra-node bandwidth)"
        )
    else:
        print("  ring comm: p=1 (fits one GPU), no ring communication.")
    print("=" * 110)


def parse_args() -> tuple[Config, list[int]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--params", type=float, default=7e9)
    p.add_argument("--bytes", type=int, default=2, dest="bytes_per_elem")
    p.add_argument("--pattern-budget", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument(
        "--selection",
        choices=["block", "token", "hierarchical", "none"],
        default="block",
    )
    p.add_argument("--index-dim", type=int, default=128)
    p.add_argument("--super-block-blocks", type=int, default=16)
    p.add_argument("--num-super-selected", type=int, default=8)
    p.add_argument("--kv-compression", type=float, default=1.0)
    p.add_argument("--gpu-peak-flops", type=float, default=989e12)
    p.add_argument("--gpu-mem-gb", type=float, default=80.0)
    p.add_argument("--mfu-dense", type=float, default=0.50)
    p.add_argument("--mfu-sparse-fwd", type=float, default=0.45)
    p.add_argument("--mfu-sparse-bwd", type=float, default=0.38)
    p.add_argument("--node-size", type=int, default=8)
    p.add_argument("--bw-intra", type=float, default=900.0)
    p.add_argument("--bw-inter", type=float, default=100.0)
    p.add_argument("--inter-node-frac", type=float, default=0.05)
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
        super_block_blocks=a.super_block_blocks,
        num_super_selected=a.num_super_selected,
        kv_compression_ratio=a.kv_compression,
        gpu_peak_flops=a.gpu_peak_flops,
        gpu_mem_bytes=a.gpu_mem_gb * 1024**3,
        mfu_dense=a.mfu_dense,
        mfu_sparse_fwd=a.mfu_sparse_fwd,
        mfu_sparse_bwd=a.mfu_sparse_bwd,
        node_size=a.node_size,
        bw_intra_gbps=a.bw_intra,
        bw_inter_gbps=a.bw_inter,
        inter_node_attn_frac=a.inter_node_frac,
    )
    return cfg, [int(x) for x in a.contexts.split(",")]


if __name__ == "__main__":
    cfg, contexts = parse_args()
    report(cfg, contexts)
