#!/usr/bin/env python3
"""Attention cost model: a-priori compute / memory / communication for dense vs. sparse attention.

The cost model behind ``docs/guides/unified-attention-architecture.md``. It quantifies, before any
training run, what is determinable a priori, reflecting both architectural hierarchies (§5.1 ring,
§5.2 routing) and now valid into the extreme (petabyte / billion-token) regime:

  * COMPUTE  -- core attention O(n^2) dense vs O(n*W) sparse. SELECTION (routing) cost by granularity:
                 token        -> O(n^2)              (DeepSeek DSA lightning indexer)
                 block        -> O(n^2 / b)          (MoBA / FlashMoBA, flat block-centroid top-k)
                 hierarchical -> O(n^2/(b*S)) + O(n)  (§5.2 multi-level routing; sub-quadratic)
  * MEMORY   -- model state (weights+grads+optimizer+master) is SHARDED across GPUs (FSDP/ZeRO/expert),
               not replicated; KV is O(n) and sharded O(n/p) by the ring, shrinkable by MLA ratio r.
  * COMM     -- the ring rotates KV per forward. Three levers: (1) HIERARCHICAL ring keeps the dense
               rotation on fast intra-node links, sparse fraction on slow inter-node (§5.1);
               (2) SPARSE-RING PRUNING only moves the KV blocks some local query selects
               (volume ~ W/n, not the whole KV); (3) MLA compression shrinks what is moved.
               At extreme context, pruning is what flips the run from comm-bound to compute-bound.

Quality is NOT modeled (not a tight a-priori number; gated by loss parity + a dropped-mass certificate).

Examples:
    python analysis/attention_cost_analysis.py
    python analysis/attention_cost_analysis.py --selection hierarchical --contexts 1073741824
    python analysis/attention_cost_analysis.py --params 500e12 --active-params 1e12 \
        --d-model 16384 --layers 128 --pattern-budget 65536 --block-size 128 \
        --selection hierarchical --kv-compression 64 --contexts 1073741824
    python analysis/attention_cost_analysis.py --no-comm-pruning   # dense ring, for comparison
"""

from __future__ import annotations

import argparse
import math
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


def human_time(s: float) -> str:
    if s >= 86400:
        return f"{s / 86400:.1f} d"
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 60:
        return f"{s / 60:.1f} min"
    return f"{s:.2f} s"


@dataclass
class Config:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    params: float = 7e9  # TOTAL params (memory)
    active_params: float = 0.0  # ACTIVE params (compute); 0 -> dense, use params
    bytes_per_elem: int = 2
    pattern_budget: int = 4096  # W: attended keys/query in the CORE attention
    block_size: int = 64
    selection_mode: str = "block"  # block | token | hierarchical | none
    index_dim: int = 128
    super_block_blocks: int = 16  # S
    num_super_selected: int = 8
    kv_compression_ratio: float = 1.0  # MLA latent KV compression
    comm_pruning: bool = True  # sparse-ring: only move selected KV blocks
    # hardware
    gpu_peak_flops: float = 989e12
    gpu_mem_bytes: float = 80 * 1024**3
    state_bytes_per_param: float = (
        16.0  # Adam mixed (2 wt + 2 grad + 4 master + 4 m + 4 v); Muon ~12
    )
    kv_mem_frac: float = 0.5  # GPU memory fraction available to the KV shard
    static_mem_frac: float = (
        0.7  # GPU memory fraction available to the sharded model state
    )
    mfu_dense: float = 0.50
    mfu_sparse_fwd: float = 0.45
    mfu_sparse_bwd: float = 0.38
    # communication
    node_size: int = 8
    bw_intra_gbps: float = 900.0
    bw_inter_gbps: float = 100.0
    inter_node_attn_frac: float = 0.05

    def compute_params(self) -> float:
        return self.active_params if self.active_params > 0 else self.params


# --- compute (FORWARD FLOPs; backward ~= 2x forward) -----------------------
def dense_attn_flops(n: int, c: Config) -> float:
    return 4.0 * n * n * c.d_model * c.n_layers


def sparse_core_flops(n: int, c: Config) -> float:
    return 4.0 * n * c.pattern_budget * c.d_model * c.n_layers


def selection_flops(n: int, c: Config) -> float:
    L, d = c.n_layers, c.index_dim
    if c.selection_mode == "none":
        return 0.0
    if c.selection_mode == "token":
        return 2.0 * n * n * d * L
    if c.selection_mode == "block":
        return 2.0 * n * (n / c.block_size) * d * L
    if c.selection_mode == "hierarchical":
        n_super = n / (c.block_size * c.super_block_blocks)
        coarse = 2.0 * n * n_super * d * L
        fine = 2.0 * n * (c.num_super_selected * c.super_block_blocks) * d * L
        return coarse + fine
    raise ValueError(f"unknown selection_mode: {c.selection_mode}")


def sparse_attn_flops(n: int, c: Config) -> float:
    return sparse_core_flops(n, c) + selection_flops(n, c)


def linear_flops(n: int, c: Config) -> float:
    return 2.0 * c.compute_params() * n


def crossover_n(c: Config) -> float:
    return c.compute_params() / (2.0 * c.d_model * c.n_layers)


# --- memory: SHARDED model state + sequence-parallel KV --------------------
def model_state_bytes(c: Config) -> float:
    """Weights + grads + optimizer + master, for the TOTAL params (sharded across the cluster)."""
    return c.params * c.state_bytes_per_param


def gpus_for_state(c: Config) -> int:
    """Min GPUs to hold the sharded model state (FSDP/ZeRO-3/expert-parallel)."""
    return max(
        1, math.ceil(model_state_bytes(c) / (c.gpu_mem_bytes * c.static_mem_frac))
    )


def kv_bytes(n: int, c: Config) -> float:
    return 2.0 * n * c.d_model * c.bytes_per_elem * c.n_layers / c.kv_compression_ratio


def ring_degree(n: int, c: Config) -> int:
    """Sequence-parallel / ring degree p so each KV shard fits the per-GPU KV budget."""
    return max(1, math.ceil(kv_bytes(n, c) / (c.gpu_mem_bytes * c.kv_mem_frac)))


def dense_scores_bytes_per_layer(n: int, c: Config) -> float:
    return float(n) * n * c.n_heads * c.bytes_per_elem


# --- communication (ring KV rotation per forward, §5.1 + sparse pruning) ---
def comm_density(n: int, c: Config) -> float:
    """Fraction of the KV a device must RECEIVE. Dense ring = 1.0; sparse ring only moves the
    selected blocks (~W/n) — the §5.1 sparse-comm-pruning lever (optimistic; real union is larger)."""
    if not c.comm_pruning:
        return 1.0
    return min(1.0, c.pattern_budget / n)


def ring_recv_bytes(n: int, c: Config, p: int) -> float:
    if p <= 1:
        return 0.0
    return kv_bytes(n, c) * (p - 1) / p * comm_density(n, c)


def comm_time_flat(n: int, c: Config, p: int) -> float:
    return ring_recv_bytes(n, c, p) / (c.bw_inter_gbps * 1e9)


def comm_time_hier(n: int, c: Config, p: int) -> float:
    b = ring_recv_bytes(n, c, p)
    if p <= c.node_size:
        return b / (c.bw_intra_gbps * 1e9)
    f = c.inter_node_attn_frac
    return b * (1 - f) / (c.bw_intra_gbps * 1e9) + b * f / (c.bw_inter_gbps * 1e9)


def fwd_attn_seconds(n: int, c: Config) -> float:
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
    print("=" * 112)
    print(
        "Attention cost model (v4 — sharded state + sparse-ring comm pruning; extreme-scale valid)"
    )
    cp = c.compute_params()
    print(
        f"  model: d_model={c.d_model} layers={c.n_layers} params(total)={c.params:.2g} "
        f"active={cp:.2g} dtype={c.bytes_per_elem}B"
    )
    print(
        f"  sparse: W={c.pattern_budget}, selection='{c.selection_mode}' (b={c.block_size}, S={c.super_block_blocks}), "
        f"kv_compression={c.kv_compression_ratio:g}x, comm_pruning={c.comm_pruning}"
    )
    print(
        f"  hw: peak={human_flops(c.gpu_peak_flops)}/s mem={human_bytes(c.gpu_mem_bytes)} "
        f"state={c.state_bytes_per_param:g}B/param | BW intra={c.bw_intra_gbps:g}/inter={c.bw_inter_gbps:g} GB/s node={c.node_size}"
    )
    print(
        f"  model state (sharded): {human_bytes(model_state_bytes(c))} -> {gpus_for_state(c):,} GPUs to hold it; "
        f"crossover n ~= {crossover_n(c):,.0f}"
    )
    print("=" * 112)
    hdr = (
        f"{'context n':>13} | {'dense attn':>11} | {'sparse core':>11} | {'selection':>11} | "
        f"{'eff attn x':>10} | {'KV/seq':>10} | {'ring deg':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in contexts:
        da, core, sel = (
            dense_attn_flops(n, c),
            sparse_core_flops(n, c),
            selection_flops(n, c),
        )
        print(
            f"{n:>13,} | {human_flops(da):>11} | {human_flops(core):>11} | {human_flops(sel):>11} | "
            f"{da / (core + sel):>9.0f}x | {human_bytes(kv_bytes(n, c)):>10} | {ring_degree(n, c):>8,}"
        )
    print("-" * len(hdr))
    print(
        "  eff attn x = dense / (sparse core + selection); ring deg = seq-parallel degree to shard KV."
    )

    n = max(contexts)
    p = ring_degree(n, c)
    total_gpus = max(gpus_for_state(c), p)
    print("=" * 112)
    print(
        f"At n={n:,}  (ring degree p={p:,}, {'spans nodes' if p > c.node_size else 'fits 1 node'}):"
    )
    ds, ss = train_step_seconds(n, c, False), train_step_seconds(n, c, True)
    # per-GPU forward compute (full fwd FLOPs spread over the cluster) — the honest comm-bound basis
    t_comp = (
        (linear_flops(n, c) + sparse_attn_flops(n, c))
        / total_gpus
        / (c.gpu_peak_flops * c.mfu_sparse_fwd)
    )
    print(
        f"  GPUs: ~{total_gpus:,} (max of {gpus_for_state(c):,} for state, {p:,} for KV) x data-parallel"
    )
    print(
        f"  compute: training step (fwd+2bwd) dense {human_time(ds)} vs sparse {human_time(ss)}  -> {ds / ss:.0f}x"
    )
    if p > 1:
        # comm without pruning (dense ring) vs with pruning (sparse ring)
        dens = c.comm_pruning
        c.comm_pruning = False
        dense_flat, dense_hier = comm_time_flat(n, c, p), comm_time_hier(n, c, p)
        c.comm_pruning = True
        spr_flat, spr_hier = comm_time_flat(n, c, p), comm_time_hier(n, c, p)
        c.comm_pruning = dens
        print(f"  comm /forward (per-GPU fwd compute ≈ {human_time(t_comp)}):")
        print(
            f"    dense ring : flat {human_time(dense_flat)} | hierarchical {human_time(dense_hier)}"
        )
        print(
            f"    sparse ring: flat {human_time(spr_flat)} | hierarchical {human_time(spr_hier)}  "
            f"(density {min(1.0, c.pattern_budget / n):.2g})"
        )
        bound = "COMPUTE-bound ✓" if spr_hier <= t_comp else "still COMM-bound"
        print(
            f"    -> with hierarchical ring + sparse pruning + {c.kv_compression_ratio:g}x KV compression: {bound}"
        )
    else:
        print("  comm: p=1, no ring communication.")
    print("=" * 112)


def parse_args() -> tuple[Config, list[int]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--d-model", type=int, default=4096)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--params", type=float, default=7e9, help="total params (memory)")
    p.add_argument(
        "--active-params",
        type=float,
        default=0.0,
        help="active params (compute); 0=dense",
    )
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
    p.add_argument(
        "--no-comm-pruning",
        action="store_true",
        help="model a dense ring (no sparse-comm pruning)",
    )
    p.add_argument("--gpu-peak-flops", type=float, default=989e12)
    p.add_argument("--gpu-mem-gb", type=float, default=80.0)
    p.add_argument("--state-bytes-per-param", type=float, default=16.0)
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
        active_params=a.active_params,
        bytes_per_elem=a.bytes_per_elem,
        pattern_budget=a.pattern_budget,
        block_size=a.block_size,
        selection_mode=a.selection,
        index_dim=a.index_dim,
        super_block_blocks=a.super_block_blocks,
        num_super_selected=a.num_super_selected,
        kv_compression_ratio=a.kv_compression,
        comm_pruning=not a.no_comm_pruning,
        gpu_peak_flops=a.gpu_peak_flops,
        gpu_mem_bytes=a.gpu_mem_gb * 1024**3,
        state_bytes_per_param=a.state_bytes_per_param,
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
