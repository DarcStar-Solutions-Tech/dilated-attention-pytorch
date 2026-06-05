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
               (volume ~ W/n if all local queries select the SAME blocks — see below; not the whole
               KV); (3) MLA compression shrinks what is moved.
               At extreme context, pruning is what *can* flip the run from comm-bound to
               compute-bound -- but ONLY if selections cluster. A rank receives the UNION of blocks
               all its ~n/p local queries select; under independent selection that union saturates
               toward the whole KV (1-(1-W/n)^(n/p) -> 1), i.e. a dense ring. comm_clustering knob:
               "clustered" (W/n, optimistic floor) vs "independent" (union, upper bound).

Quality is NOT modeled (not a tight a-priori number; gated by loss parity + a dropped-mass certificate).

Examples:
    python analysis/attention_cost_analysis.py
    python analysis/attention_cost_analysis.py --selection hierarchical --contexts 1073741824
    python analysis/attention_cost_analysis.py --params 500e12 --active-params 1e12 \
        --d-model 16384 --layers 128 --pattern-budget 65536 --block-size 128 \
        --selection hierarchical --kv-compression 64 --contexts 1073741824
    python analysis/attention_cost_analysis.py --no-comm-pruning   # dense ring, for comparison
    python analysis/attention_cost_analysis.py --params 500e12 --active-params 1e12 \
        --d-model 16384 --layers 128 --pattern-budget 65536 --block-size 128 \
        --selection hierarchical --kv-compression 64 --contexts 1073741824 \
        --train-tokens 300e12   # full-run wall-clock at a 300T-token budget
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
    # Binary units (powers of 1024) with honest IEC labels — a 727.6 TiB figure is 727.6·1024^4
    # bytes (= 800 TB decimal), NOT 727.6e12. The model is binary throughout; label it as such.
    for unit, scale in (
        ("PiB", 1024**5),
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
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
    comm_clustering: str = (
        "clustered"  # selection-clustering assumption for sparse-ring density:
        #   "clustered"    -> W/n  (all local queries select the SAME blocks; optimistic floor)
        #   "independent"  -> union 1-(1-W/n)^(n/p) (queries select independently; upper bound)
    )
    # weight-side memory levers: hierarchical fidelity (quantized-resident / full-on-disk) + offload
    expert_frac: float = (
        0.0  # fraction of TOTAL params that are offloadable/quantizable experts
    )
    weight_quant_bits: float = (
        16.0  # resident expert precision (bf16=16, FP8=8, INT4=4)
    )
    expert_offload_ratio: float = (
        0.0  # fraction of experts parked on NVMe/disk (0 GPU bytes)
    )
    disk_bytes_per_param: float = 2.0  # precision of the full expert copy on disk
    # hardware
    gpu_peak_flops: float = (
        3.5e15  # NVIDIA B300 (Blackwell Ultra) bf16 dense; H100=989e12, B200=2.25e15
    )
    gpu_mem_bytes: float = 288 * 1024**3  # B300 HBM3e (H100=80, B200=192)
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
    node_size: int = 72  # NVL72 NVLink domain (B300); H100 node = 8
    bw_intra_gbps: float = 1800.0  # NVLink-5 per GPU (B300); H100 NVLink ~900
    bw_inter_gbps: float = 100.0  # inter-rack InfiniBand (per GPU)
    inter_node_attn_frac: float = 0.05
    # expert-offload disk I/O
    nvme_bw_per_gpu_gbps: float = 6.0  # local NVMe Gen5 effective read BW per GPU
    offload_fetch_frac: float = (
        1.0  # frac of offloaded experts fetched/step (~1.0: long ctx touches all)
    )
    offload_write_back: bool = (
        False  # True if offloaded experts are TRAINED (read+write doubles I/O)
    )

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
    """Full weights+grads+optimizer+master for the TOTAL params, all-resident at bf16 (the 'before')."""
    return c.params * c.state_bytes_per_param


def resident_state_bytes(c: Config) -> float:
    """GPU-resident state after the weight-side levers: dense backbone full; experts optionally
    quantized (weight_quant_bits) and partially offloaded (expert_offload_ratio) to disk."""
    dense = (1.0 - c.expert_frac) * c.params * c.state_bytes_per_param
    experts_resident = (
        c.expert_frac
        * c.params
        * (1.0 - c.expert_offload_ratio)
        * c.state_bytes_per_param
        * (c.weight_quant_bits / 16.0)
    )
    return dense + experts_resident


def offloaded_disk_bytes(c: Config) -> float:
    """Full-precision expert copies parked on NVMe/disk (off the GPU)."""
    return c.expert_frac * c.expert_offload_ratio * c.params * c.disk_bytes_per_param


def gpus_for_state(c: Config) -> int:
    """Min GPUs to hold the (resident) sharded model state (FSDP/ZeRO-3/expert-parallel)."""
    return max(
        1, math.ceil(resident_state_bytes(c) / (c.gpu_mem_bytes * c.static_mem_frac))
    )


def offload_io_seconds(c: Config, total_gpus: int) -> float:
    """Per-step disk->HBM time to fetch offloaded experts. Offloaded bytes are sharded across the
    cluster (parallel local NVMe). At long context the routed-expert union saturates, so ~all
    offloaded experts are fetched per step (offload_fetch_frac ~ 1.0); write-back doubles it if
    those experts are trained rather than frozen."""
    io_bytes = (
        offloaded_disk_bytes(c)
        * c.offload_fetch_frac
        * (2.0 if c.offload_write_back else 1.0)
    )
    return (io_bytes / max(1, total_gpus)) / (c.nvme_bw_per_gpu_gbps * 1e9)


def kv_bytes(n: int, c: Config) -> float:
    return 2.0 * n * c.d_model * c.bytes_per_elem * c.n_layers / c.kv_compression_ratio


def ring_degree(n: int, c: Config) -> int:
    """Sequence-parallel / ring degree p so each KV shard fits the per-GPU KV budget."""
    return max(1, math.ceil(kv_bytes(n, c) / (c.gpu_mem_bytes * c.kv_mem_frac)))


def dense_scores_bytes_per_layer(n: int, c: Config) -> float:
    return float(n) * n * c.n_heads * c.bytes_per_elem


# --- communication (ring KV rotation per forward, §5.1 + sparse pruning) ---
def comm_density(n: int, c: Config) -> float:
    """Fraction of the KV a device must RECEIVE under the OPTIMISTIC (perfectly clustered) selection
    assumption: all of a rank's local queries select the SAME blocks, so the rank needs only ~W/n of
    the KV. Dense ring = 1.0. This is a LOWER bound — see comm_density_union() for the upper bound."""
    if not c.comm_pruning:
        return 1.0
    return min(1.0, c.pattern_budget / n)


def comm_density_union(n: int, c: Config, p: int) -> float:
    """UPPER bound on the received-KV fraction when a rank's local queries select INDEPENDENTLY
    (uniform). A rank holds ~n/p queries; each picks ~W/n of the KV blocks, so the UNION they
    collectively require saturates: 1 - (1 - W/n)^(n/p). This is the opposite extreme from
    comm_density()'s W/n. Reality is data-dependent and lies BETWEEN the two; the 'compute-bound at
    extreme scale' verdict holds only if selections cluster strongly toward the optimistic end. With
    independent selection the union saturates toward 1.0 (effectively a dense ring) at high n/p."""
    if not c.comm_pruning:
        return 1.0
    per_query = min(1.0, c.pattern_budget / n)
    q = max(1, math.ceil(n / max(1, p)))
    return 1.0 - (1.0 - per_query) ** q


def ring_recv_bytes(n: int, c: Config, p: int) -> float:
    if p <= 1:
        return 0.0
    density = (
        comm_density_union(n, c, p)
        if c.comm_clustering == "independent"
        else comm_density(n, c)
    )
    return kv_bytes(n, c) * (p - 1) / p * density


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


# --- full training run (steps x per-sequence cluster step) -----------------
def cluster_step_seconds(n: int, c: Config) -> dict:
    """Ideal per-sequence wall-clock on the full cluster, decomposed. Strong-scaling lower bound:
    the cluster collaborates on one sequence, so compute (fwd+2bwd) spreads over all GPUs;
    expert-offload disk I/O OVERLAPS compute (max, not sum); exposed sparse-pruned ring comm adds
    on top. Ignores pipeline bubbles, optimizer/all-reduce, data stalls, and restart overhead."""
    p = ring_degree(n, c)
    total_gpus = max(gpus_for_state(c), p)
    ss = train_step_seconds(n, c, sparse=True)
    compute = ss / total_gpus
    io = offload_io_seconds(c, total_gpus) if offloaded_disk_bytes(c) > 0 else 0.0
    comm = (
        3.0 * comm_time_hier(n, c, p) if p > 1 else 0.0
    )  # fwd + 2 bwd, hier sparse-pruned ring
    return {
        "total_gpus": total_gpus,
        "compute": compute,
        "io": io,
        "comm": comm,
        "step": max(compute, io) + comm,
    }


def full_run_seconds(n: int, c: Config, tokens: float) -> dict:
    """Wall-clock to train on `tokens` total tokens at context length n: (tokens / n) sequences,
    each one ideal cluster step. Aggregate compute is the classic 6·N_active·D plus sparse
    attention; the token budget D is the dominant (and least a-priori) assumption."""
    cs = cluster_step_seconds(n, c)
    n_steps = tokens / n
    wall = n_steps * cs["step"]
    return {**cs, "n_steps": n_steps, "tokens": tokens, "wall_seconds": wall}


# --- report ----------------------------------------------------------------
def report(c: Config, contexts: list[int], train_tokens: float = 0.0) -> None:
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
    disk = offloaded_disk_bytes(c)
    print(
        f"  model state: full {human_bytes(model_state_bytes(c))} (all-resident bf16) -> "
        f"resident {human_bytes(resident_state_bytes(c))} -> {gpus_for_state(c):,} GPUs"
        + (
            f"  (+ {human_bytes(disk)} on disk; experts={c.expert_frac:g}, "
            f"quant={c.weight_quant_bits:g}b, offload={c.expert_offload_ratio:g})"
            if (c.expert_frac > 0 and (disk > 0 or c.weight_quant_bits != 16))
            else ""
        )
    )
    print(f"  crossover n ~= {crossover_n(c):,.0f}")
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
        f"  compute: 1-GPU-equiv step (fwd+2bwd) dense {human_time(ds)} vs sparse {human_time(ss)} ({ds / ss:.0f}x);"
    )
    print(
        f"           ideal cluster step over {total_gpus:,} GPUs ≈ {human_time(ss / total_gpus)}/seq "
        f"(strong-scaling lower bound; ignores comm, pipeline bubbles, and expert-offload disk I/O)"
    )
    if p > 1:
        # comm without pruning (dense ring) vs with pruning (sparse ring), and — for the pruned
        # ring — the optimistic CLUSTERED density (W/n) vs the INDEPENDENT-selection union bound.
        dens, clk = c.comm_pruning, c.comm_clustering
        c.comm_pruning = False
        dense_flat, dense_hier = comm_time_flat(n, c, p), comm_time_hier(n, c, p)
        c.comm_pruning = True
        c.comm_clustering = "clustered"
        spr_flat, spr_hier = comm_time_flat(n, c, p), comm_time_hier(n, c, p)
        d_clustered = comm_density(n, c)
        c.comm_clustering = "independent"
        uni_hier = comm_time_hier(n, c, p)
        d_union = comm_density_union(n, c, p)
        c.comm_pruning, c.comm_clustering = dens, clk
        print(f"  comm /forward (per-GPU fwd compute ≈ {human_time(t_comp)}):")
        print(
            f"    dense ring : flat {human_time(dense_flat)} | hierarchical {human_time(dense_hier)}"
        )
        print(
            f"    sparse ring, CLUSTERED selection (density {d_clustered:.2g}): "
            f"flat {human_time(spr_flat)} | hierarchical {human_time(spr_hier)}"
        )
        print(
            f"    sparse ring, INDEPENDENT selection (union density {d_union:.2g}): "
            f"hierarchical {human_time(uni_hier)}"
        )
        bound_c = "COMPUTE-bound ✓" if spr_hier <= t_comp else "still COMM-bound"
        bound_u = "COMPUTE-bound ✓" if uni_hier <= t_comp else "COMM-bound"
        print(
            f"    -> hier ring + pruning + {c.kv_compression_ratio:g}x KV compression: "
            f"clustered {bound_c} | independent-selection {bound_u}  "
            f"(reality is between; compute-bound REQUIRES selection to cluster)"
        )
    else:
        print("  comm: p=1, no ring communication.")
    if offloaded_disk_bytes(c) > 0:
        io = offload_io_seconds(c, total_gpus)
        step = ss / total_gpus
        verdict = (
            "hidden behind compute ✓"
            if io <= step
            else f"I/O-BOUND (+{human_time(io - step)}/step)"
        )
        print(
            f"  expert-offload I/O: {human_bytes(offloaded_disk_bytes(c) * c.offload_fetch_frac)}/step "
            f"over {total_gpus:,} GPUs @ {c.nvme_bw_per_gpu_gbps:g} GB/s/GPU -> {human_time(io)}/step "
            f"vs {human_time(step)} compute -> {verdict}"
        )
    print("=" * 112)
    if train_tokens > 0:
        fr = full_run_seconds(n, c, train_tokens)
        yrs = fr["wall_seconds"] / 3.15576e7  # Julian year of seconds
        print(
            f"FULL RUN to D={train_tokens:.3g} tokens @ context {n:,}  "
            f"({fr['n_steps']:,.0f} sequences over {fr['total_gpus']:,} GPUs):"
        )
        print(
            f"  per-seq cluster step = max(compute {human_time(fr['compute'])}, "
            f"I/O {human_time(fr['io'])}) + comm {human_time(fr['comm'])} = {human_time(fr['step'])}"
        )
        print(
            f"  total wall-clock ≈ {human_time(fr['wall_seconds'])}  (~{yrs:,.2f} years)  "
            f"[ideal strong-scaling floor; real runs 1.5-3x this from bubbles/stalls/restarts]"
        )
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
    p.add_argument(
        "--comm-clustering",
        choices=["clustered", "independent"],
        default="clustered",
        help="sparse-ring density assumption: clustered=W/n (optimistic floor), "
        "independent=union 1-(1-W/n)^(n/p) (upper bound)",
    )
    p.add_argument(
        "--gpu",
        choices=["h100", "b200", "b300"],
        default="b300",
        help="hardware preset",
    )
    p.add_argument(
        "--gpu-peak-flops",
        type=float,
        default=None,
        help="override preset bf16 dense FLOP/s",
    )
    p.add_argument(
        "--gpu-mem-gb", type=float, default=None, help="override preset HBM GB"
    )
    p.add_argument("--state-bytes-per-param", type=float, default=16.0)
    p.add_argument(
        "--expert-frac",
        type=float,
        default=0.0,
        help="fraction of params that are experts",
    )
    p.add_argument(
        "--weight-quant-bits",
        type=float,
        default=16.0,
        help="resident expert precision",
    )
    p.add_argument(
        "--expert-offload-ratio",
        type=float,
        default=0.0,
        help="fraction of experts on disk",
    )
    p.add_argument("--disk-bytes-per-param", type=float, default=2.0)
    p.add_argument("--mfu-dense", type=float, default=0.50)
    p.add_argument("--mfu-sparse-fwd", type=float, default=0.45)
    p.add_argument("--mfu-sparse-bwd", type=float, default=0.38)
    p.add_argument(
        "--node-size", type=int, default=None, help="override preset NVLink-domain size"
    )
    p.add_argument(
        "--bw-intra", type=float, default=None, help="override preset intra-node GB/s"
    )
    p.add_argument("--bw-inter", type=float, default=100.0)
    p.add_argument("--inter-node-frac", type=float, default=0.05)
    p.add_argument(
        "--nvme-bw", type=float, default=6.0, help="local NVMe read GB/s per GPU"
    )
    p.add_argument("--offload-fetch-frac", type=float, default=1.0)
    p.add_argument(
        "--offload-write-back",
        action="store_true",
        help="offloaded experts trained (2x I/O)",
    )
    p.add_argument("--contexts", type=str, default="8192,32768,131072,1048576")
    p.add_argument(
        "--train-tokens",
        type=float,
        default=0.0,
        help="if >0, estimate full-run wall-clock to train on this many total tokens",
    )
    a = p.parse_args()
    # (bf16 dense FLOP/s, HBM GB, NVLink GB/s, NVLink-domain size); explicit flags override the preset
    presets = {
        "h100": (989e12, 80.0, 900.0, 8),
        "b200": (2.25e15, 192.0, 1800.0, 72),
        "b300": (3.5e15, 288.0, 1800.0, 72),
    }
    pk, mem, nvl, node = presets[a.gpu]
    peak = a.gpu_peak_flops if a.gpu_peak_flops is not None else pk
    mem_gb = a.gpu_mem_gb if a.gpu_mem_gb is not None else mem
    bw_intra = a.bw_intra if a.bw_intra is not None else nvl
    node_size = a.node_size if a.node_size is not None else node
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
        comm_clustering=a.comm_clustering,
        expert_frac=a.expert_frac,
        weight_quant_bits=a.weight_quant_bits,
        expert_offload_ratio=a.expert_offload_ratio,
        disk_bytes_per_param=a.disk_bytes_per_param,
        gpu_peak_flops=peak,
        gpu_mem_bytes=mem_gb * 1024**3,
        state_bytes_per_param=a.state_bytes_per_param,
        mfu_dense=a.mfu_dense,
        mfu_sparse_fwd=a.mfu_sparse_fwd,
        mfu_sparse_bwd=a.mfu_sparse_bwd,
        node_size=node_size,
        bw_intra_gbps=bw_intra,
        bw_inter_gbps=a.bw_inter,
        inter_node_attn_frac=a.inter_node_frac,
        nvme_bw_per_gpu_gbps=a.nvme_bw,
        offload_fetch_frac=a.offload_fetch_frac,
        offload_write_back=a.offload_write_back,
    )
    return cfg, [int(x) for x in a.contexts.split(",")], a.train_tokens


if __name__ == "__main__":
    cfg, contexts, train_tokens = parse_args()
    report(cfg, contexts, train_tokens)
