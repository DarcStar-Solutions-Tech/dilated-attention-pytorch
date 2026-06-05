# Unified Attention Architecture (v2)

> **Status:** design proposal / RFC — **v2**, revised after a prior-art research pass.
> Supersedes **v1** (`docs/archive/unified-attention-architecture-v1.md`).
> **What changed from v1:** v1 proposed hand-writing a Triton Flash kernel and inventing a
> content-adaptive sparse policy. Research showed both are **already solved** by mature work —
> PyTorch **FlexAttention** + **FlashAttention-3** (the kernel/static-pattern substrate) and
> **DeepSeek NSA** (the trainable content-adaptive policy). v2 is a **build-vs-buy** design:
> depend on / port the solved layers, and concentrate our originality on the one piece nobody
> has shipped — **correct sparse *distributed/ring* attention** — plus a **Muon** training recipe.
> **Goal (unchanged):** reduce transformer TRAINING cost at long context — sub-quadratic
> compute + O(n/p) memory with no forced quality ceiling.
> **Evidence base:** `docs/references/` (cached papers + repo pointers); cost model in
> `analysis/attention_cost_analysis.py`. Confidence per item is marked ✓ (fact-checked) or
> ◐ (background / pending fact-check) in §9.

---

## 1. Motivation (unchanged)

A correctness audit (`docs/reports/correctness-audit-main-2026-05-30-0219-UTC.md`) found that
nearly every long-sequence attention variant in this repo failed in the **same place**:
combining attention computed over *pieces* of the key space without a shared softmax
denominator. The fix — and the foundation — is one correct **online softmax** (FlashAttention's
running-max `m` / running-denominator `ℓ` / rescaled output `O`), which is **associative and
commutative** and therefore unifies block-sparse, dilated, Flash, and ring attention. That
insight stands. What v2 changes is **who builds each layer.**

## 2. Prior art — what is already solved (the v2 pivot)

| Layer / component | Already solved by | Our action | Conf. |
|---|---|---|---|
| Fused Flash kernel for **static/structured** patterns (L1, static L2) | **FlexAttention** (`torch.compile` → fused FA kernel, `BlockMask`, fwd+bwd, ~90% FA2) + **FlashAttention-3** backend | **DEPEND-ON** — do not hand-roll | ✓ |
| **Trainable content-adaptive** sparse policy (L2 adaptive) | **DeepSeek NSA** (compression + learned top-k selection + sliding window; 9×/6× @64k; quality ≥ full attn) | **PORT / LEARN-FROM** (fla-org / lucidrains, both MIT) | ✓ |
| Dense/long-context kernel speed | **FA3** (Hopper, beta) / **FA4** (Blackwell, alpha) | **ADOPT** FA3 / **TRACK** FA4 | ✓ |
| Optimizer efficiency (orthogonal to attention) | **Muon** (2D-matrix Newton–Schulz; ~52% AdamW FLOPs; 1T-scale via MuonClip) | **ADOPT** in the training recipe | ✓ |
| Online-softmax accumulator (L0) | standard math; we already have a correct one (`_merge_block_attention`) | **KEEP** (ours) | ✓ |
| **Sparse cross-device ring / distributed** reduction (L3) | **nobody** (NSA/FlexAttention are effectively single-device; rely on framework context-parallel) | **BUILD** — our distinct contribution | ✓ (gap confirmed) |

**Net repositioning:** the library's value moves from *"build the kernels and invent a policy"*
(now commodity) to *"compose the published best-in-class policy (NSA-family) on FlexAttention/FA3,
add the genuinely-missing piece — correct **sparse distributed/ring** attention — and ship a
Muon-based long-context training recipe."* Smaller scope, sharper, more defensible.

> **The one decision-relevant nuance — now RESOLVED (§9):** FlexAttention *can* do data-dependent
> learned selection — `mask_mod` encodes a learned top-k into a `BlockMask` that skips **real
> fwd+bwd compute** (~2× on 50%-sparse causal). The catch is the **per-step `create_block_mask`
> rebuild** (the "hardest, non-amortizable" case, ~hundreds of µs/call, ~10× reducible via
> `_compile`). So: **FlexAttention for static patterns AND as a learned-selection fallback;
> dedicated kernels (FlashMoBA / NSA) for the learned path when the mask-build cost matters.**

## 3. Core insight (unchanged)

| "Variant" | The *pieces* | Combination |
|---|---|---|
| Block-sparse | selected KV blocks | online softmax |
| Dilated (LongNet) | strided KV subsets | online softmax |
| Flash | KV tiles in SRAM | online softmax **is** Flash |
| Ring / sequence-parallel | KV shards per rank | online softmax across the network |

Build one correct online-softmax engine; make block-sparsity, dilation, causality, and ring
all **descriptors of which KV pieces a query block sees.** Associativity is why the same engine
scales from a single-GPU sparse kernel to a multi-node ring with no change to the math.

## 4. The architecture (v2 — build-vs-buy per layer)

```
L0  AttentionAccumulator   online-softmax merge (m, ℓ, O); associative; the ONLY place
   [BUILD — ours]          softmax-combination lives. Pure, TO BE gradcheck-tested (the seed
                           has NO gradcheck coverage yet — it is a Phase-0 gate, not a current
                           property). Seed: _merge_block_attention (sparse/block_sparse_attention.py,
                           PR #29). L0 should also consolidate the two existing LSE accumulators
                           — StableRingAccumulator (ring/utils/ring_attention_lse.py) and
                           StableAttentionAccumulator (ring/hilbert/…) — or supersede them.

L1  Flash kernel           STATIC patterns -> DEPEND-ON FlexAttention (+ FA3/FA4 backend).
   [BUY: FlexAttention]    LEARNED block selection -> PORT FlashMoBA (mit-han-lab, BSD-3): a
   [PORT: FlashMoBA/NSA]     gather-densify-scatter FA2 kernel with real fwd+bwd savings. NSA
                            (fla-org) is the richer alternative. FlexAttention also works for
                            learned selection (per-step BlockMask rebuild cost). No hand-rolled kernel.

L2  SparsityPolicy         Canonical (block-level) policy = MoBA-style content-adaptive routing:
   [PORT: FlashMoBA          score each KV block by its centroid (q · mean(keys_in_block)), select
    + BUILD interface]       top-k blocks/query. This is EXACTLY our "block-centroid top-k" idea,
                             now validated + open (FlashMoBA). Union with a static multi-scale
                             dilated-stride + local + global skeleton (FlexAttention BlockMask).
                           Alternative policies behind one interface: NSA (compression + selection
                             + sliding window, block-level) and DSA (token-level lightning-indexer
                             top-k on MLA — the DeepSeek production path; finer-grained, MLA-coupled).
                           Routing is HIERARCHICAL-capable: flat top-k for moderate context,
                             multi-level (super-block -> block) at extreme scale to cut SELECTION's
                             quadratic term (two-level = large constant factor; genuine sub-quadratic
                             needs the unbuilt log-depth recursion, §5.2).

L3  ExecutionStrategy      single-GPU (loop) | HIERARCHICAL RING / sequence-parallel: a
   [BUILD — ours, the      topology-aware multi-level ring — DENSE intra-node ring, SPARSE
    distinct value]        inter-node ring — rotate KV shards, reduce partials via L0, SKIP
                           communicating shards no local query selects, load-balanced
                           block->rank assignment. Global-position causal masking. (§5.1)

L4  Module API + factory   nn.Module wrappers, MAGNETO init, the existing factory.
   [BUILD — ours]
```

Then: dense/static = `Flash(policy=Static, backend=FlexAttention+FA3)`; NSA-style =
`Flash(policy=NSA)`; **sparse ring** = `Flash(policy=NSA|Static, strategy=Ring)`. One correct
substrate, many configs.

## 5. Flagship variant: NSA-policy × sparse-ring

Adopt **NSA's three-branch policy** (compression + learned selection + sliding window) as the
canonical content-adaptive policy — it is published, trainable end-to-end, and validated at
quality ≥ full attention. Optionally union it with a **multi-scale dilated block-stride
skeleton** (our LongNet heritage) for guaranteed `O(log n)` long-range coverage. Run it on the
FlexAttention/FA3 backend for static branches and ported NSA kernels for the learned-selection
branch, and wrap it in our **L3 hierarchical sparse-ring** strategy for multi-node scale. This is
the combination no single existing project ships.

### 5.1 The hierarchical (topology-aware) sparse ring — L3 in depth

A flat ring treats every GPU↔GPU hop as equal, but real clusters are tiered: intra-node links
(NVLink) are ~an order of magnitude faster than inter-node (InfiniBand/Ethernet). At long context
the ring's **communication**, not its compute, is the bottleneck (the MFU killer at 1M+ tokens), so
L3 is **hierarchical** — it maps attention density onto the physical communication levels:

- **Intra-node ring (dense):** GPUs within a node exchange KV over fast links → keep these
  interactions high-density.
- **Inter-node ring (sparse):** cross-node hops are expensive → attend across nodes only sparsely,
  so the slow links carry little traffic. The sparse block index does double duty here: it *prunes
  inter-node communication* to only the shards a node's queries actually select.
- **Load-balanced block→rank assignment (a BUILD item, not reused):** content-adaptive sparse/causal
  patterns are irregular *and data-dependent* — which blocks each rank's queries select changes every
  step — so query blocks attend to differing, time-varying numbers of key blocks and create *dynamic*
  ring-step stragglers. The remedy is a **per-step (or per-window) rebalancing** of selected-block→rank
  assignment driven by the live routing scores. This needs a cross-rank collective the existing
  generator does **not** have (see below); it is the hardest unbuilt piece of L3, not prior art.

This **partially reuses the project's existing `HierarchicalSparsePatternGenerator`**
(`sparse/sparse_pattern_generator.py`; guide: `docs/guides/hierarchical-patterns-guide.md`). Be precise
about what it does and does not provide: it gives the **local / global / inter-node pattern levels**,
**per-rank sparsity self-regulation** (each rank raises/lowers its own pattern density from its local
timing history — *not* cross-rank balancing), and **node-size detection** (default **8**, which must be
reconfigured to the **NVL72 domain = 72** for the §6 hierarchical-ring comm model to hold). We lift its
**topology mapping** into L3 and pair it with the corrected online-softmax reduction (L0). It is *part*
of the answer to the two tensions a flat sparse-ring leaves open — **interconnect cost** and **load
imbalance** — but the **cost-based cross-rank block→rank assignment is a genuine BUILD item** (§8 Phase
3): the generator self-regulates per-rank sparsity, it does not assign blocks to ranks, and a static
once-measured assignment cannot track data-dependent selection. *Caveat: the class that currently
consumes the generator, `BlockSparseRingDistributedDilatedAttention`, carries audited correctness bugs
— the topology mapping is reusable, but the consuming path must be rebuilt on the corrected L0/L1.*

**Causal-centroid correctness under the ring.** MoBA notes that mean-pooling a block to a centroid can
leak *future* tokens for the boundary block straddling a query's causal frontier (it force-routes and
specially masks the current block). Under the sparse ring that boundary block's centroid may live on a
*different* rank than the querying rank, so masking the pooled-future contribution requires the
centroid-computing rank to know the querying rank's **global** causal position — a cross-rank
correctness requirement that "global-position causal masking" must explicitly discharge (Phase-3 gate).

Hierarchical *ring* (communication, §5.1) and hierarchical *routing* (selection, §5.2 below) are
the two extreme-scale extensions of this design.

### 5.2 Hierarchical selection routing — L2 at scale

Content-adaptive selection must *score* candidates to pick the top-k, and that scoring is not free
(§6). **Flat** routing — score every (query, key-block) pair — is `O(n²/b)`: fine at ≤1M tokens
(~6% overhead), but at extreme context it becomes a large, *new* quadratic term. (How large is
config-dependent: at 1B tokens for the 500T/1T config, block-level selection grows to **~half the core
attention** — 295 vs 590 EFLOP, same order of magnitude; at smaller model dims, where the core is
cheaper, it *dominates* the core outright. Either way it stops being negligible.) So L2's routing is
**multi-level**, mirroring the hierarchical ring on the selection side:

- **Coarse prefilter:** group blocks into *super-blocks*, score query against super-block centroids,
  keep the top few super-blocks — `O(n²/b²)` or less.
- **Fine selection:** run block-level top-k *only within* the surviving super-blocks.
- **What the cost model actually computes is the two fixed levels above:** `O(n²/(b·S)) + O(n)` —
  still **quadratic** in `n`, but with the constant divisor `b·S` (≈16× smaller selection term:
  295 → 18 EFLOP at 1B; the headline 15,887× is this *single*-prefilter result). Genuinely
  *sub-quadratic* selection needs **true log-depth recursion** (recurse the prefilter for `~log_S n`
  levels) — described here but **not yet implemented or costed**. A constant-factor reduction of a
  quadratic is not an asymptotic-class change; the doc distinguishes the two.

This is a natural extension of mechanisms already in the field: **NSA's coarse token/block
*compression* branch is itself a one-level coarsening** that a hierarchical router generalizes, and
the project's **`HierarchicalSparsePatternGenerator`** already expresses multi-level (local / global
/ inter-node) structure we can reuse for the routing hierarchy as well as the ring. The flat
MoBA/FlashMoBA router is the **default** (correct and fast through ~1M tokens); the multi-level
router is the **scale-out path**, exposed behind the same L2 policy interface so it is opt-in by
context length. *Quality note:* coarsening risks missing a relevant block whose super-block scored
low — mitigated by training the router end-to-end (and the dense-imitation distillation in §7), and
bounded by the same dropped-mass certificate.

Symmetry to remember: the **hierarchical ring** cuts *communication* and the **multi-level router**
cuts *selection* — as modeled, each by a large **constant factor** (the two-level forms), with true
sub-quadratic behavior available only from the unbuilt recursive variants. Both levers are required
to reach the extreme-context regime the cost model describes; the flat versions are correct and
sufficient up to ~1M tokens.

### 5.3 Selector gradient flow — how the router is actually trained

The whole quality case (§7) rests on training the selector end-to-end, yet **top-k block selection is
non-differentiable** and the design must say how gradient reaches the score head. The validated
references each solve this a *specific* way, and we must pick one explicitly rather than assume it:

- **NSA** states plainly that top-k selection is non-differentiable and that an auxiliary-loss
  importance head "often degrades performance"; it sidesteps this by deriving selection scores from
  the **differentiable compression-branch softmax** — gradient flows into the compression MLP, not
  through the discrete top-k.
- **MoBA** uses a **hard 0/1 gate** `g = 1[s ∈ Topk]` with parameter-less centroid scores and **no**
  straight-through estimator and **no** aux loss; it trains only because the same Q/K that produce the
  scores also produce the attention over the *selected* blocks — unselected blocks simply receive zero
  gradient (gradient starvation is accepted, not fixed).

Our default (FlashMoBA-style centroid top-k) inherits MoBA's mechanism. But the **bespoke pieces have
no such free ride** and need a stated mechanism: the **multi-level router** (§5.2) prunes whole
super-blocks *before* scoring their members (those members get no gradient at all), and **novelty-
weighted routing** (§11 Lever D) adds a learned surprisal term to the score head. For both, specify:
is the score head differentiable, is there a straight-through estimator or an MoE-style
load-balancing aux loss, and **how do non-selected candidates receive gradient**. The Phase-2 gate
must check the router *moves* selections during training (per-block selection-frequency drift), not
only final-loss parity — a router that never changes its picks has silently stopped learning.

## 6. Cost model (refined post-research)

Reproduce with `python analysis/attention_cost_analysis.py` — now models **selection granularity**
(flat block / token / hierarchical), **split fwd/bwd MFU**, **MLA KV compression**, and **ring
communication**. 7B-class model (`d_model=4096`, 32 layers, bf16), core budget `W=4096`,
**block-level** selection (`b=64`), one **NVIDIA B300** ("Blackwell Ultra": 3.5 PFLOP/s bf16 dense,
288 GB HBM3e, NVLink-5 — the current top part; `--gpu h100`/`--gpu b200` switch presets):

```
   context n |  dense attn | sparse core |   selection | eff attn x |     KV/seq | ring deg
     131,072 |  9.01 PFLOP | 281.47 TFLOP |  2.20 TFLOP |        32x |   64.0 GiB |        1
   1,048,576 | 576.46 PFLOP |  2.25 PFLOP | 140.74 TFLOP |       241x |  512.0 GiB |        4
```
Training step (fwd + 2·bwd) at 1M: dense **16.9 min** vs sparse **36.5 s** → **~28×/step**
(1-GPU-equiv FLOPs/MFU; the B300's 288 GB HBM also drops the KV ring degree **13→4** vs the 80 GB H100).
Spread over the 4 ring GPUs, the ideal cluster step is **~9 s/seq** (strong-scaling floor).

Three refinements the research forced (all in the calculator):

1. **Selection is not free, and its granularity decides the ceiling.** Content-adaptive routing
   must *score* candidates. **Block-level** (MoBA / our lane) adds `O(n²/b)` — only ~6% at 1M, so
   the effective speedup stays **241×**. **Token-level** (DeepSeek DSA lightning indexer) adds
   `O(n²)` — at 1M it *dominates* the core attention and **caps the effective speedup at ~51×**
   (`--selection token`). → a concrete compute argument for staying **block-level**: ~`b`× cheaper
   routing. (DSA keeps its indexer cheap via FP8 + few heads precisely to fight this `O(n²)` term.)
   At **extreme context, even block-level routing becomes a wall** — and **hierarchical** routing
   (`O(n²/(b·S)) + O(n)`, §5.2) fixes it: at **1B tokens** it cuts the selection term **~16×**
   (295 → 18 EFLOP), lifting the effective attention speedup from 10,923× (flat block) to
   **15,887×** (`--selection hierarchical`).
2. **MFU splits forward vs backward.** Block-dense kernels (FlashMoBA / FlexAttention) keep ~90% of
   dense forward MFU but weaker (~85%) backward; the calculator uses `mfu_sparse_fwd`/`_bwd` for the
   training step. Net: realized speedup is *closer* to the `n/W` ceiling than the old conservative
   40% guess — but backward, not forward, is the efficiency floor to engineer.
3. **MLA KV compression is a second, multiplicative memory lever.** `--kv-compression 8` shrinks KV
   512 GiB → 64 GiB at 1M, dropping the ring degree **p = 4 → 1** (fits one GPU; 4 is the B300
   uncompressed degree at 1M, matching the table above and the 13→4 drop vs H100), compute unchanged.
   Inference-oriented (deprioritized for training *compute*), but it directly attacks the *memory*
   wall that otherwise forces ring sharding.
4. **Communication is the real long-context bottleneck — the hierarchical ring attacks it (§5.1).**
   Ring attention rotates ~the whole KV past each device per forward. A **flat** ring serializes
   that over the slow inter-node link; a **hierarchical** ring keeps the dense rotation on fast
   intra-node (NVLink) links and sends only the sparse inter-node fraction over slow links. At
   7B/1M forced across nodes (`--node-size 2`): ~410 GB/GPU/forward → **flat 4.1 s vs hierarchical
   0.42 s (~10× less)** (B300 NVLink-5 widens the intra-node advantage). Still compute-bound at 1M,
   but comm grows with context until it dominates — the reason L3 is hierarchical, not flat.

- **Crossover ≈ 27K tokens** unchanged: beyond it, dense attention exceeds the whole linear term.
- **Compute (sparse + hierarchical selection), memory (ring × KV-compression), and communication
  (hierarchical ring + sparse-ring pruning) are solved by *different* levers** — you need all of them
  at scale. Externally corroborated: NSA 9×/6× @64k; DSA `O(L²)→O(Lk)` at 1T scale.

**Extreme-scale sanity check (500T-total / 1T-active MoE, 1B-token context, on B300s).** The
calculator now shards weights+optimizer (7.1 PiB → **~37k B300s** to hold the state) and models
sparse-ring pruning, so it is valid here. With hierarchical routing + MLA-64× + sparse-ring pruning:
the selection wall shrinks **~16×** (295 → 18 EFLOP, eff attn 10,923× → 15,887×), and communication —
the binding constraint — collapses from **dense-ring ~23 min/forward** (flat) → **~2.4 min**
(hierarchical) → **~0.01 s** (hierarchical + pruning), i.e. **compute-bound** at ~47 s/GPU/forward
(ideal cluster step ~2.7 min/seq over the 37k GPUs).
**Caveat — the `0.01 s` assumes *clustered* selection.** That density (`6e-5`) holds only if all of a
rank's ~1.18M local queries select the *same* blocks. Under **independent** selection the received
union saturates (`1−(1−W/n)^(n/p) → 1.0`) and the hierarchical ring reverts to its **~2.4 min**
rotation — **comm-bound** at the 37k-GPU floor. Reality is between the two; "compute-bound" at this
floor *requires* selections to cluster strongly (`--comm-clustering clustered|independent` brackets
it). Notably the §11 **offload lever incidentally de-risks this**: at the 2.7k-GPU floor per-GPU
compute (~10.7 min) exceeds even the independent-selection ~2.4 min comm, so that configuration stays
compute-bound *regardless* of clustering.
*Reading:* hierarchical ring alone leaves you comm-bound; it's **sparse-ring pruning** (only sending
selected blocks) — when selection clusters — that makes the regime, and the "~90% of optimal"
assumption, reachable. (`--params 500e12 --active-params 1e12 --d-model 16384 --layers 128
--pattern-budget 65536 --block-size 128 --selection hierarchical --kv-compression 64 --contexts
1073741824` — the model-dimension flags are required; without them the calculator runs the default
7B dims and prints different numbers.) With attention thus handled, the **binding constraint becomes
the weight memory** (the ~37k-GPU state floor) — addressed by the §11 weight-side levers, which trade
GPU count for wall-clock and keep expert-offload disk I/O hidden behind compute.

### 6.1 Full training run — wall-clock vs. token budget

`--train-tokens D` extends the per-step model to a whole run: `wall = (D / context) × per-seq cluster
step`, where the per-seq step is the strong-scaling floor (`max(compute, offload-I/O) + exposed comm`).
For the 500T/1B config the per-seq step is **2.7 min over ~37k B300s** (no offload) or **36 min over
~2.7k B300s** (offload). The step is *determinable a priori*; the **run is not** — it is governed by
`D`, the token budget to call 500T params "completely trained," which is a scaling-law / data question,
not a hardware one. Verified spread (six independent derivations, each adversarially checked):

| D (tokens) | reading | 37k GPUs | 2.7k GPUs |
|---|---|---|---|
| 20T | Chinchilla compute-optimal **floor** (~20 tok/active-param) | 34 d | 1.3 yr |
| **100–500T** | **defensible "completely trained" band** (frontier over-training on *active* params; ~0.3–1.7 epochs of the human-text stock) | **0.5–2.4 yr** | 6–32 yr |
| 10 quadrillion (1e16) | "fill all 500T params" (Chinchilla-on-*total*) — **infeasible** | 47 yr | 638 yr |

(Ideal floor; real runs ~1.5–3× from pipeline bubbles, optimizer all-reduce, data stalls, restarts.)

Three findings the derivations converged on: (1) **there is no crisp "completely trained" point** —
loss is a power law `L = E + A/Nᵃ + B/Dᵇ` with irreducible floor `E ≈ 1.8 nats`, never flat, so "done"
is a budget choice (Kaplan: compute-optimal stops "significantly before convergence"). (2) **You cannot
fully exercise 500T params on existing data** — Chinchilla-on-total needs ~1e16 tokens ≈ **33× the
entire ~300T-token stock of human text** (Epoch/Villalobos 2024, CI 100–1000T); top-k routing trains
only the ~1T active params per token and routing gains saturate (Clark 2022), so the model is
structurally over-parameterized — the **binding constraint is data, not time or FLOPs**. (3) The
realistic budget (~100–500T tokens, anchored to DeepSeek-V3 ≈400 / Kimi K2 ≈484 tok/active) sits inside
the ~4-epoch near-lossless repetition window (Muennighoff 2023), giving **~0.5–2.4 yr on ~37k B300s**
(×1.5–3 real-world). Offload trades that for ~6–32 yr on ~2.7k GPUs — a capex-vs-wall-clock knob.

### 6.2 What the ideal floor omits (the ×1.5–3 factor, unpacked)

The per-seq step is a strong-scaling *lower bound*; the blanket "×1.5–3 real" folds several
unmodeled, load-bearing realities a real run must budget explicitly — it is asserted, not derived.

- **Global batch & data-parallelism — the scaling story is incomplete.** The model fixes **one
  sequence per optimizer step** (`n_steps = D/context`); at 1B context that is a ~1B-token global
  batch, **~50–500× the empirical critical batch size**, so most of that step's gradient signal is
  wasted (gradient-noise saturation). The §11.1 "more GPUs → near-linear" speedup is **data-parallel**
  — replicating the whole state floor across many strong-scaling groups — a *different* axis the cost
  model has no term for (`n_steps` is context-only). "~37k GPUs → ~5 months" means **~10 DP replicas of
  the 3.7k-GPU group**, not 37k GPUs on one sequence. A real recipe sets a sane global batch (a few M
  tokens), which fixes the optimizer-step count independently of context; reconcile the two before
  trusting any wall-clock.
- **Fault tolerance is first-order here, not a rounding factor.** A multi-year run on thousands of GPUs
  sees continuous failures; with **727 TiB+ resident state (+ ~778 TiB offloaded experts)** to
  checkpoint and re-shard, the checkpoint interval vs cluster MTBF, lost-work-per-restart, and
  offloaded-shard re-replication are first-order costs. Needs an explicit checkpoint / elastic-restart
  design and a "resume to bit-exact optimizer state; checkpoint overhead < X% of step" gate.
- **TCO & power are unpriced.** The framework optimizes "GPUs × wall-clock" but never the two axes that
  decide fundability: **capex** (dollars for 2.7k–37k B300s) and **energy** (tens of MW, power +
  cooling). The "capex-vs-wall-clock knob" has neither axis priced — it needs at least order-of-magnitude
  bands to be a real choice.
- **The frontier moves during the run.** At the doc's own cited rate (**~4.7×/yr**, Epoch AI), a ~3-yr
  ideal run is lapped ~70× and an ~8-yr one by hundreds× at completion, on hardware 1–2 generations
  stale. A fixed multi-year target needs an obsolescence / time-value argument (or a reason the 50T
  knowledge shell is durable while the moving frontier is not).

(Two further unmodeled load-bearers — MoE expert-routing comm/balancing, and data-corpus
quality/governance — are covered in §11 and §11.1.)

## 7. What is determinable a priori (refined post-research)

- **Compute — exact, minus a now-quantified selection term.** Core attention speedup `= n/W`; the
  realized ceiling subtracts the selection cost — `O(n²)` token, `O(n²/b)` flat-block, or
  `O(n²/(b·S))+O(n)` **two-level hierarchical** (as modeled: still *quadratic*, reduced by the
  constant `b·S`; genuinely sub-quadratic only with the unbuilt recursion, §5.2) — scaled by fwd/bwd
  MFU. Validated by NSA's measured 9×/6× @64k and DSA's `O(L²)→O(Lk)`.
- **Memory — exact, with a second lever.** non-Flash scores `O(n²)` per layer (Flash removes); KV
  `O(n)` → `O(n/p)` via ring → `O(n/(p·r))` with an MLA latent compression ratio `r`. Selection
  does **not** shrink KV; only representation compression does. The separate **weight** memory wall
  (the binding constraint at extreme scale) is attacked by the §11 levers — hierarchical fidelity
  (quantized-resident / full-on-disk experts), offload, and overlapping/compositional experts —
  **orthogonal** to all attention-map sparsity.
- **Communication — modeled, tiered, prunable.** Ring rotates KV/device per forward; three levers:
  **(1) hierarchical ring** keeps the dense rotation on fast intra-node links, sparse fraction on slow
  inter-node; **(2) sparse-ring pruning** — only the KV blocks some local query *selects* are sent, so
  comm volume is `~W/n` of the KV, **not** the whole KV; **(3) MLA** shrinks what is moved. Pruning is
  the dominant lever and is what **flips an extreme-context run from comm-bound to compute-bound**
  (§6). (Model assumes weights/optimizer are **sharded** across the cluster, not replicated.)
- **Quality — still not a tight a-priori number, but the priors hardened.** Provable: a **connected**
  pattern (our skeleton gives `O(log n)` reach) retains universal approximation — no forced ceiling.
  Per-token error bounded by dropped softmax mass: `‖o−õ‖ ≤ 2·δ·max‖v‖`, useful only *if* mass
  concentrates — and **that assumption is now empirically validated**: NSA/DSA/MoBA match-or-beat
  dense at frontier scale (NSA +0.032 LongBench; DSA ≈-parity at 1T). End-task loss stays **empirical**
  (the model trains *with* the pattern), gated by loss parity + a **runtime dropped-mass certificate**
  (`δ̂` from routing scores). *Caveat — `δ̂` is not self-validating:* it is estimated from the router's
  own scores over the blocks the router *chose to score*, so it is structurally blind to mass in
  blocks pruned before scoring (and, under hierarchical routing, to entire super-blocks never scored at
  the fine level) — and it is the very quantity the router is trained to make look small. A low `δ̂`
  therefore does not by itself upper-bound the true dropped mass `δ`. To be an actionable gate it must
  (a) be **calibrated against periodic dense (or large-`W`) spot-checks** so it is a true upper bound,
  (b) carry a **separate super-block-level estimate** for the coarse-prefilter blind spot, and (c)
  define a **pass/fail threshold and remediation** (roll back / shrink `b` / widen `W`). New active
  lever: **train the router/indexer to imitate dense top-k**
  (DSA's warm-up distillation) — turning `δ` from something we merely *certify* into something we
  *minimize*. **Native end-to-end training is the key** (post-hoc sparsification degrades; trained-in
  does not). Block size `b` is a quality↔compute knob (smaller → finer selection, more routing cost).

## 8. Revised phased build plan (build-vs-buy + gates)

| Phase | Deliverable | Build / Buy | Gate |
|---|---|---|---|
| **0. Core** | `AttentionAccumulator` (L0) — promote `_merge_block_attention` to a standalone module (consolidating the existing `StableRingAccumulator` / `StableAttentionAccumulator`) | **BUILD** | property tests: associativity, equals-one-joint-softmax, masked-partial safe, fp16/fp32; **fwd+bwd `gradcheck`** (none exists today — this is the gate, not a current property) |
| **1. Static backend** | Wire L1 static patterns to **FlexAttention** (+ FA3 when available); local/dilated-stride/global skeletons as `BlockMask` builders | **BUY** | parity vs dense masked-softmax ref; FlexAttention fwd+bwd `gradcheck`; MFU floor |
| **2. Adaptive policy** | **Port FlashMoBA** (mit-han-lab, BSD-3) for **flat** block-centroid top-k (canonical L2); expose NSA (fla-org) + DSA token-level as alternative policies behind one interface designed to also admit a **multi-level (hierarchical) router** (§5.2) for the scale-out path | **PORT** (flat) / **BUILD** (hierarchical) | parity vs the reference kernel; selection skips real fwd+bwd compute; quality parity on a small LM |
| **3. Hierarchical sparse ring (the differentiator)** | L3 **topology-aware** ring via L0: dense intra-node + sparse inter-node levels, prune shards no local query selects; **BUILD** the cost-based, *per-step (data-dependent)* cross-rank block→rank assignment (a new collective the generator lacks — it only self-regulates per-rank sparsity); reuse the generator's *topology mapping + node-size detection* (reconfigure default 8 → NVL72 **72**); global-position causal masking incl. boundary-centroid future-leak | **BUILD** (topology mapping reused; block→rank assignment + collective are new) | multi-GPU (`torchrun`) parity vs single-GPU; measured **inter-node** comm reduction; **ring-step load imbalance < X% on a real *learned* (not static) pattern**; cross-rank causal-centroid correctness |
| **4. Training recipe** | **Muon** (2D matrices) + AdamW (embeddings/norms/head); add **MuonClip** QK-clip for large-scale stability; **AdEMAMix** + the novel (no-precedent) **Muon×AdEMAMix** as experimental options; **Dion** tracked for sharded-weight ring/FSDP settings | **ADOPT** Muon/MuonClip; **EXPERIMENT** AdEMAMix | loss-curve parity vs AdamW; throughput; QK-logit stability |
| **5. Consolidation** | Route existing variants through the engine; deprecate the broken bespoke classes | **BUILD** | benchmark-suite parity (tokens/s, peak mem/GPU, loss parity) |

**Foundation already on `main`:** L0 seed (`_merge_block_attention`, #29), ring K/V-aliasing fix
(#27), the audit + cost calculator. The old v1 "Phase 3: hand-write a Triton Flash kernel" is
**deleted** — replaced by Phases 1–2 above (buy + port).

**Kernel correctness guardrail (Phases 1–2, L2 selection/routing).** Any low-precision score reduction that
feeds a `top-k`/`argmax` — the **block-centroid inner-product** scores in the **ported FlashMoBA / NSA
Triton/CUDA kernels** (the at-risk sites; `q · mean(keys_in_block)` reduced over `d_model`) — must
**accumulate in high precision (fp32/fp64) or integer space and must not enable fast-math float
reassociation.** (Note: the in-repo `sparse/block_sparse_adaptive.py` is *not* such a site — its
`ImportanceScorer` `torch.topk`s over a **learned MLP** on `cat(q, k)`, with no dot-product reduction, so
the reassociation hazard does not apply to it; it applies to the inner-product centroid kernels we port.) Float
addition is non-associative, so reassociation perturbs a dot product by ~ULP·√D; when the reduction is
consumed as a *rank* rather than a value, near-tied candidates flip. A sibling project (`gide`) saw top-k
recall collapse **0.996 → 0.030** purely from `-ffast-math` f32 reassociation in an inner-product kernel,
fixed by reverting to fp64 / integer-space comparison. Add a ranking-stability regression test (top-k under
fast-math == top-k under strict-fp). This is *orthogonal to* the quantizer's own order-preservation
(§11.3): even a correct quantizer rank-flips if the kernel reassociates.

## 9. Research resolutions & confidence ledger

Five deep-research passes (2026-05-30, 3-vote adversarial) resolved every *fact-checkable* prior-art
item; the one genuinely-novel combination (**Muon×AdEMAMix**) has no published precedent and stays an
explicit open research question (see Optimizers below):

**FlexAttention adaptive crux — RESOLVED.** It *does* support data-dependent learned selection:
`mask_mod(b,h,q,kv)` + a `BlockMask` built from learned top-k indices skips **real fwd+bwd compute**
(~2× on 50% causal; kernel ~90% FA2 fwd / ~85% bwd). Catch: the per-forward `create_block_mask`
rebuild is the hardest case (~hundreds of µs/call, ~10× reducible via `_compile`). → **ADOPT** for
static patterns; usable but **PORT-with-care** for learned selection (dedicated kernels are faster).

**MoBA + FlashMoBA — the cleanest match to our L2.** MoBA (Moonshot, MIT) is parameter-less top-k
*block* routing by centroid score — *exactly* our "block-centroid top-k" idea — and **FlashMoBA**
(mit-han-lab + NVIDIA, BSD-3, arXiv 2511.11571) is an open CUDA kernel with real fwd+bwd savings (up
to 14.7× vs FA2; 7.4× / 6.1× less memory at 64K vs reference MoBA). → **PORT FlashMoBA** as the
canonical learned-selection kernel; **LEARN-FROM** MoBA's gating. Block-level, so it composes with our
sparse-ring L3 — unlike DSA's token-level path.

**DeepSeek line: NSA → DSA → V4 (all verified, primary sources).** DSA (introduced in **V3.2-Exp,
Sept 2025**; documented in the **DeepSeek-V3.2 report, arXiv 2512.02556, Dec 2025**) is the production
NSA successor: a lightweight "lightning indexer" scores all prior tokens → top-k (k=2048) → core
attention over selected only, O(L²)→O(Lk), built **on MLA** (token-level; the indexer itself stays
O(L²) but is cheap). **DeepSeek V4** verifiably exists (preview ~Apr 2026; V4-Pro 1.6T/49B, V4-Flash
284B/13B, 1M default context); its attention is a **hybrid interleaved across layers** of **CSA**
(Compressed Sparse Attention: KV compression + DSA-style top-k selection) and **HCA** (Heavily
Compressed Attention: aggressive ~128× compression with **dense** attention, sparse selection dropped)
— *not* simply "compression layered on DSA". → **TRACK; offer DSA as a token-level policy option.** The
field is converging on learned selection; we differ by staying **block-level** (hardware- and
ring-friendly), DSA as advanced option. *(V4 is a ~1-mo preview — treat specifics as time-sensitive.)*

**MLA — deprioritize for *training* cost.** MLA's big win (~57× KV-cache; 14%/4% of MHA) is
**inference**-side; the training benefit is only modest activation memory (offset by extra matmuls), and
no source shows it composing with block/sparse. It underpins the DeepSeek serving stack (DSA/V4 sit on
it) but is misaligned with our training-cost goal. → **LEARN-FROM / IGNORE** unless we also target inference.

**Optimizers.** **MuonClip** (Muon + QK-clip) — the only Muon-stability fix validated at 1T scale
(Kimi K2, 15.5T tokens, zero loss spikes); QK-clip is a general attention-logit stabilizer (Megatron-Core
ships it). → **ADOPT** at scale. **Distributed Muon** (ZeRO-1, half AdamW's extra optimizer memory) needs
the *full* gradient matrix → clashes with sharded/ring weights; **Dion** (MSR, power-iteration, FSDP/TP-
friendly, ~3B-validated) is the sharded-weight successor → **TRACK**. **AdEMAMix** (Apple, MIT, arXiv
2409.03137): fast + very-slow (β3=0.9999) EMA mix; ~half the tokens of AdamW at 1.3B; **not** a drop-in
(4 hyperparams + β3/α warmup; weak under distribution shift). → **EXPERIMENT / adopt-with-care**.
**Muon×AdEMAMix** (your custom impl): **no published precedent found** → genuinely novel/experimental;
open question whether AdEMAMix's early-divergence interacts with Muon's stability needs (the reason for
MuonClip). Support as an experimental optimizer; no precedent to lean on.

**Caveats:** Muon's "~2×/52%" is first-party and shrinks at scale (~1.4×→~1.1× at 1.2B, independent);
NSA/DSA/V4 efficiency magnitudes are vendor-reported; **V4 is a recent (~1 mo) preview**; FlexAttention
perf figures are PyTorch's own self-report; AdEMAMix's token-efficiency is unreplicated above 1.3B.

## 10. Reference materials

Cached in `docs/references/` (papers in `docs/references/papers/`, repo pointers + index in
`docs/references/README.md`). Papers: NSA (2502.11089), FlashAttention-3 (2407.08608),
Muon/Moonlight (2502.16982), FlexAttention (2412.05496), **DeepSeek-V3.2/DSA (2512.02556)**,
**MoBA (2502.13189)**, **FlashMoBA (2511.11571)**, **AdEMAMix (2409.03137)**, **Dion (2504.05295)**,
**Kimi-K2/MuonClip (2507.20534)**, **DeepSeek-V2/MLA (2405.04434)**. Repos:
fla-org/native-sparse-attention, mit-han-lab/flash-moba, MoonshotAI/{MoBA,Moonlight},
lucidrains/{native-sparse-attention,ring-attention,local-attention}-pytorch, KellerJordan/Muon,
Dao-AILab/flash-attention, apple/ml-ademamix.

**Internal prior art (partially reused by L3, §5.1):** `src/dilated_attention_pytorch/sparse/sparse_pattern_generator.py`
(`HierarchicalSparsePatternGenerator` — local/global/inter-node pattern levels, **per-rank sparsity
self-regulation** (*not* cross-rank balancing), and node-size detection (default 8 → set to 72)) and
`docs/guides/hierarchical-patterns-guide.md`. Only the **topology mapping** is lifted into the
hierarchical sparse-ring; the **cost-based cross-rank block→rank assignment is a Phase-3 BUILD item**
(it needs a collective the generator lacks). The consuming class
(`BlockSparseRingDistributedDilatedAttention`) needs the audited correctness fixes before reuse.

## 11. Weight-side sparsity & expert design (research track — orthogonal to attention)

Everything above (§2–§9) addresses the **attention map** (which token *pairs* interact) — that
governs compute, activation memory, and communication. It does **not** touch the **model weights**,
which at extreme scale are the *binding* memory constraint (a 500T model = 7.1 PiB of state →
~37k B300s just to hold it). Attention sparsity ≠ weight sparsity; these are independent levers, and
the weight side is currently **unexploited** in our design. This section is a research track for it.

**The real weight-side sparsity is *usage*, not zeros.** A 500T model is necessarily **MoE**: for any
token only a small active set of experts fires (we model this for *compute* via `active_params`, but
still *store* all 500T). So "vast sections are unused per token" is true — they're **cold, not zero** —
and the opportunity is to stop keeping cold weights in fast, full-precision memory.

**Lever A — hierarchical fidelity (quantized-resident / full-on-disk) + offload.** Apply the hierarchy
principle to *precision*: keep a **low-precision (e.g. INT4) copy of hot experts in HBM** and the
**full-precision copy on NVMe/disk**, fetched only when a topic needs deep, high-fidelity processing;
park rarely-used experts entirely on disk. Modeled in the calculator (`expert_frac`,
`weight_quant_bits`, `expert_offload_ratio`): at 500T with 95%-expert / INT4-resident / 90%-offloaded,
the GPU state floor drops **7.1 PiB → 537 TiB resident → ~37k → ~2.7k B300s (~13.6×)**, with ~778 TiB on
disk. The trade is wall-clock for GPU count — the same fixed work now rides ~13.6× fewer GPUs, so the
ideal cluster step stretches **~2.7 min → ~36 min/seq** — but the **offload disk I/O stays hidden
behind compute**: ~778 TiB fetched/step ÷ 2,726 GPUs @ 6 GB/s ≈ **52 s**, « the 36-min compute step
(modeled by `offload_io_seconds`; tune with `--nvme-bw` / `--offload-fetch-frac` / `--offload-write-back`).
This directly attacks the binding constraint, complementary to MLA (which compresses KV, not weights).
*Quant-safety caveat (over-credits trained experts):* the calculator's `resident_state_bytes` applies
the INT4 factor to the experts' **whole** state (weights + grads + **optimizer + master**), and
`state_bytes_per_param=16` is a flat blob (≈ 2 wt + 2 grad + 4 master + 4 m + 4 v; Muon ~12). But
optimizer moments and master weights of *trained* experts are **not** validly INT4 (§11.3: lossy master
accumulates error) — only frozen, inference-side weight copies are. So the 13.6× holds for **cold /
frozen** experts; for any expert actively trained at INT4-resident it is optimistic, and the per-param
breakdown should quantize only the components that are safe.

**Lever B — heterogeneous (variable-size) experts.** Nothing fixes a uniform expert size (the "1T" in
earlier estimates was *total active* params — backbone + shared + `k` routed experts — not one
expert). Experts can be **sized by specialization**: broad/common domains get higher-capacity experts,
niche ones less — or, tying into the novelty theme, **capacity-follows-surprisal**: allocate *more*
parameters to the rare/high-information concentrations of knowledge (where novelty = utility) and
fewer to the predictable bulk. Cost: load-balancing and ragged-shape kernels (standard MoE uses
uniform size for batched matmul simplicity); surmountable via grouping/capacity factors. Under-explored.

**Lever C — overlapping / compositional experts.** Experts need not be disjoint. Forms that *reduce*
memory while covering multi-facet knowledge: **shared-base + specialized-delta** experts (a common
low-rank base stored once + per-expert LoRA-style deltas — overlap by construction, big memory win);
**shared experts** (always-on common knowledge + routed specialists, à la DeepSeek); and
**hierarchical experts** (a coarse expert for a broad facet + fine experts for sub-facets). Overlap =
parameter sharing = less total state *and* better coverage of a knowledge concentration's facets.

**Lever D — novelty-weighted routing.** Select/weight blocks or experts by **surprisal/information
gain**, not only similarity. Information-theoretically sound (rare co-occurrences carry more bits), and
it could let a *smaller* budget `W` hit a target quality (spend it on high-value rare links). But it is
**contrarian** — every validated method (NSA/DSA/MoBA, H2O heavy-hitters) selects by *similarity/mass*,
keeping the dominant connections — and risks amplifying noise. Treat as a **research bet** (a surprisal
term in the router), not a foundation.

**MoE routing mechanics — the unmodeled comm + balancing (a gap, not a solved part).** The whole
"500B-active-of-50T" premise rests on expert routing, yet the cost model reduces all of MoE to the
single scalar `active_params`. Two first-order costs are absent: (1) **expert-parallel all-to-all**
dispatch/combine — at 50T params across thousands of GPUs this is a major comm term *and* a major
straggler source, none of which appears in `cluster_step`; (2) **capacity factor / token-drop** —
over-capacity wastes FLOPs, under-capacity drops tokens and erodes the very "~1,500B/expert exposure"
the §11.1 case rests on. The "manageable with aux-loss-free balancing + shared experts" claim (§11.1)
has no balancing analysis behind it (no capacity factor, no dead-expert/utilization-variance plan).
There is also a **consistency tension to resolve**: the calculator's `offload_fetch_frac≈1.0` assumes
the routed-expert union *saturates* per step — but if the union is ~all experts, the weight side is not
sparse either, contradicting the sparsity premise (and the Lever-A 13.6×). Either the per-step union is
sparse (`fetch_frac « 1`, and offload I/O is even cheaper) or it saturates (and "sparse MoE" is
overstated) — not both. **Needs:** a routing-algorithm + capacity-factor + all-to-all-comm subsection
with a balancing gate (expert-utilization variance, fraction tokens dropped) before "manageable" is
earned.

**Status & caveats.** Lever A is engineering-ready (offload/quant are mature; the calculator quantifies
it). Levers B–D are genuine, under-explored research directions, *not* validated at scale — they are
deliberately separated from the validated attention architecture above. All four are **orthogonal to
the attention work**: they shrink the *weight* memory wall that sparse attention does not.

### 11.1 Reference variant — 50T total / 500B active (the capability-vs-cost sweet spot)

The §6 extreme case (500T/1T) is a *limit* study; for a buildable target the open levers above are the
total-params, active-params and sparsity knobs. Sweeping them (`--params 50e12 --active-params <a>`,
data-matched `D ≈ 300 × a` tokens) shows **active params, not total, are the capability axis**: total
sets the knowledge *ceiling* (identical for any active count), active×data sets reasoning depth and
*how fully* the shell is filled. The sweet spot is **50T total / 500B active (100× sparsity)**:

| Property | 250B (200×) | **500B (100×)** | 1T (50×) |
|---|---|---|---|
| Train compute (data-matched) | 1.1e26 (~6× GPT-4) | **4.5e26 (~22× GPT-4, ~12× Llama-3-405B)** | 1.8e27 (~90× GPT-4) |
| Per-token compute | ~GPT-4 class | **> Llama-3-405B / GPT-4** | unprecedented |
| Token budget D | 75T (0.25 ep) | **150T (0.5 epochs of ~300T stock)** | 300T (1 ep — at the data wall) |
| Per-expert exposure | ~375B | **~1,500B (richly trained)** | saturated/over-trained |
| Sparsity vs validated (Kimi 48×) | ~4× beyond | **~2× beyond (safest aggressive)** | ~1× (at frontier) |
| Full run @ ~3.7k floor, **1B ctx, clustered ideal** | ~0.9 yr | **~3.1 yr** | ~11.8 yr |

**Why 500B is the build target.** It is the first config where *both* halves are strong at once:
reasoning **above the current frontier** (per-token compute exceeds any deployed model; 22× GPT-4 total
compute), *and* the 50T knowledge shell is **actually realized** (150T tokens, ~1,500B/expert clears the
training floor — broad recall/multilingual/long-tail written in, not merely provisioned). Sparsity 100×
is the lowest-risk of the aggressive options (~2× past Kimi-K2's validated 48×, manageable with
DeepSeek-style aux-loss-free balancing + shared experts, Lever C), and 0.5 epochs leaves data headroom
(1T-active would consume the *entire* human-text stock). Contrast the naive "middle" **50T/50B (1000×)**:
~15B tokens/expert leaves experts *below* their training floor, and total train compute is **4.5e24 —
below GPT-4** — a sub-frontier brain in an unfillable shell at full 50T systems cost. The capability axis
is active params; do not chase a low active count to save cost.

**Cost scales with GPUs (compute-bound, *if* selection clusters).** At the true **1B target context** the
**~3.7k-GPU memory-floor** ideal run is **~3.1 yr** (clustered selection); under *independent* selection it
is comm-bound at **~8.3 yr** (the §6 #4 bracket), and **×1.5–3** further for real-world (§6.2). Otherwise
compute-bound, so adding **data-parallel replicas** of the 3.7k-GPU group scales near-linearly — ~15k GPUs
(≈4 replicas) → **~0.8 yr**, ~37k (≈10 replicas) → **~3.7 months** — making 50T/500B a months-to-a-year run
on a frontier-scale cluster. (This is the data-parallel axis, **not** 37k GPUs on one sequence; the global
batch must be reconciled per §6.2.) Reproduce: `--params 50e12 --active-params 500e9 --train-tokens 150e12
--contexts 1073741824` (the default 1M context gives **~2.8 yr**; the old "4.3 yr" reproduced from no single
setting — it landed mid-bracket).

**Caveats (carry from the capability analysis).** Every placement past ~48× sparsity / ~1T total is
**extrapolation** beyond any trained model; capability is **empirical**, not a-priori. All configs clear
emergence *onset* (CoT/ICL/instruction-following) — the difference is *ceiling*, a smooth gradient
(Schaeffer 2023), not a cliff. The √(total·active) ≈ 5T "effective params" is a heuristic that likely
**saturates lower** (Clark 2022 caps effective size ~80–900B dense-equivalent), and per-expert figures
assume an (unfixed) expert granularity. Treat 50T/500B as the best-justified *bet*, validated bottom-up
from 250B before committing the full budget.

**Frontier-comparison caveat (read the "22× GPT-4" with care).** The GPT-4 (~2e25 FLOP, ~280B active)
and Llama-3-405B anchors are **leaked/estimated, 2023-era** figures, not disclosed numbers. Two
corrections follow. (1) **Stale baseline:** frontier training compute has scaled **~4.7×/yr** (Epoch
AI); the frontier crossed **~1e26 FLOP** in 2025 (Grok-3 first), so 4.5e26 is only **low-single-digit×
the *current* frontier, not 22×**. (2) **Wrong axis:** the 2025–26 frontier shifted from raw
pre-training scale to **post-training efficiency** — GPT-5 reportedly matched/beat GPT-4.5 at **~10×
less** pre-training compute via RLHF/RLVR/reasoning-distillation + test-time compute. Recent frontier
models (Claude Opus 4.8, GPT-5.5) **disclose no parameter or compute figures at all**, so a direct
compute comparison is not possible. Net: train-FLOP is a **weak capability proxy** here; a 4.5e26-FLOP
pre-training bet must be paired with a modern post-training stack (see below) to be frontier-relevant,
and "22× GPT-4" should be read as scale context, not a capability claim.

**Data is the binding constraint as *quality*, not just *quantity* (an unaddressed dimension).** §6.1
declares data binding but treats it purely as stock size (does ~150–300T tokens exist?). At 0.5–4 epochs
over essentially the entire scraped human-text corpus, **corpus integrity becomes load-bearing** and is
nowhere in the plan: **dedup + benchmark-contamination control** (leakage directly poisons the §11.2
AIME/MATH/GPQA anchors the capability case rests on), **poisoning/adversarial-content defense** (a
multi-month run on a near-exhaustive web crawl is a prime target), **licensing / copyright / PII
governance** at this scale, and **quality filtering** (the "~1,500B/expert clears the floor" claim
presupposes the tokens carry usable signal, not duplicated/low-quality text). Realistic post-dedup yield
also lowers the effective epoch count. A data-engineering/governance subsection (and a contamination gate)
is required before "data-matched 150T" is a usable target, not just a count.

### 11.2 Post-training extrapolation (reaching the base's ceiling)

The 2025–26 frontier lesson (GPT-5 ≈ GPT-4.5 at ~10× less pre-training, via post-training) has a precise
implication for this base. **RLVR / reasoning-RL mostly *elicit* latent capability — raising pass@1
toward the base's existing pass@k ceiling — rather than *adding* new capability** (Yue et al. 2025;
ProRL's prolonged-RL expansion is the contested margin). So post-training is a ceiling-*reacher*, not a
ceiling-*raiser*: the pre-training run sets the ceiling, and a 500B-active / 50T-knowledge base sets an
unusually high one. **This resolves the §11.1 frontier worry — the pivot to post-training does not make
the pre-training bet obsolete; a strong base is the prerequisite that makes post-training pay.**

Win / saturate (corrected anchors; all extrapolation — no model post-trained near this scale):

| Lever | Anchor (real models) | Effect on 50T/500B |
|---|---|---|
| RLVR / reasoning-RL | R1 vs *its base* V3: AIME 39→80, MATH-500 90→97, GPQA-D 59→72; o1 AIME ~12→**74% pass@1** | Largest lift; higher floor → near-saturates verifiable math/code |
| Test-time compute | ~log-linear vs inference compute (o1/o3; Snell 2024) | **1B context is the real multiplier** (whole search trees / agent transcripts); per-query *expensive* |
| Agentic / long-horizon RL | SWE-bench agents ~80% today | **The standout** — 50T shell cuts missing-fact hallucination, 1B context kills eviction; the two base properties *compound* |
| Preference-RLHF / open-ended | diminishing (4.4%→1.9% gain, 9B→200B policy) | Modest; no verifier → formatting-level lift only |

**What post-training buys (not a compute multiplier).** The GPT-5 anecdote is a **substitution rate** —
*same quality at ~10× less pre-training* — i.e. post-training lets you *reach* a fixed quality with a
smaller base. It is **not** a multiplier you can stack *on top of* an already-large base to manufacture a
"~2e27–7e27-equivalent, frontier-leading" figure: a substitution rate and an additive bonus are different
regimes, and post-training that *elicits a fixed ceiling* cannot multiply a base that already sits at a
high ceiling. So treat post-training as a **fixed-budget capability-elicitation lever on specific axes**
(verifiable math/code, agentic/long-horizon), not as extra effective pre-training FLOPs. The differentiated
payoff is **agentic / knowledge-grounded long-horizon work** (the 50T-shell + 1B-context + agentic-RL
stack compounding); open-ended reasoning stays bounded by the base. (This is also consistent with the
table above — the lifts are axis-specific, not a uniform scale-up.)

**Caveats.** All extrapolation: the cleanest anchor (V3→R1) is ~75× smaller in total / ~13× in active
params, and parameter count does not *provably* raise the pass@k ceiling (more knowledge ≠ more
compositional reasoning). RL cannot conjure what the 150T pre-training tokens didn't store — coverage gaps
are permanent (the data wall, again); distillation self-cancels at the frontier (no superior teacher). The
binding ceiling may be **inference economics** (500B-active × long-CoT × best-of-N) or **alignment**
(weak-to-strong: naive RLHF scales poorly to stronger models; oversight harder), not latent capability.
Benchmark step-functions are partly metric artifacts (Schaeffer 2023; AIME = 30 problems).

### 11.3 External-technique evaluation — TurboQuant (and the `gide` cross-project findings)

**TurboQuant** (Zandieh et al., Google Research/DeepMind, ICLR 2026, arXiv 2504.19874) is a *data-oblivious,
online* vector quantizer bounding **both** MSE *and* inner-product distortion (random rotation → per-
coordinate optimal scalar quant + a 1-bit QJL residual; ~2.7× off the information-theoretic limit; ~3.5-bit
KV quality-neutral). Sold as an inference/KV technique; evaluated here as a *training-cost* lever.

**Direct verdicts.**
- **Offloaded-expert fetch I/O (Lever A) — best fit (~4.5×).** Disk copy is bf16; compressing it cuts the
  disk→HBM fetch term (~52 s → ~11 s/step at 500T). Read-only, no backward pass, no calibration — its home
  turf. *Not* for trained/write-back experts (lossy master accumulates error) or resident weights
  (calibrated GPTQ/AWQ win).
- **Routing / selection-index quantization — research, with a top-k rank-flip risk** (mitigation below).
- **KV-ring wire codec — research/insurance.** Inner-product preservation is right for ring QK^T, but
  sparse-ring pruning + MLA already make comm compute-bound; sits in the autograd path (untested); does
  *not* compose multiplicatively with MLA.
- **Gradient/activation-comm codec — research.** Needs error-feedback, which we **already implement**
  (`sparse/distributed_memory_optimization.py`, top-k + EF residual carry); real blockers (biased-compressor
  convergence at 1B-activation scale, per-tensor EF state in EP/TP all-to-all) are untouched.

**`gide` cross-project findings** (`../gide/docs/research/turboquant-cognitive-infrastructure.md` — the
sibling project generalized TurboQuant's `rotation + Lloyd-Max + QJL + PQ/OPQ + rerank` toolkit; only the
inner-product/quantization primitives transfer, not its gradient-free evolutionary applications):
- **Prescreen + exact re-rank** promotes the routing-quant item from "open risk" to "candidate mitigation
  with a validation gate." Raw low-bit inner products do *not* preserve top-k at scale (gide 4-bit HNSW
  recall **0.21** @ DIM=1536/N=2048); an oversampled cheap-sketch prescreen + **full-precision re-rank**
  recovers recall **1.0** (gide `searchForRerank`, oversample=10 — the FAISS/ScaNN pattern). Final routing
  scores are FP, so the margin rank-flip is structurally removed. Caveats: validated for **PQ-ADC + scalar**
  re-rank, *not* QJL (QJL-prescreen recall unmeasured); all gide numbers are *static, isotropic-Gaussian,
  read-only*, whereas our router is **gradient-trained with the sketch in the loop on clustered, drifting
  centroids** — a stale prescreen can persistently exclude a centroid that drifts into the true top-k,
  **starving its gradient**. **Gate:** a training-trajectory study of top-k recall + per-expert selection
  frequency on real (anisotropic, moving) centroids, oversample recalibrated across the run, before adopt.
- **OPQ** (one-time learned Jacobi-eigendecomposition rotation) beats data-free random rotation on
  correlated vectors (+2.3%, understated on low-correlation data) — the right rotation if we quantize the
  routing/KV index. **SRHT** is the rotation *implementation* (≈40× storage, ≈9× speed vs dense Gaussian;
  **sparse-Rademacher is slower than dense — avoid**); gide's numbers are CPU-Zig, so the GPU/bf16 win needs
  re-measuring. **Dual-chirality** variance reduction is a **negative result** in gide (can't move the
  quantization-set recall ceiling) — drop.

**Negative results — do *not* pursue (saved by `gide`'s own experiments):**
- **QJL weight-space expert-diversity signal — DROP.** Tempting for the dead/redundant-expert concern at
  100–1000× sparsity, but gide's pre-registered study found weight-space Hamming **does not correlate with
  functional/behavioral diversity** (ρ ≈ 0). Worse for us: our experts are full FFN sub-networks with exact
  **hidden-neuron permutation symmetry** — functionally identical experts can be maximally far in
  weight-Hamming. Expert redundancy is a *functional* problem; weight-space distance is the wrong notion.
- **Delta-Sigma "error feedback" — already have it** (classic EF under a signal-processing name).

**Net:** one adoptable correctness guardrail (§8), a validated de-risking pattern (prescreen+re-rank + OPQ)
for the already-parked routing-quant research item, and two negatives that save build effort. TurboQuant
stays an *inference-side adopt* (KV-cache when serving) + a *training-side research* lever, not a primary
training-cost reducer. External refs (now cached in `docs/references/papers/`): QJL (arXiv 2406.03482),
SpinQuant (2405.16406), QES (2602.03120).

*Provenance caveat:* unlike every other claim in this doc (each backed by a cached, arXiv-stable PDF),
the `gide` de-risking evidence is an **out-of-repo, unversioned** sibling project
(`../gide/docs/research/turboquant-cognitive-infrastructure.md`) whose numbers cannot be re-verified from
this repo — and its regime differs from ours (**CPU-Zig, static, isotropic-Gaussian, read-only** vs. our
gradient-trained, anisotropic, drifting-centroid router with the sketch in the loop). So the "open risk →
candidate mitigation with a validation gate" promotion is **pending in-repo replication**, not settled;
the training-trajectory gate flagged just above (top-k recall + per-expert selection frequency on real
anisotropic, moving centroids, oversample recalibrated across the run) is the thing that would settle it.

### 11.4 Compute-operation levers — beyond dense matmul

The compute bound (≈ `6·N_active·D`; per-step forward for the **50T/500B build target** ≈ **99.8%
FFN/linear matmul + 0.2% attention + 0.01% selection** — the attention share rises toward ~20% only at the
500T/1T extreme dims, and is never the ~35% an earlier draft asserted; if anything this *strengthens* the
"FFN GEMM is the bound" thesis) is *dense multiply-accumulate*. "Can we beat matmul?" is a real research axis — the bound is not
algorithmically irreducible, but it is **irreducible on B300 tensor cores at frontier quality**: the hardware
delivers 3.5 PFLOP/s *only* for dense FMA, so a FLOP cut on a non-tensor-core operation becomes a wall-clock
*slowdown*. Three classes, surveyed + adversarially verified (refs cached in `docs/references/`):

| Class | Replaces matmul with | Theoretical | B300 wall-clock | Frontier quality | Verdict |
|---|---|---|---|---|---|
| Matmul-free / ternary (BitNet b1.58, MatMul-free LM) | signed **add** (ternary {−1,0,+1}) | ~71× per-op energy | **no** — no ternary datapath; unpacks to INT8 (**integer MMA**), a *distinct* datapath from the FP8 precision lever (finding 3) — no ternary-specific speedup | unproven (native parity ≤2B) | research |
| Structured weights (Monarch / M2) | `O(d log d)` butterfly blocks | 2–8× FFN FLOPs | ~break-even — GEMM-friendly but ~25% naive util; quality-matched ≈ dense | ≤1.3B only | research |
| Approximate / sub-cubic (MADDNESS, Strassen, AlphaTensor) | LUT gathers / fewer MACs | 10–100× (CPU) | **no** — strands tensor cores; unstable; tiny sizes | none at scale | track |

Three findings:
1. **FLOP-cut ≠ wall-clock.** At training the FFN GEMM is compute-bound (above the B300 roofline ridge) — the
   only regime where a cheaper op *could* help, and only if it is tensor-core-native. Ternary adds, LUT
   gathers, and butterfly permutes are not, so they go memory-bound and lose.
2. **Low-bit favors *under*-trained models** (Ouyang et al., arXiv 2411.17691): quantization degradation rises
   with tokens/param. Our heavy over-training (100–500T tokens, §6.1/§11.1) is precisely where ternary and
   aggressive FP4 hurt *most* — a direct tension with low-bit compute. **Prefer FP8 over NVFP4/ternary** at our
   token budget.
3. **The only tensor-core-native "cheaper op" is precision** (FP8 ~2×, NVFP4 up to ~4× if quality holds) — a
   *precision* trade, not an operation change. Otherwise "better than matmul" = a different op approximating the
   same linear map at a quality cost, i.e. the same quality↔compute frontier as sparsity/precision, not a free lunch.

**The one path that flips this: hardware–software co-design** — a ternary / LUT / in-memory-analog accelerator
where add- or table-based compute is the *native* op. Out of scope for a B300 plan (**track**), but at
50T/500B / billion-token scale a custom training ASIC is exactly the bet that would rewrite this table.

---

*Cost figures: `analysis/attention_cost_analysis.py` (re-derived 2026-06-04; numbers in §6/§6.1/§11.1
reproduce from the cited commands — note the §6 extreme case needs its model-dimension flags). Prior-art
verdicts: five deep-research passes (2026-05-30; every fact-checkable item resolved, Muon×AdEMAMix left
open). Pressure-tested 2026-06-04 (41 findings; corrections applied). v1:
`docs/archive/unified-attention-architecture-v1.md`.*
