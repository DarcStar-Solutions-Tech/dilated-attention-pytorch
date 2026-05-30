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
   [BUILD — ours]          softmax-combination lives. Pure, gradcheck-tested.
                           Seed exists: _merge_block_attention (PR #29).

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
                             multi-level (super-block -> block, log-depth) at extreme scale to keep
                             SELECTION itself sub-quadratic (§5.2).

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
- **Load-balanced block→rank assignment:** sparse/causal patterns are irregular (query blocks attend
  to differing numbers of key blocks), which would create ring-step stragglers; assign blocks to
  ranks by measured compute/comm cost to keep steps balanced.

This **reuses the project's existing `HierarchicalSparsePatternGenerator`**
(`sparse/sparse_pattern_generator.py`; guide: `docs/guides/hierarchical-patterns-guide.md`), which
already implements the local / global / inter-node levels + load balancing and auto-detects node
size — we lift its topology mapping into L3 and pair it with the corrected online-softmax reduction
(L0). It is the direct answer to the two tensions a flat sparse-ring leaves open — **interconnect
cost** and **load imbalance** — and is the layer that most distinguishes this library from
single-device NSA/DSA/FlexAttention. *Caveat: the class that currently consumes the generator,
`BlockSparseRingDistributedDilatedAttention`, carries audited correctness bugs — the topology
mapping is reusable, but the consuming path must be rebuilt on the corrected L0/L1.*

Hierarchical *ring* (communication, §5.1) and hierarchical *routing* (selection, §5.2 below) are
the two extreme-scale extensions of this design.

### 5.2 Hierarchical selection routing — L2 at scale

Content-adaptive selection must *score* candidates to pick the top-k, and that scoring is not free
(§6). **Flat** routing — score every (query, key-block) pair — is `O(n²/b)`: fine at ≤1M tokens
(~6% overhead), but at extreme context it becomes the dominant cost and a *new* quadratic wall (at
1B tokens our cost model shows selection rivalling the core attention even block-level). So L2's
routing is **multi-level**, mirroring the hierarchical ring on the selection side:

- **Coarse prefilter:** group blocks into *super-blocks*, score query against super-block centroids,
  keep the top few super-blocks — `O(n²/b²)` or less.
- **Fine selection:** run block-level top-k *only within* the surviving super-blocks.
- Recurse for more levels at higher context → **log-depth** routing, keeping total selection
  sub-quadratic instead of `O(n²/b)`.

This is a natural extension of mechanisms already in the field: **NSA's coarse token/block
*compression* branch is itself a one-level coarsening** that a hierarchical router generalizes, and
the project's **`HierarchicalSparsePatternGenerator`** already expresses multi-level (local / global
/ inter-node) structure we can reuse for the routing hierarchy as well as the ring. The flat
MoBA/FlashMoBA router is the **default** (correct and fast through ~1M tokens); the multi-level
router is the **scale-out path**, exposed behind the same L2 policy interface so it is opt-in by
context length. *Quality note:* coarsening risks missing a relevant block whose super-block scored
low — mitigated by training the router end-to-end (and the dense-imitation distillation in §7), and
bounded by the same dropped-mass certificate.

Symmetry to remember: **hierarchical ring keeps *communication* sub-quadratic; hierarchical routing
keeps *selection* sub-quadratic.** Both are required to actually reach the extreme-context regime
the cost model describes; the flat versions are correct and sufficient up to ~1M tokens.

## 6. Cost model (refined post-research)

Reproduce with `python analysis/attention_cost_analysis.py` — now models **selection granularity**
(flat block / token / hierarchical), **split fwd/bwd MFU**, **MLA KV compression**, and **ring
communication**. 7B-class model (`d_model=4096`, 32 layers, bf16), core budget `W=4096`,
**block-level** selection (`b=64`), one H100:

```
   context n |  dense attn | sparse core |  selection | eff attn x | model x | KV(all L) | ring p
     131,072 |  9.01 PFLOP |  281 TFLOP  | 2.20 TFLOP |       32x  |   5.1x  |   64.0 GB |   1
   1,048,576 | 576.46 PFLOP|  2.25 PFLOP | 140.7 TFLOP|      241x  |  34.6x  |  512.0 GB |   8
```
Training step (fwd + 2·bwd) at 1M: dense **3586 s** vs sparse **129 s** → **~28×/step**.

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
   512 GB → 64 GB at 1M, dropping the ring degree **p = 8 → 1** (fits one GPU), compute unchanged.
   Inference-oriented (deprioritized for training *compute*), but it directly attacks the *memory*
   wall that otherwise forces ring sharding.
4. **Communication is the real long-context bottleneck — the hierarchical ring attacks it (§5.1).**
   Ring attention rotates ~the whole KV past each device per forward. A **flat** ring serializes
   that over the slow inter-node link; a **hierarchical** ring keeps the dense rotation on fast
   intra-node (NVLink) links and sends only the sparse inter-node fraction over slow links. At
   7B/1M across nodes (`--node-size 2`): 448 GB/GPU/forward → **flat 4.8 s vs hierarchical 0.7 s
   (~6× less)**. Still compute-bound at 1M, but comm grows with context until it dominates — the
   reason L3 is hierarchical, not flat.

- **Crossover ≈ 27K tokens** unchanged: beyond it, dense attention exceeds the whole linear term.
- **Compute (sparse + hierarchical selection), memory (ring × KV-compression), and communication
  (hierarchical ring) are solved by *different* levers** — you need all three at scale. Externally
  corroborated: NSA 9×/6× @64k; DSA `O(L²)→O(Lk)` at 1T scale.

## 7. What is determinable a priori (refined post-research)

- **Compute — exact, minus a now-quantified selection term.** Core attention speedup `= n/W`; the
  realized ceiling subtracts the selection cost — `O(n²)` token, `O(n²/b)` flat-block, or
  `O(n²/(b·S))+O(n)` **hierarchical** (sub-quadratic, §5.2) — scaled by fwd/bwd MFU. Validated by
  NSA's measured 9×/6× @64k and DSA's `O(L²)→O(Lk)`.
- **Memory — exact, with a second lever.** non-Flash scores `O(n²)` per layer (Flash removes); KV
  `O(n)` → `O(n/p)` via ring → `O(n/(p·r))` with an MLA latent compression ratio `r`. Selection
  does **not** shrink KV; only representation compression does.
- **Communication — modeled, tiered (§5.1).** Ring rotates ~`O(n)` KV/device per forward; a **flat**
  ring serializes it on the slow inter-node link, a **hierarchical** ring keeps it mostly on fast
  intra-node links (only the sparse inter-node fraction crosses slowly, `O(n·frac/BW_inter)`). Comm
  grows with context and eventually dominates compute — the binding constraint at extreme scale.
- **Quality — still not a tight a-priori number, but the priors hardened.** Provable: a **connected**
  pattern (our skeleton gives `O(log n)` reach) retains universal approximation — no forced ceiling.
  Per-token error bounded by dropped softmax mass: `‖o−õ‖ ≤ 2·δ·max‖v‖`, useful only *if* mass
  concentrates — and **that assumption is now empirically validated**: NSA/DSA/MoBA match-or-beat
  dense at frontier scale (NSA +0.032 LongBench; DSA ≈-parity at 1T). End-task loss stays **empirical**
  (the model trains *with* the pattern), gated by loss parity + a **runtime dropped-mass certificate**
  (`δ̂` from routing scores). New active lever: **train the router/indexer to imitate dense top-k**
  (DSA's warm-up distillation) — turning `δ` from something we merely *certify* into something we
  *minimize*. **Native end-to-end training is the key** (post-hoc sparsification degrades; trained-in
  does not). Block size `b` is a quality↔compute knob (smaller → finer selection, more routing cost).

## 8. Revised phased build plan (build-vs-buy + gates)

| Phase | Deliverable | Build / Buy | Gate |
|---|---|---|---|
| **0. Core** | `AttentionAccumulator` (L0) — promote `_merge_block_attention` to a standalone module | **BUILD** | property tests: associativity, equals-one-joint-softmax, masked-partial safe, fp16/fp32 |
| **1. Static backend** | Wire L1 static patterns to **FlexAttention** (+ FA3 when available); local/dilated-stride/global skeletons as `BlockMask` builders | **BUY** | parity vs dense masked-softmax ref; FlexAttention fwd+bwd `gradcheck`; MFU floor |
| **2. Adaptive policy** | **Port FlashMoBA** (mit-han-lab, BSD-3) for **flat** block-centroid top-k (canonical L2); expose NSA (fla-org) + DSA token-level as alternative policies behind one interface designed to also admit a **multi-level (hierarchical) router** (§5.2) for the scale-out path | **PORT** (flat) / **BUILD** (hierarchical) | parity vs the reference kernel; selection skips real fwd+bwd compute; quality parity on a small LM |
| **3. Hierarchical sparse ring (the differentiator)** | L3 **topology-aware** ring via L0: dense intra-node + sparse inter-node levels, prune shards no local query selects, load-balanced block→rank assignment (reuse `HierarchicalSparsePatternGenerator`); global-position causal masking | **BUILD** (reuse prior pattern generator) | multi-GPU (`torchrun`) parity vs single-GPU; measured **inter-node** comm reduction + balanced ring steps |
| **4. Training recipe** | **Muon** (2D matrices) + AdamW (embeddings/norms/head); add **MuonClip** QK-clip for large-scale stability; **AdEMAMix** + the novel (no-precedent) **Muon×AdEMAMix** as experimental options; **Dion** tracked for sharded-weight ring/FSDP settings | **ADOPT** Muon/MuonClip; **EXPERIMENT** AdEMAMix | loss-curve parity vs AdamW; throughput; QK-logit stability |
| **5. Consolidation** | Route existing variants through the engine; deprecate the broken bespoke classes | **BUILD** | benchmark-suite parity (tokens/s, peak mem/GPU, loss parity) |

**Foundation already on `main`:** L0 seed (`_merge_block_attention`, #29), ring K/V-aliasing fix
(#27), the audit + cost calculator. The old v1 "Phase 3: hand-write a Triton Flash kernel" is
**deleted** — replaced by Phases 1–2 above (buy + port).

## 9. Research resolutions & confidence ledger

Five deep-research passes (2026-05-30, 3-vote adversarial) resolved every v1/v2 open item:

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

**DeepSeek line: NSA → DSA → V4 (all verified, primary sources).** DSA (V3.2-Exp, Sept 2025; report
arXiv 2512.02556) is the production NSA successor: a lightweight "lightning indexer" scores all prior
tokens → top-k (k=2048) → core attention over selected only, O(L²)→O(Lk), built **on MLA** (token-level;
the indexer itself stays O(L²) but is cheap). **DeepSeek V4** verifiably exists (preview ~Apr 2026;
V4-Pro 1.6T/49B, V4-Flash 284B/13B, 1M default context) and layers **token-wise KV compression on top
of DSA** ("CSA+HCA"). → **TRACK; offer DSA as a token-level policy option.** The field is converging on
learned selection; we differ by staying **block-level** (hardware- and ring-friendly), DSA as advanced option.

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

**Internal prior art (reused by L3, §5.1):** `src/dilated_attention_pytorch/sparse/sparse_pattern_generator.py`
(`HierarchicalSparsePatternGenerator` — local/global/inter-node levels + load balancing + node-size
detection) and `docs/guides/hierarchical-patterns-guide.md`. The topology mapping is lifted into the
hierarchical sparse-ring; the consuming class (`BlockSparseRingDistributedDilatedAttention`) needs the
audited correctness fixes before reuse.

---

*Cost figures: `analysis/attention_cost_analysis.py`. Prior-art verdicts: five deep-research passes
(2026-05-30; all open loose ends resolved). v1: `docs/archive/unified-attention-architecture-v1.md`.*
