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

> **The one decision-relevant nuance** (◐ pending direct fact-check, §9): the NSA reference
> impls (fla-org, lucidrains) ship **dedicated Triton top-k *selection* kernels**. That strongly
> implies FlexAttention alone is **not** efficient for *data-dependent learned* selection — it
> covers **static** masks well, but the **adaptive** path needs NSA-style custom kernels. So:
> **FlexAttention for static patterns; ported NSA kernels for learned selection.**

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
   [BUY: FlexAttention]    LEARNED selection -> PORT NSA Triton kernels (fla-org).
   [PORT: NSA kernels]     Do NOT hand-write a general Flash kernel.

L2  SparsityPolicy         Canonical policy = NSA's 3 branches:
   [PORT: NSA + BUILD]       (a) coarse token/block COMPRESSION,
                             (b) fine-grained learned top-k block SELECTION,
                             (c) SLIDING WINDOW (local).
                           Static skeletons (local/dilated-stride/global) -> FlexAttention
                           BlockMask. Our additions: a clean policy interface + multi-scale
                           dilated-as-block-stride skeleton unioned with NSA selection.

L3  ExecutionStrategy      single-GPU (loop) | RING / sequence-parallel: rotate KV shards,
   [BUILD — ours, the      reduce partials via L0; use the block index to SKIP communicating
    distinct value]        KV shards no local query selects (sparse ring). Global-position
                           causal masking. This is the unsolved, high-value layer.

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
branch, and wrap it in our **L3 sparse-ring** strategy for multi-node scale. This is the
combination no single existing project ships.

## 6. Cost model (unchanged; corroborated by NSA)

Reproduce with `python analysis/attention_cost_analysis.py`. For a 7B-class model (`d_model=4096`,
32 layers, bf16) with a context-independent per-query budget `W=4096` on one H100:

```
   context n |  dense attn | sparse attn | attn x | model x | KV (all L) | scores/L | ring p
       8,192 | 35.18 TFLOP | 17.59 TFLOP |     2x |    1.1x |     4.0 GB |    4.0 GB |     1
      32,768 |   563 TFLOP | 70.37 TFLOP |     8x |    1.9x |    16.0 GB |   64.0 GB |     1
     131,072 |  9.01 PFLOP |   281 TFLOP |    32x |    5.1x |    64.0 GB |    1.0 TB |     1
   1,048,576 |   576 PFLOP |  2.25 PFLOP |   256x |   34.9x |   512.0 GB |   64.0 TB |     8
```

- **Crossover ≈ 27K tokens:** beyond it, dense attention exceeds the entire rest of the model.
- **Compute (sparse) and memory (ring) are solved by different layers** — sparse cuts the `n²`
  FLOPs, ring shards the `O(n)` KV; Flash is the prerequisite. Need all three for 1M context.
- **External corroboration:** NSA reports **9.0× fwd / 6.0× bwd / 11.6× decode @64k** — real
  fwd+bwd compute reduction (not just masking), consistent with our `n/W` model. Recalibrate
  `W` and MFU floors against NSA's measured numbers during Phase 2.

## 7. What is determinable a priori (unchanged; condensed)

- **Compute — exact.** attention speedup `= n/W`; whole-model folds in the linear (QKVO+FFN) term.
- **Memory — exact.** non-Flash scores `O(n²)` per layer (Flash removes); KV `O(n)` → `O(n/p)` via ring.
- **Quality — not a tight a-priori number.** Provable: a **connected** pattern (our skeleton gives
  `O(log n)` reach) retains universal approximation (no forced ceiling). Per-token error bounded by
  dropped softmax mass: `‖o−õ‖ ≤ 2·δ·max‖v‖` (boundable only under a concentration assumption, which
  learned top-k makes hold). End-task loss is empirical → gate with loss parity + a **runtime
  dropped-mass certificate** (`δ̂` from routing scores). NSA's "+0.032 LongBench over full attention"
  is empirical evidence that trainable top-k need not degrade quality.

## 8. Revised phased build plan (build-vs-buy + gates)

| Phase | Deliverable | Build / Buy | Gate |
|---|---|---|---|
| **0. Core** | `AttentionAccumulator` (L0) — promote `_merge_block_attention` to a standalone module | **BUILD** | property tests: associativity, equals-one-joint-softmax, masked-partial safe, fp16/fp32 |
| **1. Static backend** | Wire L1 static patterns to **FlexAttention** (+ FA3 when available); local/dilated-stride/global skeletons as `BlockMask` builders | **BUY** | parity vs dense masked-softmax ref; FlexAttention fwd+bwd `gradcheck`; MFU floor |
| **2. NSA policy** | **Port** NSA (compression + learned top-k selection + sliding window) from fla-org; expose as an L2 policy with our interface | **PORT** | parity vs the NSA reference; selection actually skips compute (fwd+bwd); quality parity on a small LM |
| **3. Sparse ring (the differentiator)** | L3 ring/sequence-parallel reduction via L0; **skip communicating KV shards no local query selects**; global-position causal masking | **BUILD** | multi-GPU (`torchrun`) parity vs single-GPU; measured comm-volume reduction + load balance |
| **4. Training recipe** | **Muon** (2D matrices) + AdamW (embeddings/norms/head) split; long-context benchmark harness | **ADOPT** | loss-curve parity vs AdamW baseline; throughput; the audit's no-silent-cap discipline |
| **5. Consolidation** | Route existing variants through the engine; deprecate the broken bespoke classes | **BUILD** | benchmark-suite parity (tokens/s, peak mem/GPU, loss parity) |

**Foundation already on `main`:** L0 seed (`_merge_block_attention`, #29), ring K/V-aliasing fix
(#27), the audit + cost calculator. The old v1 "Phase 3: hand-write a Triton Flash kernel" is
**deleted** — replaced by Phases 1–2 above (buy + port).

## 9. Confidence ledger & open loose ends

**✓ Fact-checked (3-vote adversarial, 2 research passes):** FlexAttention mechanism (fused
score_mod/mask_mod + BlockMask, fwd+bwd); NSA design + 9×/6× + quality; fla-org/lucidrains repos
(MIT, Triton kernels, port-ready); FA3 (beta) / FA4 (alpha); Muon (adopt; ~52% AdamW FLOPs,
1T-scale via MuonClip, low integration cost).

**◐ Not yet fact-checked — Phase-2 follow-up (in progress):**
1. **FlexAttention content-adaptive crux** — can `create_block_mask` cheaply support *data-dependent*
   top-k selection per forward, or only static masks? (Determines L1/L2 boundary.)
2. **DeepSeek-V3.2 DSA** ("lightning indexer") — the production NSA successor; what it changes.
3. **DeepSeek V4** — does it improve on V3.2's attention?
4. **MLA** (low-rank latent KV compression) — training vs inference benefit; composability.
5. **MoBA** — alternative trainable block-sparse router.
6. **Muon variants + AdEMAMix** — variant landscape and AdEMAMix (and Muon×AdEMAMix) as optimizer options.

**Caveats:** Muon's "~2×" is first-party and shrinks at scale (~1.4×→~1.1× at 1.2B independently);
frontier stability needed MuonClip, not vanilla Muon. NSA headline numbers are first-party (27B),
not independently reproduced at scale. FA3 beta / FA4 alpha.

## 10. Reference materials

Cached in `docs/references/` (papers in `docs/references/papers/`, repo pointers in
`docs/references/README.md`): NSA (2502.11089), FlashAttention-3 (2407.08608), Muon/Moonlight
(2502.16982), FlexAttention (2412.05496); repos: fla-org/native-sparse-attention,
lucidrains/{native-sparse-attention,ring-attention,local-attention}-pytorch,
KellerJordan/Muon + MoonshotAI/Moonlight, Dao-AILab/flash-attention.

---

*Cost figures: `analysis/attention_cost_analysis.py`. Prior-art verdicts: deep-research passes
(2026-05-30). v1: `docs/archive/unified-attention-architecture-v1.md`.*
