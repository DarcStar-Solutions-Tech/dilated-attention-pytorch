export const meta = {
  name: 'dilated-attention-scoped-audit',
  description: 'Scoped correctness audit of dilated-attention-pytorch core library via 1M-context lens agents',
  phases: [
    { title: 'Orient',     detail: 'map public API + high-risk/forked files (whole src/ in one context)' },
    { title: 'Review',     detail: 'one lens agent per dilated/ring-attention failure mode' },
    { title: 'Verify',     detail: 'adversarial skeptics per finding (refute or repro)' },
    { title: 'Synthesize', detail: 'dedup, rank by severity, write the report' },
  ],
}

// --- Preconditions ---------------------------------------------------------
// Agents inherit the SESSION model. This workflow is DESIGNED for a 1M-context
// model: the Orient and Synthesize agents hold all of src/ (~230k tokens) at
// once, and each lens holds its entire domain (e.g. all of ring/ ~60k tokens)
// without sharding. On a 200k model it still runs, but Orient/Synthesize and
// the ring-distributed lens will be context-pressured.

const ROOT = '/home/mharris/Projects/DarcStar-Technologies/dilated-attention-pytorch'
const PKG  = `${ROOT}/src/dilated_attention_pytorch`
const VERIFIERS = 2   // adversarial skeptics per finding. 1 ≈ ~1.0M tokens, 2 ≈ ~1.7M, 3 ≈ ~2.5M

// --- Schemas ---------------------------------------------------------------
const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          title:        { type: 'string' },
          file:         { type: 'string', description: 'repo-relative path' },
          lines:        { type: 'string', description: 'e.g. "120-145"' },
          severity:     { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:     { type: 'string' },
          description:  { type: 'string', description: 'the incorrect behavior, precisely' },
          reasoning:    { type: 'string', description: 'why it is wrong; the trace or math' },
          suggested_fix:{ type: 'string' },
        },
        required: ['title', 'file', 'lines', 'severity', 'category', 'description', 'reasoning', 'suggested_fix'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    isReal:            { type: 'boolean', description: 'true only if the defect genuinely exists and reaches live code' },
    confidence:        { type: 'string', enum: ['high', 'medium', 'low'] },
    corrected_severity:{ type: 'string', enum: ['critical', 'high', 'medium', 'low', 'none'] },
    assessment:        { type: 'string', description: 'the guard/invariant that refutes it, OR the concrete input that triggers it' },
  },
  required: ['isReal', 'confidence', 'corrected_severity', 'assessment'],
}

const ORIENTATION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    public_api:           { type: 'array', items: { type: 'string' }, description: 'classes/functions exported from __init__.py and built by factory.py' },
    high_risk_files:      { type: 'array', items: { type: 'object', additionalProperties: false, properties: { file: { type: 'string' }, why: { type: 'string' } }, required: ['file', 'why'] } },
    forked_or_versioned:  { type: 'array', items: { type: 'object', additionalProperties: false, properties: { files: { type: 'string' }, canonical: { type: 'string' }, concern: { type: 'string' } }, required: ['files', 'canonical', 'concern'] } },
    notes:                { type: 'string' },
  },
  required: ['public_api', 'high_risk_files', 'forked_or_versioned', 'notes'],
}

// --- Lenses, tuned to dilated/ring-attention failure modes -----------------
const LENSES = [
  {
    key: 'numerical',
    title: 'numerical correctness (attention math)',
    scope: [
      `${PKG}/base/`,
      `${PKG}/ring/utils/ring_attention_lse.py`,
      `${PKG}/ring/base/ring_dilated_attention_correct.py`,
      `${PKG}/ring/base/ring_dilated_attention_sdpa.py`,
      `${PKG}/utils/attention_utils.py`,
      `${PKG}/utils/validation.py`,
    ],
    focus: [
      'Online-softmax / log-sum-exp accumulation: running-max tracking, exp rescaling of prior partial outputs across ring/segment steps — an off-by-one or missing rescale silently corrupts results.',
      'Attention scaling (1/sqrt(head_dim)) applied exactly once, before softmax.',
      'Causal masking correctness, especially when sequence length is NOT divisible by the dilation rate or segment length.',
      'Dilation/segment gather-scatter indexing: does the dilated pattern select the intended keys?',
      'Accumulation dtype/precision (fp16/bf16 accumulating into fp32?).',
    ].join('\n  - '),
  },
  {
    key: 'ring-distributed',
    title: 'ring + distributed communication correctness',
    scope: [
      `${PKG}/ring/`,
      `${PKG}/ring/base/ring_communication_mixin.py`,
      `${PKG}/ring/utils/ring_communication_fix.py`,
      `${PKG}/ring/distributed_ring_attention.py`,
      `${PKG}/ring/distributed/ring_distributed_dilated_attention.py`,
    ],
    focus: [
      'K/V block rotation around the ring: is every rank-pair visited exactly once, no double-count, no skip?',
      'send/recv ordering and potential deadlock; isend/irecv waited correctly.',
      'rank / world_size / group handling; behavior when world_size == 1.',
      'WHY ring_communication_fix.py exists: what bug it patches, whether the fix is applied consistently everywhere the bug occurs, and whether the unpatched path is still reachable.',
      'Partial-output accumulation across ring steps stays consistent with the LSE rescaling.',
    ].join('\n  - '),
  },
  {
    key: 'autograd',
    title: 'backward / autograd correctness',
    scope: [
      `${PKG}/ring/utils/ring_attention_autograd.py`,
      `${PKG}/base/head_parallel_dilated_attention_optimized.py`,
      `${PKG}/ring/base/base_ring_attention.py`,
    ],
    focus: [
      'Custom autograd.Function: does backward implement the exact gradient of forward?',
      'Saved tensors vs recomputation; nothing needed in backward is freed or detached.',
      'Ring communication replayed correctly in the backward pass (gradients flow around the ring the right direction).',
      'Gradient checkpointing recompute matches the original forward (same RNG/masking).',
    ].join('\n  - '),
  },
  {
    key: 'memory',
    title: 'memory: O(n) claims, pools, and caches',
    scope: [
      `${PKG}/core/unified_memory_pool.py`,
      `${PKG}/core/memory_profiler.py`,
      `${PKG}/core/optimized_pattern_cache.py`,
      `${PKG}/core/simple_gpu_cache.py`,
      `${PKG}/core/pattern_cache.py`,
    ],
    focus: [
      'Does ring attention actually achieve O(n) memory, or does some buffer/materialized score matrix make it O(n^2)? Check against any complexity claims in docstrings.',
      'Memory pool: double-free, aliasing of live tensors, leaks, returning buffers still referenced elsewhere.',
      'Pattern cache: invalidation and eviction correctness; stale pattern served after config change; key collisions across different segment/dilation configs.',
    ].join('\n  - '),
  },
  {
    key: 'kernels',
    title: 'Triton kernel correctness (Hilbert attention)',
    scope: [
      `${PKG}/kernels/`,
      `${PKG}/kernels/hilbert_attention_core.py`,
      `${PKG}/kernels/hilbert_attention_triton_wrapper.py`,
      `${PKG}/utils/hilbert_attention_mixin.py`,
      `${PKG}/utils/hilbert_curve.py`,
    ],
    focus: [
      'Triton block/grid indexing and masking for sequence lengths not divisible by BLOCK size (out-of-bounds loads/stores, missing boundary masks).',
      'dtype handling and accumulation inside the kernel.',
      'autotune config validity (BLOCK sizes, num_warps) vs the indexing math.',
      'Hilbert index mapping (hilbert_curve.py) vs how the kernel/wrapper consume it: does the reorder + inverse-reorder round-trip exactly, and is it applied consistently per-segment?',
      'Wrapper fallback path: does the non-Triton (eager/PyTorch) branch in hilbert_attention_triton_wrapper.py compute the SAME result as the Triton kernel, and is the Triton path correctly gated on availability/shape?',
    ].join('\n  - '),
  },
  {
    key: 'sparse-deadcode',
    title: 'sparse pattern correctness + forked/dead files',
    scope: [
      `${PKG}/sparse/`,
      `${PKG}/sparse/block_sparse_adaptive.py`,
      `${PKG}/sparse/block_sparse_adaptive_fixed.py`,
      `${PKG}/sparse/block_sparse_attention.py`,
      `${PKG}/sparse/block_sparse_attention_fixed.py`,
    ],
    focus: [
      'Block-sparse mask generation: does the produced mask match the intended dilation/sparsity pattern? Any blocks wrongly zeroed or wrongly kept?',
      'Sparse causal masking correctness at block boundaries.',
      'The *_fixed.py forks: is the (presumably buggy) non-fixed original still imported, exported from __init__, or built by a factory? If so, the known bug is still live.',
    ].join('\n  - '),
  },
]

// --- Prompts ---------------------------------------------------------------
function reviewPrompt(lens, orientText) {
  return [
    `You are auditing the dilated-attention-pytorch library through ONE lens: ${lens.title}.`,
    ``,
    `Shared orientation from the mapping pass (use it to judge whether a file is live and how severe a bug is):`,
    orientText,
    ``,
    `Read these paths IN FULL — they fit in your context, so read them, do not sample:`,
    ...lens.scope.map(s => `  ${s}`),
    ``,
    `Hunt specifically for:`,
    `  - ${lens.focus}`,
    ``,
    `Report ONLY real defects: correctness, numerical, memory-safety, distributed-correctness, or dead/forked code that is actually wired into live paths. No style nits, no speculative refactors.`,
    `For each finding give an exact file path + line range, a precise description of the WRONG behavior, the reasoning/trace or math that proves it, and a concrete fix.`,
    `If you find nothing real, return an empty findings array — do not invent issues.`,
  ].join('\n')
}

function verifyPrompt(f, stance) {
  const base = [
    `A prior audit agent reported this finding in dilated-attention-pytorch. Verify it independently by reading the actual code.`,
    ``,
    `  Title:    ${f.title}`,
    `  File:     ${f.file}  (lines ${f.lines})`,
    `  Severity: ${f.severity}`,
    `  Claim:    ${f.description}`,
    `  Reasoning:${f.reasoning}`,
    ``,
    `Open ${ROOT}/${f.file} and read the relevant region plus any callers/guards it depends on.`,
  ]
  const lens = stance === 0
    ? [
        ``,
        `Your stance: ASSUME THE FINDING IS WRONG. Look hard for the guard, invariant, type constraint, or upstream check that makes the code correct. Default to isReal=false unless you cannot find any such protection.`,
      ]
    : [
        ``,
        `Your stance: TRY TO TRIGGER IT. Construct a concrete input (tensor shapes, dtype, seq length vs dilation/segment, world_size, causal flag) that would actually exercise the bug. If you can specify a triggering case, isReal=true; if every realistic input is guarded, isReal=false.`,
      ]
  return [...base, ...lens].join('\n')
}

function synthPrompt(findings) {
  return [
    `You are writing the final report for a scoped correctness audit of the dilated-attention-pytorch core library.`,
    `Below are findings that survived adversarial verification (each tagged "confirmed" = both skeptics agreed it is real, or "contested" = split decision). Some may be near-duplicates reported by different lenses.`,
    ``,
    JSON.stringify(findings, null, 2),
    ``,
    `Produce a markdown report:`,
    `  1. One-paragraph executive summary (how many confirmed, by severity, the single most serious issue).`,
    `  2. Findings grouped by severity (critical -> low). Merge duplicates. Each: title, file:lines, what's wrong, why it matters, suggested fix. Mark contested ones clearly.`,
    `  3. A short "forked/dead code" section if any *_fixed / versioned-variant issues were confirmed.`,
    `Be precise and cite file:line. Do not pad.`,
  ].join('\n')
}

// --- Orchestration ---------------------------------------------------------
phase('Orient')
const orientation = await agent(
  [
    `Map the dilated-attention-pytorch package at ${PKG} to orient a downstream audit.`,
    `Read __init__.py, core/factory.py, ring/factory.py, sparse/block_sparse_factory.py and skim the rest of the tree.`,
    `Report: the public API surface (what's exported / built by factories), the highest-risk files and why,`,
    `and ESPECIALLY every set of forked/versioned/duplicated files — e.g. the *_fixed.py sparse files (block_sparse_attention_fixed.py, block_sparse_adaptive_fixed.py), the duplicated ring_communication_mixin.py (one under ring/base/, one under ring/utils/) alongside ring_communication_fix.py, and the deprecated memory pools (memory_pool / enhanced / bucketed / fragment_aware / numa) versus unified_memory_pool.py — for each say which is canonical and whether the non-canonical one is still imported, exported, or built by a factory.`,
  ].join('\n'),
  { schema: ORIENTATION_SCHEMA, label: 'orient', phase: 'Orient' }
)
const orientText = orientation ? JSON.stringify(orientation, null, 2) : '(orientation unavailable)'
log(`Orientation complete; running ${LENSES.length} lenses with ${VERIFIERS} verifiers each.`)

// Review -> Verify as a pipeline: each lens's findings start verifying as soon
// as that lens completes (no barrier between lenses).
const perLens = await pipeline(
  LENSES,
  (lens) => agent(reviewPrompt(lens, orientText), { schema: FINDINGS_SCHEMA, label: `review:${lens.key}`, phase: 'Review' }),
  (review, lens) => parallel(
    (review?.findings || []).map((f) => () =>
      parallel(
        Array.from({ length: VERIFIERS }, (_, i) => () =>
          agent(verifyPrompt(f, i), { schema: VERDICT_SCHEMA, label: `verify:${lens.key}:${i}`, phase: 'Verify' })
        )
      ).then((verdicts) => ({ finding: f, lens: lens.key, verdicts: verdicts.filter(Boolean) }))
    )
  )
)

// Classify by adversarial vote.
const classified = perLens
  .flat()
  .filter(Boolean)
  .map(({ finding, lens, verdicts }) => {
    const real = verdicts.filter((v) => v.isReal).length
    const refuted = verdicts.length - real
    const status = real > refuted ? 'confirmed' : real === refuted ? 'contested' : 'dropped'
    return { ...finding, lens, status, verdicts }
  })
  .filter((f) => f.status !== 'dropped')

log(`${classified.filter(f => f.status === 'confirmed').length} confirmed, ${classified.filter(f => f.status === 'contested').length} contested after verification.`)

phase('Synthesize')
const report = await agent(synthPrompt(classified), { label: 'synthesize', phase: 'Synthesize' })

return {
  confirmed: classified.filter((f) => f.status === 'confirmed').length,
  contested: classified.filter((f) => f.status === 'contested').length,
  findings: classified,
  report,
}
