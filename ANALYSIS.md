# ANALYSIS — fraud-provable LLM inference, end to end

*A synthesis of what was built and what was measured. Every number here is
reproduced by a committed binary or test; where a result is noisy or
negative it is reported as such. Companion docs: [SPEC.md](SPEC.md) (the
normative design), [PRIOR_ART.md](PRIOR_ART.md) (the landscape),
[benches/README.md](benches/README.md) (the raw measurement tables).*

## 1. The thesis

A prediction market needs a resolver it can trust without trusting the
resolver. We make an LLM judge's verdict **optimistically final** and
**cheaply falsifiable**: the resolver asserts `(output, final_state_root)`
on-chain with a bond; anyone who recomputed the same model and disagrees
opens a bisection game that converges, in `⌈log₂ N⌉` rounds, to a single
disputed micro-operation; that one micro-op is re-executed by a ~500-line
Sui Move contract, and the comparison — not the word of either party —
decides who is slashed.

The bet is that this can be done with **no quality loss and no visible
speed penalty** versus an ordinary deployment, because:

- **Determinism is free on the honest path.** The committed arithmetic is
  integer-only; integer addition is associative, so *any* kernel order —
  scalar, NEON, multi-threaded, GPU — produces the bit-identical result.
  There is no "deterministic mode" tax (§9.1). This is the structural
  advantage over float engines, where reproducibility costs a fixed
  reduction order.
- **Commitment is the only honest-path cost, and it hides.** State is a
  Merkle tree over 1 KiB pages; a token dirties a bounded set of pages;
  hashing them runs on a second thread behind the next token's compute.
  Measured wall-clock overhead: **0–3%** (§4).
- **The dispute is bounded and tiny.** One micro-op, a few page openings,
  `verify_step` — the chain's minimum compute bucket (§3).

## 2. What was built (and proven)

| Layer | Artifact | Proof it works |
|---|---|---|
| ISA + VM | `vm/` — 25-op integer ISA, tagged-SHA3 Merkle memory, total step relation, one-step verifier | 32 conformance goldens + 4000-trial verifier↔machine equivalence fuzz |
| Toy judge | `models/toy` + `compiler/` + `game/` | full local fraud game, 100-seed dishonest-resolver fuzz: challenger always wins, isolated step == fault step |
| On-chain verifier | `dispute/` Sui Move package | 43 Move tests incl. exhaustive MAC8, Rust↔Move cross-vectors, real localnet dispute (Phase 2) |
| Real model | `models/qwen` — integer Qwen3-0.6B, pure-integer LUTs/rope, W8A8(+i16) quant, float reference | float-ref PPL 34.60 ≈ llama Q8 34.99 (validates the float path + eval convention) |
| Speed | `kernels/` — persistent pool + NEON, GPU (`gpu/`) | 151,936 LM-head logits **bit-identical GPU vs CPU** on Apple M4 |
| **Qwen → VM compiler** | `compiler/src/qwen.rs` | **C-14 at scale: native checkpoint roots == VM-oracle roots, bit-identical at every token boundary** |
| **Qwen fraud game** | `game/src/bin/qwen_dispute.rs` | one flipped weight bit in a 29.5M-step judgment, **isolated in 25 rounds, convicted** |
| **On-chain conviction** | `dispute/tests/qwen_conviction.move` | the challenger's atomic-step **DOTBM** StepProof — the **Sui Move verifier convicts the real Qwen fault** |

The last three rows are the load-bearing result: the whole chain runs
end-to-end over a real LLM, and the on-chain contract convicts a real
fault, with the predictor running at native integer speed.

## 3. The dispute, measured

A faulty resolver flips one bit of one weight byte (layer-14 gate matrix)
partway through a Qwen judgment. Both parties hold a cursor machine pinned
at the agreed `lo` (which only increases) and answer each midpoint query
by cloning and advancing — **no per-step trace is materialized** (29.5M
roots is infeasible; the two-level scheme of SPEC §7.4/§8.6 makes it
unnecessary). Measured (`qwen_dispute`, 2+1 tokens, one core):

- **25 bisection rounds** narrow `[0, 29 551 781]` to the exact corrupted
  transition (step 13 889 868).
- The challenger submits one `DOTBM` StepProof; `verify_step` recomputes
  the honest post-state from the agreed pre-state and finds it ≠ the
  resolver's claim → **ChallengerWins**, resolver slashed.
- The same StepProof, emitted as a Move vector, is convicted by the actual
  Sui Move contract (`sui move test`), with **no 1 GiB image on-chain** —
  a StepProof carries only the opened pages and their sibling hashes.

On-chain cost (Phase 2 localnet, toy): `verify_step` = the chain's
**minimum** compute bucket (1,000,000 MIST), ~557 bytes of calldata for a
register-light op. One micro-op is computationally trivial to verify; that
is the point.

## 4. Honest-path overhead, measured

Two separable claims, both measured at Qwen scale (`qwen_demo`):

1. **Determinism tax = 0.** The integer kernels are bit-exact regardless of
   thread count or SIMD width; the GPU produces bit-identical logits. There
   is nothing to measure because there is no special mode — the fast path
   *is* the committed path.
2. **Commitment overhead = 0–3% wall-clock.** Per-token the model dirties
   a bounded page set (KV append + activation scratch; the LM-head logits
   never enter committed state thanks to the streaming `ARGMAX_OFF` head).
   Sequential hashing is ~4.5% of compute; pipelined behind the next token
   it falls into the run-to-run noise (0.00% on the cleanest run, roots
   asserted identical to the sequential path).

Extrapolation and the levers (parallel leaf hashing, HW SHA3, checkpoint
sparsity) are in [benches/README.md](benches/README.md).

## 5. Quality, measured — and the honest gap

Bars: our float reference **PPL 34.60**, llama.cpp Q8_0 **34.99** (same
text, llama-perplexity convention). The float reference matching llama
within noise validates the float implementation, the integer rope tables,
and the eval convention in one number.

The integer path reaches **PPL ~421, top-1 agreement 20.5%** — coherent
English judgments, but an ~11× perplexity gap to the bar. The full
measured ladder (calibration corpus, SmoothQuant α, headroom, per-block
scales) is in [benches/README.md](benches/README.md). Two findings worth
stating plainly:

- **The gap is resolution, not clipping.** Percentile activation
  calibration — the obvious "reclaim range from outliers" lever — was
  implemented and **measured to make things worse** (PPL 1031–1220 vs
  421). The large q/k/gate/up/v activations are *signal*: attention logits
  and SwiGLU gates ride their tail, and a tighter saturating scale destroys
  it. The lever is *more bits* on the worst sites, not narrower range.
- **The float-ISA path (FW-6) is exact by construction.** When the
  committed DOT semantics are a canonical fixed-order fp32 reduction (the
  same one batch-invariant GPU kernels already perform), the integer
  quantization gap disappears entirely and quality is the model's own. That
  is the strategic end-state; the integer path de-risks everything else
  first.

This is the one place the "no quality loss" goal is not yet met on the
integer path, and it is reported as an open gap with a measured diagnosis,
not papered over.

## 6. Speed, measured — and the honest gap

~28 tok/s integer decode vs llama.cpp Q8_0 101 tok/s (pure CPU) — a ~3.6×
gap. **Single-run tok/s on this machine drifts ±20% with thermals**, so
kernel changes were evaluated by a thermal-robust same-process A/B ratio
(`benches/kernel_ab`), not wall-clock:

- A fused blocked-GEMV kernel (one horizontal reduction per block instead
  of four) is a real but small **+3%** — kept.
- An `sdot` two-limb decomposition (bit-exact, 4× fewer MAC instructions)
  measured **0.75× — slower**: inline `asm!` is opaque to the compiler's
  scheduler, killing cross-block pipelining. The win needs nightly
  `vdotq_s32` intrinsics or a hand-pipelined whole-row asm loop.
- Fusing q/k/v into one pool dispatch was also slower — pool barriers are
  not the bottleneck.

The root cause is precise and **inherent to the quality choice**: i16
activations force `vmlal_s16` (4 MACs/instruction); llama's Q8 path uses i8
activations and `sdot` (16 MACs/instruction). Closing the gap means either
nightly DotProd intrinsics or per-block *dynamic* i8 activations (llama's
design — a quality re-derivation). Crucially, **the protocol is
kernel-agnostic** (§9.1): any bit-exact kernel is admissible, so this is an
optimization runway, not a design constraint.

## 7. Where this sits versus prior art

- **opML** (LLaMA-7B over a MIPS FPVM) validates the two-level bisection
  but bridges two state representations at the phase seam; ours samples
  **one** root sequence over **one** memory tree — no seam gadget.
- **EigenAI / Gensyn** achieve bitwise-deterministic GPU inference but
  settle disputes by node/TEE committees; ours settles by a tiny L1
  contract. FW-6 makes our committed DOT *be* their canonical reduction —
  their speed, our trust root.
- **zkML** is exact but not LLM-scale for real-time judging.

Nobody production-grade does trustless **L1** verification of a real LLM
today. The integer core demonstrated here, plus the FW-6 float track, is a
concrete path into that open lane.

## 8. Invariants — held

1. **Bit-exactness** — integer-only committed path, no floats in the
   runtime (clippy-enforced; floats quarantined to offline quantization).
2. **Deterministic ordering** — `BTreeSet` tree writes, fixed reduction
   structure; CI-diffed against the sequential oracle.
3. **Everything committed by hash** — tagged SHA3-256 pages and state
   roots; artifacts by blake3 identity.
4. **Tiny on-chain verifier** — `verify_step` is one micro-op, the chain's
   minimum compute bucket; the Move interpreter stays in the few-hundred-
   line budget even with the v0.4.0 wide ops.

## 9. Open gaps (stated, not hidden)

| Gap | Status | Path |
|---|---|---|
| Integer quality ~11× PPL to bar | measured; percentile lever ruled out | more activation bits on worst sites; ultimately FW-6 (exact by construction) |
| CPU speed ~3.6× to llama | measured; root cause = i16 MAC density | nightly DotProd intrinsics, or dynamic-i8 activations |
| GPU integer GEMV perf | bit-exactness proven; perf is dp4a-class HW dependent | packed dot4I8, on-GPU i64 reduction |
| Localnet Qwen E2E (real txs) | on-chain conviction proven via Move vector | drive the existing localnet client over the Qwen program |
| `run_committed` register file | provisional; mem roots already match VM | wire the compiler's pinned boundary registers |

The trustless-verification thesis is demonstrated end-to-end at real-model
scale. The remaining work is performance and quality engineering against
clearly-measured walls, not open design questions.
