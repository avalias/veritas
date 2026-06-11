# PRIOR_ART.md — Fraud-provable / verifiable LLM inference

```
Status:  survey as of 2026-06-10, feeding SPEC.md v0.2.x
Verdict: prior art VALIDATES the architecture (two-level traces, predictor/
         reference split, fixed-point determinism, bisection-to-one-op).
         No structural spec change required. Sui Move remains greenfield.
```

## 1. opML / ORA — the closest prior art

[opML: Optimistic Machine Learning on Blockchain (arXiv 2401.17555)](https://arxiv.org/abs/2401.17555),
[ora-io/opml](https://github.com/hyperoracle/opml). Deployed by ORA as an
on-chain AI oracle (EVM).

- **What it is:** optimistic assertion of ML inference with an interactive
  bisection game ending in a single MIPS instruction executed by an on-chain
  FPVM (Solidity). Ran **LLaMA-7B** on commodity CPUs this way.
- **Multi-phase protocol:** Phase 1 bisects over the *computation graph*
  (per-node states, executed natively with CPU/GPU); Phase 2 re-executes the
  disputed node inside the MIPS VM and bisects micro-instructions. This is
  the direct ancestor of our two-level trace (§8.6).
- **Determinism:** fixed-point/quantized arithmetic plus SoftFloat fallback.
  Confirms our premise that float-free (or float-emulated) execution is the
  price of admission.
- **Their costs we avoid:**
  - The phase-1→phase-2 transition *bridges two state representations*
    (graph state → VM memory image), which needs an extra verification
    gadget at the seam. Our levels are sparser/denser samples of **one**
    root sequence over **one** state representation (VM memory tree) — no
    bridge, no seam (§7.4).
  - Their on-chain referee interprets **MIPS** (full CPU ISA in Solidity);
    ours interprets 21 tensor micro-ops in Move (~330 lines, §1.3).
  - MIPS FPVM memory cap of 4 GB constrained their model loading; our
    address space is 16 GiB (d ≤ 24, §3.1) and weights never move — they
    are genesis pages.

## 2. Cartesi — deterministic RISC-V machine, LLM demos

[cartesi/machine-emulator](https://github.com/cartesi/machine-emulator),
[cartesi/dave](https://github.com/cartesi/dave),
[edubart/machine-kernels-llama2.c](https://github.com/edubart/machine-kernels-llama2.c),
[ThinkChain PoC](https://rolluplab.cartesi.io/thinkchain/).

- A full RISC-V Linux machine with Merkleized state and bisection to a
  single instruction replayed in Solidity. Llama-2 has been run inside it
  deterministically; the interesting trick in `machine-kernels-llama2.c` is
  **offloading matmuls to the host CPU for speed while a RISC-V kernel can
  replay the same matmul for proofs** — i.e., exactly our predictor /
  reference split (§9.2), independently re-invented.
- **Dave / Permissionless Refereed Tournaments:** their answer to
  multi-challenger and sybil-griefing disputes. We cite it as the design to
  study for our FW-2 (multi-challenger) future work.
- Why we don't build on it: the one-step verifier is a full RV64 interpreter
  in Solidity (nothing exists for Sui Move), and committed state is
  incidental CPU state (stack/heap), so only emulation can reproduce
  checkpoints — the honest path pays emulator speed (see SPEC §1.4 and the
  predictor-split rationale).

## 3. Gensyn — Verde + RepOps (deterministic floats across GPUs)

[Verde: Verification via Refereed Delegation for ML (arXiv 2502.19405)](https://arxiv.org/abs/2502.19405),
[gensyn-ai/repops-demo](https://github.com/gensyn-ai/repops-demo),
[Gensyn blog](https://blog.gensyn.ai/verde-a-verification-system-for-machine-learning-over-untrusted-nodes/).

- **RepOps:** a library making float ML operators **bitwise reproducible
  across GPUs** by fixing reduction order (plus correctly-rounded
  functions). Proves "fraud-proof GPUs" are real at the determinism level.
- **Verde:** dispute resolution pinpoints the first disagreeing *operator*
  in the computation graph; the referee recomputes only that operator.
- **The trust-model difference that keeps us integer-only:** Verde's
  referee must re-execute a whole operator (e.g. a full matmul) — feasible
  for a *node* acting as referee, impossible for an L1 contract. Our hard
  requirement is a few-hundred-line Sui Move referee, which caps the
  re-executed unit at one micro-op — hence integer state, Merkle pages, and
  bisection all the way down. RepOps-style deterministic-float GPU code is
  still valuable to us — as a **predictor** (§9.2) and as a possible future
  relaxation *if* the referee ever moves off-chain (FW-5).

## 4. EigenAI and the TEE/committee lane (2025–2026)

[EigenAI: Deterministic Inference, Verifiable Results (arXiv 2602.00182)](https://arxiv.org/abs/2602.00182),
[EigenCloud blog](https://blog.eigencloud.xyz/deterministic-ai-inference-eigenai/),
[Optimistic TEE-Rollups (arXiv 2512.20176)](https://arxiv.org/pdf/2512.20176),
[LLM-42 (arXiv 2601.17768)](https://arxiv.org/html/2601.17768v1),
[VeriLLM (arXiv 2509.24257)](https://www.arxiv.org/pdf/2509.24257),
survey: [Equilibrium Labs — State of Verifiable Inference](https://equilibrium.co/writing/state-of-verifiable-inference).

- EigenAI: bit-exact deterministic LLM inference **on production GPUs**
  (<2% overhead claimed) + optimistic re-execution backed by restaking;
  disputes resolved by a **committee re-executing in TEEs**. Optimistic
  TEE-Rollups similarly lean on H100 confidential-computing TEEs.
- Positioning: these systems trade our trust assumptions for speed — the
  final arbiter is a TEE/committee (hardware + quorum trust), not a public
  L1 contract anyone can verify. Our design keeps the arbiter a dumb
  contract; theirs is the right comparison column for `ANALYSIS.md`
  (Phase 4) alongside zkML.
- **2026 update — the determinism layer is now open commodity:** Thinking
  Machines' batch-invariant ops, [SGLang deterministic mode](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
  (production engine; ~34% overhead with CUDA graphs per LMSYS, improving),
  and [vLLM's official batch-invariance feature](https://docs.vllm.ai/en/latest/features/batch_invariance/).
  EigenAI's paper documents the strongest kernel discipline (deterministic
  block→tile mapping, no inter-block communication, **canonical binary-tree
  warp reduction** — ~2% claimed overhead, 95% GEMM throughput on Hopper).
  That kernel spec is, in effect, a draft of a *committed float semantics*:
  if our DOT micro-op adopts the same canonical tree as its normative
  reduction order, the GPU engine becomes a bit-exact implementation of the
  committed semantics and the only missing piece is a softfloat one-step
  verifier in Move. This is SPEC FW-6 — the candidate flagship invention:
  deterministic-engine speed, full model quality, L1-trustless slashing.

## 5. What this changes in our design

| Finding | Action |
|---|---|
| opML proved LLaMA-scale models survive bisection fraud proofs end-to-end | Confidence, not change — proceed |
| opML's two-phase game has a representation seam between phases | Keep our single-representation two-level design (§7.4, §8.6) — now an explicit, defended differentiator |
| Cartesi independently uses host-compute + VM-replay | Keeps §9.1/§9.2 split; cite as precedent |
| Cartesi Dave (PRT) solves multi-challenger tournaments | FW-2 now points at Dave as the reference design |
| RepOps / EigenAI: deterministic float GPU inference exists | New FW-5: deterministic-GPU **predictor** backend; integer core unchanged (referee must stay a tiny contract) |
| No fraud-proof / bisection infra found on Sui Move | Phase 2 is greenfield as assumed; nothing to reuse, nothing to conflict with |
| EigenAI/TEE lane is the market-adjacent alternative | Add as a comparison column in Phase 4 `ANALYSIS.md` |

**Net: no structural change to SPEC.md.** Two future-work rows added and
informative citations wired into §8.6/§9.2.

---

## 6. June 2026 refresh: fastest engines + the determinism/verification landscape

*(Researched 2026-06-11 with sources; informs the FW-6 backend strategy:
"rely on their tech for speed, add the missing piece — adjudication.")*

### Fastest engines (batch-1 decode is bandwidth-bound everywhere)

| platform | fastest today | notes |
|---|---|---|
| CPU | llama.cpp (de-facto reference); ik_llama.cpp fork up to 1.5–2.1× on x86 quants | every desktop stack wraps it |
| Apple Silicon GPU | **MLX** ≈1.4–1.8× llama.cpp-Metal on small dense models; Ollama switched to MLX (v0.19) | llama.cpp still wins long-context |
| CUDA bs=1 | llama.cpp-CUDA / ExLlama for quantized small models; TRT-LLM ~+8% over vLLM at concurrency 1 | RTX 4090 ~189 t/s @7B Q4 (ggml #15013) |

### Deterministic-inference offerings — and how each VERIFIES correctness

| system | pins | overhead | verification |
|---|---|---|---|
| Thinking Machines `batch_invariant_ops` | batch invariance (fixed tiles/splits) | ~1.6× (PoC) | **none** — reproducibility only |
| SGLang deterministic mode | TML kernels + 3 attention backends + seeded sampling | 24–55% | **none** — own test suite |
| vLLM `VLLM_BATCH_INVARIANT=1` | FlexAttention + op substitution | unpublished | **none** |
| **EigenAI** (arXiv 2602.00182, mainnet 2026-01) | llama.cpp-CUDA fork: fixed block mapping, warp tree reductions, no FP atomics, pinned driver/container, one GPU SKU per pool | **≈1.8% E2E** | **optimistic re-execution by a TEE committee** (EigenLayer restaking, ≥2/3 vote); commits the FINAL OUTPUT HASH ONLY; a dispute re-runs the ENTIRE inference inside attested TEEs |

The survey's key sentence, verbatim: *"none of the four offerings commits
intermediate state, so none supports O(1)/logarithmic on-chain fraud
proofs of a single wrong operation … I found no published system in this
space offering per-operation challengeable commitments as of June 2026."*

### What this means for us

1. **Determinism is commodity** — EigenAI proves deterministic CUDA at
   ~1.8% overhead on llama.cpp's own kernels. We ADOPT that lineage (our
   committed reduction tree is exactly the pinned-kernel idea), we do not
   claim it.
2. **Adjudication is the open lane.** EigenAI's trust root is a staked
   TEE committee re-running the whole inference; ours is a few-hundred-
   line L1 contract executing ONE micro-op from a Merkle opening. Their
   challenge costs a full re-execution + committee vote; ours costs
   ~log₂(N) bisection txs + one minimum-gas `verify_step` — and we hold
   the on-chain artifacts to prove it (qwen_conviction.move for the
   integer path; softfloat.move block_dot for the float path).
3. **Backend adoption path:** pin the committed float tree to match the
   reduction shapes the fast engines already use (EigenAI's warp trees on
   CUDA; MLX/Metal with fast-math off — measured on M4), so *their*
   kernels become *our* predictor with ~zero overhead, and our protocol
   adds the missing property: cheap on-chain falsifiability.
