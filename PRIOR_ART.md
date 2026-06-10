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
