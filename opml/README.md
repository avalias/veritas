# opml: verifiable LLM inference for Sui

Run a real language model off-chain at close to native speed, and let anyone
prove on-chain that you ran it wrong. If the output is honest, nothing happens
and you pay almost nothing extra. If it's a lie, a challenger walks the
computation down to a single arithmetic operation and a ~330-line Sui Move
contract re-executes that one operation and slashes you.

There's no zero-knowledge proving and no trusted hardware anywhere in the loop.
The only thing you have to trust is the Sui contract, and you can read all of it.

This is the engine. The prediction market in `demos/` is one thing you can build
on top of it. We think there are many others.

---

## The problem

You want a chain to act on the output of a model it cannot run. The model is
billions of operations; the chain can afford a few thousand. Everyone solves
this in one of three ways, and each gives something up:

- **Re-run it in a TEE / a committee** (EigenAI, optimistic TEE-rollups). Fast,
  but the final arbiter is a trusted enclave or a staked quorum. A dispute
  re-runs the *entire* inference inside attested hardware. You trade the chain's
  trust for Intel's and a voting set's.
- **Prove it in zero knowledge** (zkML: EZKL, Risc0). No trust needed, but
  proving a single LLM forward pass is orders of magnitude slower than running
  it, and usually forces you to shrink or distort the model.
- **Optimistic fraud proofs** (opML/ORA, Cartesi). The trust model is right:
  anyone can challenge and the chain is the referee. But these systems run the
  model inside a general-purpose CPU emulator (MIPS, RISC-V). The honest path
  pays emulator overhead, and the on-chain referee is a whole ISA interpreter.

We took the third road and removed the thing that made it slow. The insight is
simple: you do not need a CPU emulator to run a neural network. A neural network
is a few dozen tensor operations. So the virtual machine *is* those operations,
the honest path runs your real kernels, and the on-chain referee only has to
know how to redo one of them.

---

## How it works

```
  your model (any quantized LLM, float activations)
        │  opml/compiler
        ▼
  a VM program: ~20k instructions over ~30M float micro-ops per token
        │
        ├─► run it fast off-chain          (opml/kernels, opml/gpu)
        │     your real NEON / Metal kernels (the speed path)
        │
        └─► commit a Merkle root of the trace, for free
              (the commitment is built as the forward pass runs)

  someone disputes your output
        │  opml/game  → on-chain bisection
        ▼
  log₂(N) rounds narrow 30M micro-ops to ONE
        │
        ▼
  opml/move :: dispute::verify_step           (~330 lines of Move)
  re-executes that single micro-op from a Merkle opening → slashes the liar
```

Three properties make it work, and each is tested, not asserted:

**The VM speaks tensors, not assembly.** Twenty-seven micro-ops: a block dot
product (`FDOT`), scalar float ops (`FOP`), the usual layout shuffles. The
on-chain interpreter for them is a few hundred lines of Move, not a CPU. One
disputed micro-op costs the minimum gas bucket to verify.

**The model is bit-for-bit deterministic, including on the GPU.** We compile to
a *committed float* semantics: a fixed reduction order, correctly-rounded
operations, soft-float in pure integers as the normative reference. Every
backend (scalar, NEON, 10-threaded, Apple Metal) produces the *identical*
logits. We measured all 151,936 logits of Qwen3-0.6B's LM head as bit-identical
across 1-vs-10 CPU threads and Metal-vs-CPU on an M4. That's what makes an
output disputable: there is exactly one correct answer, and it's the one the
contract will reproduce.

**The commitment is free.** The trace's Merkle root is built incrementally as
the model runs, in checkpoint mode, so you never materialize 30M micro-ops in
memory. Measured honest-path overhead over just running the predictor: **0–3%**.

The reference model is Qwen3-0.6B, run *as published*. Committed-float
perplexity is **34.5974**, matching the libm float reference (34.60) and
llama.cpp Q8 (34.99) on the same text. We don't quantize it into a worse model
to make it provable; the thing you dispute is the real one.

---

## Honest numbers

Everything here is measured on this machine (Apple M4), not projected.

| | result |
|---|---|
| model quality | committed-float PPL **34.5974** = published Qwen3-0.6B (libm 34.60, llama Q8 34.99) |
| determinism (CPU) | 151,936 logits bit-identical, 1 thread vs 10 |
| determinism (GPU) | 151,936 logits bit-identical, Metal vs CPU (fast-math off) |
| honest-path overhead | **0–3%** over the bare predictor |
| dispute size | one judgment ≈ 30M micro-ops → ~25 bisection rounds → one `verify_step` at chain-minimum gas |
| on-chain referee | ~330 lines of Move; integer + float conviction vectors on-chain |

The one number we won't dress up: raw decode speed. Our own reference predictor
runs Qwen3-0.6B at ~32–39 tok/s on CPU; llama.cpp Q8 does ~101 on the same box.
That's a ~3× gap, and it's a kernel-optimization gap, not an architectural one.
The determinism tax is the 0–3% above, measured on top of whatever predictor you
give it. Which is the real point:

**The fast deterministic engines are our speed path, not our competition.**
EigenAI shows deterministic CUDA at ~1.8% overhead on llama.cpp's own kernels;
MLX and Metal are bit-exact with fast-math off. Pin the committed reduction tree
to the shape those engines already use and *their* kernels become *our*
predictor with near-zero overhead. We adopt the determinism work the field has
done; what we add is the part nobody else has: cheap per-operation on-chain
falsifiability.

---

## How it compares

The honest one-line: everyone can now make inference *deterministic*. Almost
nobody can make a *single wrong operation* cheaply *provable on a public chain*.
A June-2026 survey of deterministic-inference systems put it plainly: none of
them commit intermediate state, so none supports an O(log N) on-chain fraud
proof of one wrong op.

| system | trust root on dispute | cost of a dispute | runs the real model? | on Sui? |
|---|---|---|---|---|
| **opml (this)** | a ~330-line public Move contract | ~25 txs + one min-gas micro-op | yes, bit-exact | **yes** |
| opML / ORA | a public contract, but a full **MIPS** interpreter | bisect + re-run a MIPS instruction | yes | EVM only |
| Cartesi | a public contract, full **RISC-V** interpreter | honest path pays emulator speed | yes | EVM only |
| Gensyn Verde | a referee **node** re-runs a whole operator | re-execute a full matmul | yes | off-chain referee |
| EigenAI / TEE | a staked **TEE committee** (≥2/3) | re-run the **entire** inference in enclaves | yes, fast | EVM / restaking |
| zkML (EZKL, Risc0) | math (no trust) | seconds-to-minutes of proving **per pass** | usually a shrunk model | partial |

Where we are genuinely better, and why:

- **Versus opML and Cartesi.** Same trust model (public referee, anyone
  challenges), but they emulate a CPU. Their referee is a whole ISA in Solidity;
  ours is 27 tensor ops in Move. They bridge two state representations across a
  two-phase game; we have one representation and no seam. Their model loading is
  capped by the emulator's address space; our weights never move, because they
  are genesis pages of the Merkle tree.
- **Versus Gensyn Verde.** Its referee re-executes a whole operator (a full
  matmul). That's fine when the referee is a beefy node. It is impossible for an
  L1 contract. Our hard constraint, that the referee must be a tiny contract, is
  exactly what forces bisection all the way down to one micro-op, and it's why
  we can live on a chain.
- **Versus EigenAI and the TEE lane.** This is the fastest of the bunch and the
  closest to us in spirit. The difference is the arbiter. Theirs is hardware plus
  a quorum vote, and a challenge re-runs the full inference in attested enclaves.
  Ours is a contract anyone can read, and a challenge re-runs one operation. We
  commit intermediate state; they commit only the final output hash. That single
  design choice is the whole game.
- **Versus zkML.** We never pay proving cost on the honest path. zkML pays it on
  *every* inference; we pay only when someone is actually lying, and even then
  it's ~25 cheap transactions, not a proof.

First time any of this exists on Sui Move. The chain has the native crypto we
lean on (keccak, ecrecover, ed25519, groth16, even Nitro attestation), and no
prior fraud-proof or bisection infrastructure to conflict with.

---

## Use it in your project

The engine is a compiler plus a Move package. Integration is three moves:

1. **Compile your model once.** Point `opml/compiler` at a quantized checkpoint
   with float activations; it emits the VM program and the genesis pages.
2. **Run it where you want, commit the root.** Use `opml/kernels` (CPU) or
   `opml/gpu` (Metal) to run the forward pass and emit the trace root. Your
   resolver posts `(input, output, root)` on-chain with a bond.
3. **Let anyone dispute.** Add the `opml` Move package as a dependency. A
   challenger who disagrees runs the same trace, finds the first differing
   checkpoint, and the bisection ends in `dispute::verify_step`. You write none
   of the hard part. The VM, the interpreter, the bisection, and the slashing
   are all here.

```toml
# your Move.toml
[dependencies]
opml = { local = "../opml/move" }   # or git, once published
```

What you bring: a model and a reason for a chain to care about its output
(a market resolution, an insurance trigger, a moderation decision, an agent's
on-chain action). What you get: that output, with the same finality guarantee
as any other optimistic system on Sui, and a dispute that costs a few dollars
instead of a TEE cluster or a ZK prover.

---

## What's in the box

```
opml/
  vm/          the fraud-proof VM: ISA, soft-float, the one-step verifier (Rust)
  kernels/     fast deterministic CPU kernels (NEON, persistent thread pool)
  gpu/         bit-exact Metal / wgpu GEMV
  compiler/    any quantized LLM (float activations) → VM program + genesis
  models/      toy model (for fast tests) + Qwen3-0.6B reference runtime
  game/        the dispute game: actors, bisection, a local mock chain, fuzzers
  client/      on-chain harness: stage and resolve a real dispute on devnet
  benches/     honest-path overhead and tok/s measurements
  move/        the Sui package: dispute, interp, merkle, softfloat, signed, tee
  docs/        SPEC (the full design), ANALYSIS (measurements), PRIOR_ART (survey)
```

Start with `docs/SPEC.md` for the design, `docs/ANALYSIS.md` for what's measured,
`docs/PRIOR_ART.md` for the field. Run the Rust tests with `cargo test`, the Move
tests with `sui move test` in `move/`.

---

## What this is not, yet

- The reference model is small (0.6B) because that's what fits a laptop demo.
  Nothing in the design is size-specific; bigger models are more compile time
  and more genesis pages, not a different protocol.
- Our own predictor is ~3× slower than llama.cpp. Closing that is kernel work,
  and the determinism-engine adoption path above is how it closes to ~native.
- Multi-challenger tournaments (sybil-resistant disputes) are future work; the
  design to copy is Cartesi's Dave / refereed tournaments.

## License

Apache-2.0. The Qwen3-0.6B weights are the Qwen team's (Apache-2.0); we vendor
the config and tokenizer and pin the weight hash (see `models/qwen/artifacts`).
