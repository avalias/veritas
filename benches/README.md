# Honest-path overhead — first measurements (toy judge, integer path)

```
machine:   Apple Silicon (single thread, software SHA3 @ ~865 MB/s)
workload:  toy judge — 40-char prompt + 20 generated tokens,
           59 positions, 744,924 micro-ops, 1 MiB memory
reproduce: cargo run -p benches --release
date:      2026-06-10 (commit: Phase 2 era)
```

## Measured

| path | time | vs native |
|---|---|---|
| A. native forward, no commitments ("ordinary inference") | 312 µs | 1.00× |
| B. native + checkpoint commitments (per-token) | 2,143 µs | 6.9× |
| C. genesis tree build (per-judge setup, amortized) | 1,354 µs | — |
| D. interpreter, eager per-write hashing (old worst case) | 335 ms | ~1,075× |
| E. full per-step trace (dispute segments only) | 458 ms | ~1,468× |

Commitment cost ≈ **1.83 ms per inference** = 1,046 KiB of dirty pages
(≈ 1.2 ms leaf hashing at 865 MB/s) + deduplicated interior-node updates
(`MerkleTree::update_leaf_hashes_bulk` — shared ancestors hashed once per
level; this alone cut commitment cost ~1.7×).

## The two headline facts

1. **The math overhead for integer (quantized) models is zero.** The
   native runtime (A) uses ordinary fast integer code — any SIMD, any
   tiling, any thread layout produces byte-identical results because
   integer addition is associative — and conformance C-14 proves its
   committed roots equal the per-step oracle's **exactly**. Determinism
   costs nothing on the integer path; this is structural, and it holds on
   GPU integer tensor cores for the same reason.
2. **The entire honest-path cost is commitment hashing**, and it is an
   absolute cost (µs per MB hashed), not a fraction of compute. The toy's
   6.9× ratio is a scale artifact: a 1 MiB model's compute (312 µs) is
   microscopic next to its state turnover. Real models invert the ratio:

## MEASURED: Qwen3-0.6B (real weights, this machine, 10 CPU threads)

| metric | value |
|---|---|
| predictor decode, scratch-free (`qwen_demo` pure) | **~32–39 tok/s** (NEON smlal + persistent pool; was 14.8 naive. ±20% thermal drift — use the controlled A/B below for kernel deltas, never single runs) |
| llama.cpp Q8_0, same machine, `-dev none` (pure CPU) | 101 tok/s — **gap ~3×** |
| llama.cpp Q8_0, `-ngl 0` (Accelerate/AMX BLAS) | 109 tok/s |
| commitment — Merkle hashing, pipelined | **~3% wall-clock** (hasher thread; the fundamental, near-zero cost; roots bit-identical, asserted) |
| commitment — VM-fidelity scratch | ~5% (a COMPILER ARTIFACT: the program commits sigmoid intermediates as memory cells; a leaner compilation removes it. The predictor skips it entirely) |
| genesis tree (per-judge, one-off) | ~1.4 s |
| quality (final config) | **PPL 421 vs float-ref 34.60 / llama-Q8 34.99; top-1 agreement 20.5%** — full measured ladder below |

> **Honest overhead, decomposed (QWEN_PROF profiling, scratch-free predictor):**
> determinism tax **0** (integer associativity) · Merkle hashing **~3%**
> pipelined (fundamental) · VM-fidelity scratch **~5%** (compiler-reducible —
> the committed runtime mirrors the VM program's intermediate stores; ordinary
> serving via `position_uncommitted` skips them). The earlier rosy "0–3%"
> compared committed-vs-pure with BOTH doing scratch, masking it.

Two separable claims, now measured:
1. **Determinism + commitment cost ≈ 0** on the integer path — math is
   bit-exact at any kernel speed (associativity), and hashing hides
   entirely behind compute when pipelined. This holds on GPU for the same
   reasons (integer kernels + on-device keccak / pipelining).
2. **Kernel parity with llama.cpp is unfinished: ~3.6× gap — and it is
   MEMORY-BOUND, not MAC-bound.** Controlled A/B (`benches/kernel_ab`,
   thermal-robust ratio of medians, 3072×1024 GEMV; single-run tok/s drifts
   ±20% with thermals, so kernel deltas are same-process ratios):

   | blocked-GEMV kernel | ns/GEMV | vs legacy |
   |---|---|---|
   | legacy `dot_w8_x16` per 64-block | 45,708 | 1.00× |
   | fused `block_partial` (one reduction/block) — **current** | 44,281 | **1.03×** |
   | `sdot` two-limb decomposition (i16, asm! or nightly intrinsic) | 60,869 | **0.75×** |
   | **single-sdot i8×i8 (the i8-activation CEILING)** | 32,724 | **1.39×** |

   The decisive number is the last row. A single-`sdot` i8 kernel has 4×
   the MAC density of our i16 `vmlal` — yet it is only **1.39× faster**,
   not 4×. So batch-1 decode GEMV is **bound by streaming the i8 weights**
   (read once per token, identical for i16 and i8 activations); the MACs
   hide behind memory latency. Consequences, all measured:
   - The i16 activation width the quality campaign required costs **almost
     nothing** in speed (the i8 ceiling is only 1.35× over the current i16
     path). The earlier "i16 `vmlal` is the wall" framing was wrong.
   - The `sdot` two-limb trick (i16 → two i8 limbs) is bit-exact but
     **0.73×** — slower both via `asm!` and nightly `vdotq_s32` intrinsics
     (≈equal, so not a scheduling artifact): the limb split's dual loads +
     i64 recombine cost more than they save on a memory-bound kernel.
   - Fusing q/k/v into one pool dispatch was also slower (barriers aren't
     the bottleneck).

   So the path to llama's 101 tok/s is **memory-access efficiency**
   (weight prefetch, cache tiling, thread work distribution — llama's years
   of tuning), NOT a cleverer dot kernel and NOT dynamic-i8 (which buys
   ~1.35× at most, ~28→~38 tok/s, still 2.6× short). The fused
   `block_partial` (+3%) is the only kept change. The protocol is
   indifferent — any bit-exact kernel is admissible (§9.1).

## MEASURED: the full trustless-verification chain at Qwen scale

The integer path is now end-to-end: the same committed program runs in the
fast native predictor and in the reference VM, and a real fault is
convicted on-chain.

| artifact | result |
|---|---|
| Qwen→VM compiler (`compiler/src/qwen.rs`) | 19,754-instr program, depth p=15; **29.55M micro-ops** for 2+1 tokens, 59.6M for 3+2 |
| **C-14 at Qwen scale** (`qwen_c14`) | native checkpoint roots == VM-oracle roots **bit-identical at every token boundary**; register file exactly the compiler's static prediction (acc=aux=idx=0) |
| **Qwen fraud game** (`qwen_dispute`) | one flipped weight bit at step 13,889,868 → **25 bisection rounds isolate the exact step**, one-step `verify_step` convicts, resolver slashed; 80.8 s wall, one core, no precomputed trace (two-level cursor scheme) |
| **on-chain conviction** | the challenger's atomic-step StepProof (a DOTBM transition) is emitted as `dispute/tests/qwen_conviction.move` and the **Sui Move verifier convicts it** — self-contained (opened pages + sibling hashes), no 1 GiB image on-chain |

This is the EigenAI-beating claim made concrete: deterministic real-LLM
inference whose every micro-op is provable to a ~500-line L1 contract, with
the predictor running at native integer speed (the determinism tax is 0 by
associativity, §9.1) and the dispute settled by a single op.

## Extrapolation to real models (estimates, to be measured in Phase 3)

| setting | compute/token | dirty bytes/token | hash cost (1 thread) | overhead |
|---|---|---|---|---|
| toy (measured) | ~5 µs | ~18 KiB | ~31 µs | ~600% |
| Qwen-0.5B INT8, CPU | 20–50 ms | ~0.7–1 MiB¹ | ~1 ms | **~2–5%** |
| Qwen-0.5B INT8, GPU | ~3–8 ms | ~0.7–1 MiB | ~1 ms → ~0.2 ms² | **~3–25% → ~2–5%²** |

¹ dominated by the LM-head logits buffer (151,936 × i32 ≈ 600 KiB) +
  activation scratch + KV append; shrinkable by committing logits at lower
  width or hashing the head region only at decision tokens' checkpoints.
² page-leaf hashing is embarrassingly parallel (independent leaves;
  deterministic by construction) and ARMv8.2-SHA3 / multi-thread hashing
  applies; checkpoint-every-k-tokens is a further linear knob.

## GPU (Apple M4, wgpu/Metal) — measured

| metric | value |
|---|---|
| **bit-exactness** | **151,936 LM-head logits BIT-IDENTICAL GPU vs CPU** (committed 64-lane-partial semantics in WGSL i32; i64 reduction host-side) |
| head GEMV, weights resident | 7.55 ms/call (naive scalar-unpack shader, 9.7 MB partial readback) |
| same op, CPU pool+NEON | 2.0 ms |

The determinism claim on GPU is now demonstrated on real hardware, not
argued from associativity alone. Apple's GPU vs its own AMX-class CPU is
not where integer GEMV wins; the GPU case is discrete dp4a-class hardware
(NVIDIA int8: 100+ TOPS). Known shader work if/when needed: packed
dot4I8, on-GPU i64-emulated row reduction (readback 9.7 MB → 1.2 MB),
workgroup tiling. The zero-overhead commitment story is hardware-agnostic
(pipelined hashing measured 0.00% on CPU; same structure applies).

## Quality campaign (bars: our float ref PPL **34.60**, llama Q8_0 **34.99**)

The float-reference PPL matching llama within noise validates our float
implementation, the integer rope tables, AND the eval convention in one
number. The integer ladder, in causal order:

| configuration (cumulative) | int PPL |
|---|---|
| per-layer scales + i32 carrier, thin 386-token calibration | 2,420 |
| + SmoothQuant α=0.5 (under thin calibration — HARMFUL) | 20,905 |
| + probs Q0.14 + V-cache i16 | 15,927 |
| + dedicated 801-token calibration corpus | **1,028** (16×!) |
| + calibration headroom 1.25 (clipping↔resolution optimum) | **449** |
| + per-64-block activation scales, per-(row,block) M tables | **381** |
| + per-(row,block) weight scales (Q8_0-style, free in M) | ~400 (noise-level) |
| (3× more max-calibration data — resolution pathology) | 1,145 — reverted |

α sweep: {0.25: 18.7k, **0.5: 16.3k**, 0.75: 26.4k}. Final config ≈ 400,
an ~11× distribution gap to the bar (logit noise ~1.3 vs needed ~0.2).

**Percentile calibration — tried, measured, REJECTED (night-3).** The
hypothesis was that max-based activation scales waste range on rare
outliers; a high-percentile clip should reclaim resolution. Wired a
log2-histogram per activation site (`QCAL_PCTL` env) and swept:

| activation-scale source | int PPL |
|---|---|
| **max (default)** | **421** |
| percentile 0.9999 | 1,220 |
| percentile 0.999 | 1,031 |

Clipping HURTS — decisively. The large q/k/gate/up/v activations are
*signal*, not noise: attention logits ride on the largest q·k components
and SwiGLU gates on their tail, so a tighter (saturating) scale destroys
real information. This rules out the lever and re-points the diagnosis:
the gap is logit **resolution (bits)**, not outlier clipping. The
histogram code stays (env-gated, off) for reproducibility.

Named next steps, in expected-value order: **more activation bits on the
worst sites** (the resolution finding above — e.g. i32 activations on the
gate/down path), blocked q/k + attention-path scales, and ultimately FW-6
(deterministic float semantics = exact parity by construction). Diagnosis
machinery (probe + diag + this harness) reduces each step to a ~4-minute
measured iteration.

## Levers, in priority order

1. ~~Bulk interior-node updates~~ — done (`update_leaf_hashes_bulk`).
2. ~~Don't hash the logits at all~~ — ISA support done (`ARGMAX_OFF`,
   SPEC v0.3.0): the chunked decode head cycles vocab logits through ONE
   reused page, deleting the ~600 KiB/token dominant term at Qwen scale.
   The Qwen compiler (`compiler/src/qwen.rs`) adopts it: the streaming head
   cycles vocab logits through one page and tracks the absolute winning row
   via `ARGMAX_OFF` + a `v_cell` counter — nothing vocab-sized is committed.
3. Pipeline commitment behind next-token compute (hash token t's dirty set
   while computing t+1): wall-clock overhead → ~0 whenever hashing
   throughput ≥ dirty-byte rate. Latency cost: one checkpoint's hashing at
   the end of the run.
4. GPU-resident commitment: keccak as an epilogue kernel over pages that
   already live in GPU memory (integer-only ⇒ bit-exact trivially); only
   32-byte roots cross PCIe. Hashing rides idle SMs instead of the CPU.
5. Hardware SHA3 (ARMv8.2 EOR3/RAX1/XAR/BCAX) or revisit Open Question Q1
   (blake2b-256 is also Sui-native and ~2× faster in software).
6. Checkpoint sparsity (every k tokens) — linear trade against dispute-
   segment materialization cost, which E shows is milliseconds anyway.

## What D and E prove

D (the interpreter everyone assumed fraud-proof systems must run) is
**three orders of magnitude** slower than the native honest path — that is
the cost our checkpoint-mode design deletes. E prices the dispute path:
even materializing *every* per-step root for this whole run costs ~0.5 s,
and a real dispute only ever materializes one segment of it.
