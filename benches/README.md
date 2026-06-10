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
| integer decode (`qwen_demo`) | **14–19 tok/s** (naive autovectorized kernels) |
| llama.cpp Q8_0, same machine, `-dev none` (pure CPU) | 101 tok/s |
| llama.cpp Q8_0, `-ngl 0` (Accelerate/AMX BLAS) | 109 tok/s |
| commitment, sequential | 1.5 ms/token = **2.3–2.8% of compute** (~105 dirty pages/token after the row-major V fix; was 7.8 MB/token before) |
| commitment, **pipelined** (hasher thread) | **0.00% wall-clock** (794 ms vs 846 ms pure; roots bit-identical, asserted) |
| genesis tree (per-judge, one-off) | ~1.4 s |
| quality | coherent English judgments; int/float token agreement 0 (W8A8-static; SmoothQuant-class equalization is the known next step) |

Two separable claims, now measured:
1. **Determinism + commitment cost ≈ 0** on the integer path — math is
   bit-exact at any kernel speed (associativity), and hashing hides
   entirely behind compute when pipelined. This holds on GPU for the same
   reasons (integer kernels + on-device keccak / pipelining).
2. **Kernel parity with llama.cpp is unfinished**: ~6× gap, fully
   accounted for by naive scalar dot loops + per-projection thread spawns
   vs years of NEON/AMX tuning. Known path: persistent thread pool, NEON
   `sdot`/`smlal` kernels, fused projection batches. The protocol is
   indifferent to this — any bit-exact kernel is admissible (§9.1).

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

## Quality campaign (integer vs llama.cpp Q8_0 bar: PPL 34.99 ± 5.11)

| configuration | int PPL | top-1 vs float |
|---|---|---|
| per-layer scales + i32 carrier (pre-SmoothQuant) | 2,420 | 15.8% |
| + SmoothQuant (α=0.5, norm-folded, tied-emb handled) | (running) | |
| + probs Q0.14 + V-cache i16 | (next) | |

Method: 512-token chunks, second half scored (llama-perplexity's exact
convention), calibration on the file tail (no leakage). The "coherent
text" of early decodes hid a catastrophic distribution gap — hence
measurement-first iteration.

## Levers, in priority order

1. ~~Bulk interior-node updates~~ — done (`update_leaf_hashes_bulk`).
2. ~~Don't hash the logits at all~~ — ISA support done (`ARGMAX_OFF`,
   SPEC v0.3.0): the chunked decode head cycles vocab logits through ONE
   reused page, deleting the ~600 KiB/token dominant term at Qwen scale.
   The Phase 3 compiler adopts it; expected per-token dirty set drops to
   ~100–200 KiB (KV append + activation scratch).
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
