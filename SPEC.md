# SPEC.md — Deterministic Tensor VM + On-Chain One-Step Fraud Verification

```
Version:  0.2.0-draft   (Phase 0 deliverable — review gate before any code)
Changed:  0.2.0 — DOT8/DOT16 line-dot micro-ops (§5.2), lazy level-0 hashing
          (§7.4), bit-exact parallelism rules (§9.1), performance budget
          (§1.4). Honest-path trace generation: ~2 h → ~1–3 min single-core.
          0.2.1 — prior-art survey (PRIOR_ART.md): validates architecture,
          adds citations to §8.6/§9.2 and future-work rows FW-2/FW-5.
          0.3.0 — ARGMAX_OFF (0x16): streaming/chunked decode head — the
          vocab logits buffer (the dominant per-token commitment term at
          Qwen scale, ~600 KiB/token) never enters committed state.
          Opcode numbering append-only; all prior goldens unchanged.
          0.4.0 — wide-activation ops the measured Qwen quality work
          forced: LD16 (0x17), DOT8X16 (0x18: i8-weight × i16-activation
          line dot), DOTBM (0x19: fused dot-line × per-block multiplier
          accumulate — the per-(row,block) quantization structure at one
          op per block). W-slot is a READ for DOTBM (documented asymmetry).
Scope:    VM state, commitment scheme, ISA, numeric formats, traces,
          dispute protocol, determinism rules, conformance tests.
```

This document is the single source of truth for the byte-level semantics of the
system. The Rust reference runtime, the MockChain, and the Sui Move verifier
are all *implementations of this spec*; where they disagree with it, they are
wrong. Every constant in this document states its rationale, because every
constant here becomes a consensus rule.

Keywords **MUST**, **MUST NOT**, **SHOULD**, **MAY** are used RFC-2119-style.
Sections marked *(informative)* explain intent and do not bind implementations.

---

## 0. Deviations from the project brief *(informative — read first)*

These are deliberate design changes from the prompt, called out for review:

1. **Commitment hash is SHA3-256, not blake3.** The on-chain verifier must
   recompute Merkle nodes and state roots. Sui Move's native hash set is
   `{sha2_256, sha3_256, keccak256, blake2b256}` — blake3 is **not** native,
   and implementing blake3 in Move would alone blow the "few hundred lines"
   verifier budget (violating Invariant 4). SHA3-256 is native in the Move
   stdlib *and* ubiquitous in Rust. blake3 is kept where the brief wanted it
   for **artifact identity** (weights/tokenizer/template/input files), which
   is never recomputed on-chain. Alternative: `blake2b256` (also Sui-native,
   ~3–5× faster off-chain) — one-line switch, flagged as Open Question Q1.
2. **ISA is larger than the rough list** (21 ops vs ~10). Additions and why:
   `DIV32` (softmax normalization needs one integer division per attention
   row — the I-BERT approach; a reciprocal LUT would need range reduction and
   *more* on-chain code), `MAC16`/`CLAMP16` (Q/K activations are kept in i16
   so rotary embeddings and attention logits don't degrade to ~6-bit trig),
   `LD8`/`LD32`/`LDC`/`ST32` (move data through the accumulator), `LDIDX`
   (runtime token id → embedding row address), `JEQ` (the brief's `CMP`,
   fused with its only consumer, the EOS-exit branch), `JMP`. Each op is
   5–15 lines in Move; the interpreter stays well inside budget (§1.3).
3. **`HALT` carries no output payload.** Output is a designated memory region
   bound to the final state root via a cheap Merkle challenge (§8.5). This
   keeps every micro-op's on-chain footprint uniform.
4. **Genesis handling:** the static part of the initial state (weights, LUTs,
   template) is a per-judge audited constant `static_genesis_root`; the
   per-question input pages are inserted **on-chain** at FactSpec creation
   with Merkle update proofs (§7.2), so the disputable trace starts from an
   on-chain-derived root. Verifying the static root itself on-chain (a "hash
   game" over the weights blob) is explicitly future work.
5. **Q/K path is i16, V/residual path is i8** (§6.2). Pure-i8 rotary needs
   Q1.6 sin/cos which materially corrupts positional information; i16 costs
   nothing in protocol complexity (MAC16 ≅ MAC8).
6. **Attention probabilities are i8 Q0.7** after exp/divide — a documented
   precision sacrifice (max representable ≈ 0.992) consistent with standard
   INT8 attention.
7. **`DOT8`/`DOT16` line-dot ops** (v0.2.0): the hot micro-op processes one
   64-byte cache line (≤ 64 i8 or ≤ 32 i16 lanes) per step instead of one
   element. Calldata per disputed step is unchanged — the opening unit is
   the 1 KiB page either way — and the Move interpreter gains one bounded
   loop (~15 lines), while off-chain trace generation and bisection depth
   improve ~64×. Wrapping-i64 addition is associative, so lane order inside
   a DOT cannot affect the result: SIMD implementations are bit-exact by
   algebra, not by luck (§9.1).

---

## 1. System overview *(informative)*

### 1.1 Actors and lifecycle

```
 Resolver                    Chain (Sui / MockChain)              Challenger
    │  assert FactSpec:           │                                   │
    │  (judge_id, input,          │                                   │
    │   N, root_N, output) ──────▶│  bond escrowed                    │
    │                             │  challenge window opens           │
    │                             │◀────────────── challenge + bond   │
    │◀──── bisection over steps [0, N]: resolver posts midpoint roots,│
    │      challenger agrees/disagrees, interval halves ─────────────▶│
    │                             │                                   │
    │   when hi−lo == 1: either party submits verify_step             │
    │   (registers, Merkle openings, ONE micro-op executed in Move)   │
    │                             │                                   │
    │                     loser slashed, winner paid,                 │
    │                     FactSpec resolved or rejected               │
```

The judged function is `f(weights_hash, prompt_hash, input_hash) → output`,
nothing else (Invariant 3). All three hashes resolve to byte arrays that are
mapped into the VM's initial memory by a committed layout (§7.2); from there,
execution is a pure function of the initial state.

### 1.2 Why each piece exists

- **Bit-exact integer VM** — so two honest parties always compute *identical*
  state roots, and any divergence isolates to exactly one micro-op.
- **Merkle-committed state** — so a single micro-op's claim can be checked
  on-chain with O(log n) data instead of the full gigabyte state.
- **Bisection** — so the chain never executes more than ONE micro-op, no
  matter how long the inference run is (~5×10⁹ steps for a 500-token Qwen
  judgment, §1.4).

### 1.3 On-chain budget (Invariant 4)

The Move verifier consists of: SHA3-256 calls (native), Merkle fold (~40
lines), signed-integer helpers over u64 two's-complement (~80 lines, §9.3),
the 21-op interpreter (~330 lines), and the dispute state machine (~300
lines). Target: **≤ 800 lines total, ≤ 450 for the one-step core.** Any spec
change that grows the on-chain core beyond "hashing + integer arithmetic +
Merkle paths" MUST be rejected or redesigned.

### 1.4 Performance budget *(informative)*

For a ~500-token Qwen-0.5B judgment: ~2.5×10¹¹ MACs → ~5×10⁹ micro-ops
with DOT8/DOT16 (GEMMs dominate; per-element ops are ~100× fewer).

| Path | Cost | Why |
|---|---|---|
| Answer only (Phase 3 predictor runtime) | seconds | native int8 GEMM; never feeds traces (§9.2) |
| Reference trace, checkpoint mode, 1 core | ~1–3 min | DOT amortizes dispatch ~64×; floor is weight-streaming bandwidth (~15–30 s) |
| Reference trace, checkpoint mode, row-parallel | ~20–40 s | §9.1 algebra-licensed parallelism |
| Checkpoint hashing (~12k ckpts, dirty pages only) | ~10–15 s | incremental Merkle (§3.4), overlappable with compute |
| Level-0 materialization (disputes only) | seconds | ~10⁵–10⁶ steps per disputed segment (§7.4) |
| Toy model (Phase 1), full per-step trace | < 1 min | ~10⁵–10⁶ steps total |

Both parties pay the trace cost — a challenger must run the reference to
know where to disagree — so every lever above helps both symmetrically.

**Phase 3 performance targets** (measured against llama.cpp Q8 on the same
hardware; these are product goals, not consensus rules):

| Target | Bar | Lever |
|---|---|---|
| Predictor decode throughput | ≥ 0.9× llama.cpp Q8 tok/s | native int8 kernels implementing the integer semantics |
| Assertion-ready trace | ≤ 2× predictor wall-clock | checkpoint mode + §9.1 parallelism, fully off the answer path |
| Hashing share of trace time | ≤ 10% | per-token checkpoint schedule, pipelined hashing, HW SHA3 where present |

The floor is *visibility*, not zero: commitment hashing and determinism are
irreducible, but they can be made product-invisible (async assertion,
bounded multiple of native time).

**First measurements (toy, Phase 2 era — see benches/README.md):** native
checkpoint-mode runtime reproduces oracle roots exactly (C-14) at 312 µs vs
the eager interpreter's 335 ms (~1,075×); the whole honest-path cost is
1.83 ms of commitment hashing per inference (1 MiB dirty pages, 1 thread,
software SHA3). For integer models the determinism tax on math is 0% by
associativity — the §1.4 targets are about hashing engineering only.

---

## 2. Conventions

### 2.1 Byte order and integers

- **All multi-byte values are little-endian** — in VM memory, in register
  encodings, in hash preimages, in instruction encodings. No exceptions.
- `uN`/`iN` denote N-bit unsigned / two's-complement signed integers.
- Signed values are *specified by their two's-complement u64 bit pattern*.
  Rust implements them as native `i64`; Move emulates them on `u64` (§9.3).
  The canonical encoding of `i64` is the LE bytes of its bit pattern.
- `+w`, `−w`, `·w` denote wrapping (mod 2⁶⁴) arithmetic. Where this spec says
  an operation wraps, that is normative semantics (the verifier is total even
  on adversarial states); where honest programs cannot overflow, the compiler
  guarantees it and §6 documents the bound.

### 2.2 Hash functions and domain tags

| Purpose | Function | Where computed |
|---|---|---|
| State commitments, Merkle trees, trace/schedule commitments | **SHA3-256** (`H`) | Rust + Move (`std::hash::sha3_256`, native) |
| Artifact identity: weights blob, tokenizer files, prompt template, input text | **blake3** | off-chain only — never recomputed on-chain |

Every `H` preimage starts with a 1-byte domain tag (second-preimage guard:
a leaf can never be reinterpreted as an interior node, nor a page as an
instruction):

| Tag | Preimage | Meaning |
|---|---|---|
| `0x00` | `0x00 ‖ page[1024]` | memory page leaf (1025 B) |
| `0x01` | `0x01 ‖ left[32] ‖ right[32]` | interior node, **all** trees (65 B) |
| `0x02` | `0x02 ‖ mem_root[32] ‖ regs[45]` | state root (78 B) |
| `0x03` | `0x03 ‖ instr[96]` | program leaf (97 B) |
| `0x04` | `0x04 ‖ root[32]` | trace-commitment leaf (Phase 4) |
| `0x05` | `0x05 ‖ LE64(step)` | checkpoint-schedule leaf (Phase 3) |
| `0x06` | `0x06 ‖ judge-identity fields (§7.2)` | `judge_id` (off-chain reference hash) |

Distinct trees (memory, program, schedule, trace) are verified against
distinct stored roots, so cross-tree proof replay is additionally prevented
by construction, not only by tags.

---

## 3. Machine state

### 3.1 Memory

- Flat byte-addressable array of `MEM_BYTES = 2^d · 1024` bytes, i.e. `2^d`
  pages of **`PAGE = 1024` bytes**.
  - *Why 1024:* balances proof size (page is sent as calldata in openings;
    1 KiB + d·32 B sibling path ≈ 1.7 KiB per opening at d = 20 — far below
    Sui's pure-arg and tx-size limits) against tree depth (d stays ≤ 24).
    `1024 % 8 == 0`, so a naturally aligned access never straddles a page.
  - `d` is a per-program constant from the header, `10 ≤ d ≤ 24`
    (1 MiB … 16 GiB; 24 caps sibling paths at 768 B).
- **Alignment rule:** every memory access of size k ∈ {1, 2, 4} bytes MUST
  satisfy `ea % k == 0`. Violations trap (§4.4). Consequence: **no access
  ever crosses a page boundary**, so every operand needs exactly one page
  opening.
- Uninitialized memory is zero-filled.

### 3.2 Register file

Canonical encoding: 45 bytes, fields in this order, LE.

| Offset | Size | Register | Type | Reset | Meaning |
|---|---|---|---|---|---|
| 0 | 4 | `pc` | u32 | 0 | index into program (instruction units, not bytes) |
| 4 | 1 | `halted` | u8 | 0 | 0 = running, 1 = halted, 2 = trapped |
| 5 | 8 | `step` | u64 | 0 | micro-ops executed so far — the bisection coordinate |
| 13 | 8 | `acc` | i64 | 0 | wide accumulator (i64 so a ≤ 2¹⁵-element i16·i16 dot product cannot overflow: 2¹⁵·2¹⁵·2¹⁵ = 2⁴⁵ ≪ 2⁶³) |
| 21 | 8 | `aux` | i64 | 0 | argmax index carrier |
| 29 | 16 | `idx[0..4]` | u32 ×4 | 0 | loop counters / address indices |

(Offsets are an *encoding*, not memory — odd alignment is irrelevant.)

### 3.3 State root

```
state_root = H( 0x02 ‖ mem_root ‖ regs_45_bytes )
```

The register file is committed in full inside the preimage (it is only 45
bytes), so a one-step proof reveals registers directly — no Merkle openings
for register reads/writes.

### 3.4 Memory Merkle tree

- Full binary tree of depth `d` over the `2^d` pages.
- `leaf(i)   = H(0x00 ‖ page_i)`            — page index NOT in the preimage;
  position is bound by the fold order below (a proof for index i recombines
  in an order fully determined by i's bits, so content cannot be relocated
  without a collision).
- `node      = H(0x01 ‖ left ‖ right)`
- **Zero subtrees:** `Z_0 = H(0x00 ‖ 0^1024)`, `Z_{l+1} = H(0x01 ‖ Z_l ‖ Z_l)`.
  Implementations precompute `Z_0..Z_d` (enables sparse off-chain storage and
  cheap genesis construction).
- **Proof format:** for page index i, the `d` sibling hashes bottom-up.
  Verification folds LSB-first: at level l, if bit l of i is 0 the running
  hash is the left child:

  ```
  cur = H(0x00 ‖ page)
  for l in 0..d:
      cur = bit_l(i) == 0 ? H(0x01 ‖ cur ‖ sib[l]) : H(0x01 ‖ sib[l] ‖ cur)
  assert cur == mem_root
  ```
- **Incremental update (off-chain):** the runtime keeps all interior nodes;
  a micro-op writes at most one page (§5), so a step re-hashes one 1 KiB leaf
  + `d` interior nodes + the 78-byte state preimage — O(log n), satisfying
  the Phase 0 requirement. Per-step re-hashing only happens when level-0
  roots are being materialized (inside a disputed segment, §7.4); the honest
  path tracks dirty pages and hashes only at checkpoints.
- **Update during verification (on-chain):** the write-page's pre-inclusion
  is verified with siblings `sib[]`; the post-root is computed by re-folding
  the *modified* page with the *same* siblings (read-only pages don't move).
- **Checkpoint flush (off-chain, normative-result/free-strategy):** when k
  dirty pages flush at once, shared ancestors MUST NOT be re-hashed per
  page — batched per-level updates cost ~k + k/2 + … node hashes instead
  of k·d (measured ~1.7× on commitment cost). Page-leaf hashing has no
  ordering dependencies (independent leaves) and MAY be parallelized;
  tree-write iteration order stays deterministic (`BTreeSet`, Invariant 2).
  Commitment hashing is not execution — §9.1's two-transform limit does
  not constrain it.

### 3.5 Program tree

Same node rule, leaf rule `H(0x03 ‖ instr_96)`. Depth `p` (header constant,
`p ≤ 32` since `pc` is u32). Slots beyond the last real instruction are
zero-filled (opcode `0x00` = invalid → trap, §4.4), so fetches into padding
are well-defined. `program_root` is part of the judge identity. Program
length is deliberately NOT an on-chain parameter — `p` alone suffices.

---

## 4. Program format

### 4.1 Instruction encoding — fixed 96 bytes

| Offset | Size | Field | Notes |
|---|---|---|---|
| 0 | 1 | `opcode` | §5 table |
| 1 | 1 | `k` | index-register selector (0–3) or store-source selector — per-op |
| 2 | 1 | `s` | shift amount for `SHIFT_RNDN` (0–63) |
| 3 | 1 | reserved | compiler MUST write 0; verifier ignores |
| 4 | 4 | `imm` | u32 or i32, per-op |
| 8 | 4 | `target` | u32 program index for `JMP`/`JEQ`/`LOOP` |
| 12 | 4 | reserved | as above |
| 16 | 24 | `opA` | read operand descriptor |
| 40 | 24 | `opB` | read operand descriptor |
| 64 | 24 | `opW` | write operand descriptor |
| 88 | 8 | reserved | as above |

Operand descriptor (24 bytes): `base: u64`, then `stride[0..4]: u32 ×4`.

Fields not used by an opcode are ignored by the verifier (semantics never
read them) but MUST be zeroed by the compiler so program bytes are canonical.

*Why fixed 96 B:* uniform leaves make the program tree and the on-chain
decoder trivial; with loops, real program sizes are ~10⁵–10⁶ instructions
(≈ 100 MB worst case for 500-token Qwen — held off-chain; only one leaf +
path ever goes on-chain).

### 4.2 Effective addresses

For operand X of the current instruction:

```
ea(X) = base  +w  Σ_{j=0..3} ( u64(idx[j]) ·w u64(stride[j]) )      (mod 2^64)
```

All arithmetic wraps; the bounds check happens after (§4.4). Exception:
`LUT16` ignores `opA.stride[]` and computes its address from `acc` (§5).

### 4.3 Program header (manifest, off-chain artifact)

Normative fields (binary, LE, fixed order; committed inside `judge_id`):

```
vm_version  u32      = 1
d           u8       memory tree depth
p           u8       program tree depth
out_base    u64      output region start (page-aligned)
out_len     u32      output region size, ≤ 4096 bytes (one logical answer)
n_regions   u32
region[i]: base_page u32, n_pages u32, source u8 (0=ZERO, 1=ARTIFACT, 2=INPUT),
           blake3[32] (zero if not ARTIFACT), artifact_offset u64
schedule_root  [32]  Phase 3 checkpoint schedule (zero in Phase 1)
```

Exactly one region MUST have `source = INPUT`, at most 16 pages (16 KiB ≈
4096 tokens — Open Question Q4).

### 4.4 Traps

A step **traps** when any of the following holds. Trap transition: `halted ← 2`,
`step ← step + 1`, every other register and all memory unchanged.

| # | Condition |
|---|---|
| T1 | `pc ≥ 2^p` at fetch (no instruction leaf exists — verifier handles without an opening) |
| T2 | `opcode` not in the §5 table (includes `0x00` padding) |
| T3 | any computed `ea` with `ea + size > MEM_BYTES` or `ea % size ≠ 0` |
| T4 | `SHIFT_RNDN` with `s > 63` |
| T5 | `DIV32` with divisor ≤ 0 |
| T6 | `k` out of range for the op (`> 3` for idx ops, `> 5` for `ST32`) |
| T7 | `DOT8`/`DOT16` with `imm` lanes = 0 or > cap (64 / 32), or an operand line with `ea % 64 ≠ 0` or `ea + 64 > MEM_BYTES` |

**Terminality:** states with `halted ≠ 0` have **no successor**. A one-step
claim whose (revealed) pre-state has `halted ≠ 0` is fraud by rule (§8.4
step V3). Honest programs never trap; traps exist so the verifier is *total*
— every adversarially claimed state still has exactly one defined transition
or is terminal.

---

## 5. ISA semantics

### 5.1 Helper functions (normative)

```
sext8(b)  / sext16(h) / sext32(w)  : sign-extend to i64
low32(x)                           : truncate i64 → u32 (bit pattern)
sat8(x)   = clamp(x, −128, 127)            — saturation, never wraparound
sat16(x)  = clamp(x, −32768, 32767)
m8[a] / m16[a] / m32[a]            : LE read of 1/2/4 bytes at a (after T3 check)

trunc_div(a: i64, dv: i64)         : dv > 0 guaranteed (T5)
  q = |a| / dv      (u64 magnitude division; |i64::MIN| = 2^63 is exact in u64)
  return a < 0 ? −q : q            — truncation toward zero
  (i64::MIN / 1 = i64::MIN — wraps correctly through the magnitude path)

rnd(x: i64, s: u8)                 : arithmetic shift right, ROUND-HALF-TO-EVEN
  if s == 0: return x
  q    = x >> s                    (arithmetic, toward −∞)
  r    = x −w (q << s)             — in [0, 2^s); q·2^s ∈ (x − 2^s, x] so no overflow
  half = 1 << (s−1)
  if r > half:  q = q + 1          — cannot overflow: |q| ≤ 2^62 for s ≥ 1
  if r == half: q = q + (q & 1)    — to even
  return q
```

**`rnd` boundary vectors** (these become mandatory unit tests, §11):

| x | s | exact | result | why |
|---|---|---|---|---|
| 3 | 1 | 1.5 | 2 | half → even (2) |
| 5 | 1 | 2.5 | 2 | half → even (2) |
| −3 | 1 | −1.5 | −2 | half → even (−2) |
| −5 | 1 | −2.5 | −2 | half → even (−2) |
| 7 | 2 | 1.75 | 2 | above half rounds up |
| −7 | 2 | −1.75 | −2 | symmetric |
| −1 | 63 | ≈ −0 | 0 | q = −1, r = 2⁶³−1 > half ⇒ rounds up to 0 |

This is the **single rounding rule** of the entire system: requantization,
rotary, LUT input scaling — everything routes through `SHIFT_RNDN`, i.e.
through `rnd`.

### 5.2 Opcode table

Every executed step does `step ← step + 1`; `pc ← pc + 1` unless stated.
"Reads/Writes" are memory accesses (= required Merkle openings).

| Op | Code | Reads | Writes | Semantics |
|---|---|---|---|---|
| `MAC8` | 0x01 | A:1, B:1 | — | `acc ← acc +w (sext8(m8[eaA]) ·w sext8(m8[eaB]))` |
| `MAC16` | 0x02 | A:2, B:2 | — | `acc ← acc +w (sext16(m16[eaA]) ·w sext16(m16[eaB]))` |
| `LD8` | 0x03 | A:1 | — | `acc ← sext8(m8[eaA])` |
| `LD32` | 0x04 | A:4 | — | `acc ← sext32(m32[eaA])` |
| `LDC` | 0x05 | — | — | `acc ← sext32(imm)` |
| `ADD32` | 0x06 | A:4 | — | `acc ← acc +w sext32(m32[eaA])` |
| `MUL32` | 0x07 | A:4 | — | `acc ← acc ·w sext32(m32[eaA])` |
| `DIV32` | 0x08 | A:4 | — | `dv = sext32(m32[eaA])`; trap T5 if dv ≤ 0; `acc ← trunc_div(acc, dv)` |
| `SHIFT_RNDN` | 0x09 | — | — | trap T4 if s > 63; `acc ← rnd(acc, s)` |
| `CLAMP8` | 0x0A | — | W:1 | `m8[eaW] ← sat8(acc)` (acc unchanged) |
| `CLAMP16` | 0x0B | — | W:2 | `m16[eaW] ← sat16(acc)` |
| `LUT16` | 0x0C | A:2 | — | `t = sat16(acc)`; `ea = opA.base +w 2·(t + 32768)`; `acc ← sext16(m16[ea])` — strides ignored; tables stored from most-negative input |
| `ST32` | 0x0D | — | W:4 | src by `k`: 0 → `low32(acc)`, 1 → `low32(aux)`, 2–5 → `idx[k−2]`; trap T6 if k > 5; `m32[eaW] ← src` |
| `LDIDX` | 0x0E | A:4 | — | trap T6 if k > 3; `idx[k] ← m32[eaA]` (as u32) |
| `ARGMAX_STEP` | 0x0F | A:4 | — | trap T6 if k > 3; `v = sext32(m32[eaA])`; if `v >s acc` { `acc ← v`; `aux ← u64(idx[k])` } — strictly greater ⇒ first maximum wins ⇒ **ties break to lowest index** under the compiler's mandatory ascending scan |
| `JMP` | 0x10 | — | — | `pc ← target` |
| `JEQ` | 0x11 | A:4 | — | if `m32[eaA] == imm` (u32 compare) `pc ← target` else `pc ← pc+1` |
| `LOOP` | 0x12 | — | — | trap T6 if k > 3; `nxt = idx[k] +w32 1`; if `nxt <u imm` { `idx[k] ← nxt`; `pc ← target` } else { `idx[k] ← 0`; `pc ← pc+1` } — bottom-tested: body executes `imm` times for imm ≥ 1 (idx values 0..imm−1), auto-resets for clean nesting; compiler MUST NOT emit imm = 0 expecting zero iterations (body has already run once) |
| `HALT` | 0x13 | — | — | `halted ← 1`; pc unchanged |
| `DOT8` | 0x14 | A: 64-B line | — | trap T7 unless `1 ≤ imm ≤ 64` and both lines 64-aligned and in-bounds; for j = 0..imm−1: `acc ← acc +w (sext8(m8[eaA+j]) ·w sext8(m8[eaB+j]))` — bounds/alignment checked on the full 64-byte line even when imm < 64 |
| | | B: 64-B line | | |
| `DOT16` | 0x15 | A,B: 64-B lines | — | as `DOT8` with `1 ≤ imm ≤ 32`, lanes `m16[eaA+2j] · m16[eaB+2j]` |
| `LD16` | 0x17 | A:2 | — | `acc ← sext16(m16[eaA])` |
| `DOT8X16` | 0x18 | A: 64-B line (i8) B: 128-B line (i16) | — | trap T7 unless `1 ≤ imm ≤ 64`, `eaA % 64 == 0`, `eaB % 128 == 0`, lines in bounds (128 \| 1024 ⇒ no straddle); for j < imm: `acc ← acc +w sext8(m8[eaA+j]) ·w sext16(m16[eaB+2j])` |
| `DOTBM` | 0x19 | A: 64-B i8 line, B: 128-B i16 line, W: 4 (READ!) | — | as `DOT8X16` but the line dot accumulates into a FRESH partial p, then `acc ← acc +w p ·w sext32(m32[eaW])` — one op per quantization block (per-(row,block) multiplier tables, the measured-quality structure). The W slot is a READ here — the one ISA asymmetry, chosen over a 4th operand |
| `ARGMAX_OFF` | 0x16 | A:4 | — | trap T6 if k > 3; `v = sext32(m32[eaA])`; if `v >s acc` { `acc ← v`; `aux ← u64(imm) +w u64(idx[k])` } — `ARGMAX_STEP` with an immediate index offset, so a chunk-local scan records a GLOBAL row index. Enables the chunked decode head: per 256-row chunk, dots fill ONE reused buffer page, then a scan with `imm = chunk·256` carries the running max through a memory cell across chunks (`LD32 saved_max … scan … ST32 saved_max`). The vocab logits array never exists in committed state — at Qwen scale this deletes ~600 KiB/token of commitment hashing (the dominant term, benches/README.md) |

All other opcodes (including `0x00`): trap T2. Opcode numbering is
**append-only** across spec versions (golden vectors must never be
invalidated by renumbering).

Notes:

- **At most 2 reads and 1 write per op, never both B-read and a write.**
  Worst-case openings per step: 2 (the MACs and DOTs).
- **Why the 64-byte DOT line:** it is one cache line; it divides every
  common LLM dimension (64 | 896, 64 | 4864, d_head = 64) so rows need no
  padding; 64 | 1024 guarantees no page straddle; and it amortizes
  interpreter dispatch ~64×, which is what makes the reference trace
  producer fast (§1.4). The normative lane order is sequential, but
  wrapping-add associativity makes any evaluation order bit-identical, so
  the same semantics autovectorize safely (§9.1). Register-only ops need zero.
- Jump targets are NOT validated at execution; an out-of-range `pc` traps at
  the *next* fetch (T1/T2). This keeps the jump ops trivial.
- If two operands land on the same page, their openings are simply both
  verified against the same `mem_root`; collision resistance forces the page
  bytes to agree, so no dedup rule is needed.
- Saturation vs wraparound, pinned once: **requantization stores saturate**
  (`CLAMP8`/`CLAMP16`/`sat16` in `LUT16`); **accumulator arithmetic wraps**
  (and the compiler proves honest programs never reach the wrap, §6.4).

---

## 6. Numeric formats, LUTs, lowering recipes

### 6.1 Requantization (normative pattern)

All scale changes use one shape — gemmlowp-style fixed-point multiply:

```
y = sat( rnd( x ·w M, s ) )        M: i32, 1 ≤ M ≤ 2^31−1;  s: 0..63
```

emitted as `MUL32(M-cell) ; SHIFT_RNDN(s) ; CLAMP8|CLAMP16(dst)`. `(M, s)`
pairs come from the quantizer manifest. Bound for no-wrap: |x| ≤ 2³¹ before
`MUL32` (compiler-checked per §6.4).

### 6.2 Data formats

| Tensor | Format | Why |
|---|---|---|
| Residual-stream activations, V, MLP activations | i8, per-tensor scale | standard INT8 |
| Weights | i8, per-output-channel scales | per-channel keeps GEMM accuracy |
| Biases | i32 (pre-scaled to the accumulator's scale) | loaded via `LD32` |
| Q, K (post-projection, post-rotary) | **i16**, per-tensor scale | preserves rotary + logit precision (Deviation 5) |
| Attention logits scale-in to exp | Q4.11 i16 (range ±16) | exp LUT domain |
| exp output | Q1.14 i16, in [0, 16384] | 1.0 = 2¹⁴ exactly |
| Attention probs | i8 **Q0.7** ∈ [0, 127] | prob 1.0 saturates to 127 ≈ 0.992 — accepted artifact (only exact-1.0 rows, e.g. attention over a single key) |
| RMSNorm rsqrt output | Q2.14 i16 | rsqrt(ms ≥ 1) ≤ 1.0 |
| sin/cos tables | Q1.14 i16, **both `sin` and `−sin` stored** | no NEG op; negation is baked into a second table |
| Logits | i32 (raw accumulator scale) | argmax is scale-invariant |
| Token ids | u32 | `ST32`/`LDIDX`/`JEQ` operate on u32 cells |

### 6.3 LUT tables (normative build rule)

Each LUT is 65 536 × i16 = 128 KiB, indexed by `t + 32768` for input
t ∈ [−32768, 32767], generated **offline** by the build tooling, then frozen
as an artifact (blake3-hashed, mapped into genesis memory like weights).

> **Float policy:** high-precision floating point (or MPFR) MAY be used to
> *generate* table entries and quantizer scales at build time, because the
> resulting bytes are committed artifacts with golden hashes. Floats MUST
> NOT appear anywhere in the runtime execution path (Invariant 1). This is
> the only float exception in the system.

| Table | Input interp. | Entry formula (then `sat16`) | Domain edges |
|---|---|---|---|
| `EXP` | Q4.11 (x = t/2¹¹) | `round_half_even(e^x · 2^14)` | x > 0 saturates toward 32767; compiler only feeds x ≤ 0 (post max-subtraction); x ≤ −16 → 0 |
| `RSQRT` | integer ms = t | t ≤ 0 → 32767 (documented sentinel); else `rhe(2^14/√t)` | i8 mean-square ∈ [0, 16384] fits domain directly — no range reduction |
| `SILU` | Q4.11 | `rhe(x·σ(x) · 2^11)` (out Q4.11) | saturating edges |

Out-of-domain inputs cannot occur at the *index* level: `LUT16` saturates
`acc` to i16 first, so the table edge entries ARE the clamp semantics —
deterministic by construction.

### 6.4 Lowering recipes *(informative — the compiler's contract)*

The compiler MUST document, per emitted program, the worst-case |acc| at
every `MUL32` to prove no i64 wrap. Standing bounds: i8·i8 dot products of
length ≤ 2¹⁵ stay under 2²⁹; i16·i16 under 2⁴⁵; requant multiply adds ≤ 31
bits — all ≪ 2⁶³.

- **GEMM row:** `LD32(bias); { DOT8(w_line, x_line) }×(d/64); MUL32(M);
  SHIFT_RNDN(s); CLAMP8(y)` — addresses via `idx` loops (`LOOP` over
  rows/cols), strides in operand descriptors. `MAC8` remains for scalar
  odds and ends; dimensions not divisible by 64 are padded by layout
  (zeros contribute exactly 0 — integer math, no error).
- **RMSNorm:** `LDC(0); {DOT8(x_line, x_line)}×(d/64); DIV32(const d);
  LUT16(RSQRT); ST32(r32)` (A = B aliasing the same line is legal)
  then per element `LD8(x); MUL32(r32); MUL32(γ_c); SHIFT_RNDN; CLAMP8`.
- **Softmax row (length L):**
  1. max: `LDC(I32_MIN); {ARGMAX_STEP}×L` → acc = m
  2. negate via multiply (no NEG op): `MUL32(const −1); ST32(neg_m)`
  3. per logit: `LD32(z); ADD32(neg_m); MUL32(M_z); SHIFT_RNDN(s_z)` → Q4.11;
     `LUT16(EXP); CLAMP16(e_i)`; sum: `ADD32` chain → `ST32(sum)`
     (L ≤ 4096 ⇒ sum ≤ 4096·2¹⁴ = 2²⁶, no overflow)
  4. per logit: `LD32(e_i); MUL32(const 2^14); DIV32(sum); SHIFT_RNDN(7); CLAMP8(p_i)` → Q0.7
- **Rotary (on i16 q/k):** per pair `(x, y)`:
  `LDC(0); MAC16(x, cos_t); MAC16(y, negsin_t); SHIFT_RNDN(14); CLAMP16(x')`
  and `LDC(0); MAC16(x, sin_t); MAC16(y, cos_t); SHIFT_RNDN(14); CLAMP16(y')`.
- **Attention:** logits `q·k` via `DOT16 ×(d_head/32)` (d_head ≤ 256 ⇒
  |Σ| < 2⁴⁰ ✓ — two steps per logit at d_head = 64); probs as above;
  context `Σ p_i·v` via `DOT8(p_line, v_line)` over L padded to a multiple
  of 64 with zero probs (exact); requant; store. V is laid out
  column-contiguous so prob/value lines are both contiguous.
- **Decode step:** logits → `ARGMAX_STEP` scan (ascending j via `LOOP`,
  k = that loop's register) → `ST32(aux → token_cell)`; embedding row fetch:
  `LDIDX(row ← token_cell)` then `LD8/CLAMP8` copy with `stride = d_model`;
  EOS: `JEQ(token_cell, eos_id, target=tail)`. Greedy only; ties → lowest id
  by `ARGMAX_STEP` semantics.
- **Causal attention without masks:** the token loop is unrolled by the
  compiler (one segment per position t, attention loops bounded by t+1) —
  uniform per-token programs, no runtime-dependent bounds. Prefill is
  processed token-by-token through the same path (simple > fast; the
  accelerated runtime may batch, §9.2).

---

## 7. Execution, genesis, output, traces

### 7.1 Execution and canonical N

From genesis state S₀, the step relation of §5 yields S₀ → S₁ → … → S_N
where S_N is the first state with `halted ≠ 0`. **N (the step count, equals
`S_N.step`) and `root_N` are the resolver's claim.** Halted/trapped states
have no successors; there is no padding of traces — a FactSpec claiming
steps past a halt loses via rule V3 (§8.4).

### 7.2 Genesis

- Registers: all zero. Memory: built from the header's region table —
  `ZERO` regions zero, `ARTIFACT` regions byte-copied from blake3-identified
  artifacts (weights blob, LUTs, trig tables, prompt-template token region),
  `INPUT` region zero **in the static image**.
- `static_genesis_root` = memory tree root of that static image. It is part
  of the judge identity, computed by tooling, and **trusted as an audited
  constant** (anyone can recompute it off-chain from the published
  artifacts; an on-chain "hash game" proving it is future work — §13).
- **Judge identity** (created once per judge, referenced by every FactSpec):

  ```
  judge_id = H(0x06 ‖ vm_version ‖ d ‖ p ‖ out_base ‖ out_len ‖
               program_root ‖ static_genesis_root ‖ schedule_root ‖
               blake3_weights ‖ blake3_tokenizer ‖ blake3_template)
  ```
- **Per-question genesis (on-chain):** FactSpec creation supplies the input
  region's pages (≤ 16 × 1 KiB calldata) plus, per page in ascending index
  order, a Merkle update proof. The contract verifies the old leaf is `Z_0`
  (zero page), folds in the new page, and chains the root forward:
  `static_genesis_root → … → genesis_F`. The disputable interval starts at
  `(0, genesis_F)` — **agreed by construction, on-chain**.
- Input region layout: `[n_tokens u32][token_id u32 × n]`, zero-padded to
  whole pages. `input_hash = blake3(token bytes)` — the protocol's canonical
  input commitment is **token-level**; text→token fidelity is re-checkable
  off-chain via the vendored tokenizer (Phase 3) and is outside the fraud
  game (§13).

### 7.3 Output binding

The output region `[out_base, out_base + out_len)` holds
`[n u32][token_id u32 × n]` (≤ 4096 B). The FactSpec's `claimed_output`
bytes are bound to `root_N` by the final-state challenge (§8.5): resolver
must reveal the `root_N` preimage (`mem_root`, `regs`) and Merkle openings
of the output pages; the contract checks `halted == 1` and byte equality.

### 7.4 Traces and checkpoint levels

- **Level-0 trace:** state roots at every step — *defined* always,
  *materialized* only inside a disputed level-2 segment (~10⁵–10⁶ steps →
  seconds; it costs ~1 KiB + d·32 B of hashing per step). The honest path
  hashes nothing per-step: it executes in checkpoint mode and commits only
  checkpoint roots via dirty-page incremental hashing (§3.4, §9.1).
- **Level-1 trace (Phase 3):** roots at compiler-chosen checkpoints — one
  per (token, layer) segment, plus genesis and final. The **schedule** (the
  ascending list of checkpoint step numbers, step₀ = 0, step_C = N) is
  committed as a Merkle tree (leaves `H(0x05 ‖ LE64(step))`, padding leaves
  `LE64(2⁶⁴−1)`), `schedule_root` inside the judge identity. The schedule is
  static per program — EOS early-exit ends the trace at a checkpoint that is
  itself in the schedule (the compiler places one at the halt tail).
- **Phase 1 (toy):** single-level bisection directly over steps; runs are
  ~10⁵–10⁷ steps, fully materializable. The "checkpoint every K" mode is the
  degenerate uniform schedule.
- The resolver MUST derive all traces from the **reference runtime** only
  (§9.2).

---

## 8. Dispute protocol

Identical logic in `MockChain` (logical tick clock) and Sui Move
(`sui::clock` ms). One challenger per FactSpec (first bond wins the slot;
multi-challenger designs are future work, §13).

### 8.1 FactSpec object

```
judge fields:  judge_id parameters (d, p, out_base/len, program_root,
               static_genesis_root, schedule_root)
question:      input pages (inserted at creation → genesis_F stored)
claim:         N: u64, root_N: 32B, claimed_output: vector<u8>
economics:     resolver addr+bond, challenge_window, created_at
status:        Open | Challenged(dispute) | Finalized(output) | Rejected
```

No challenge within the window ⇒ `finalize()` ⇒ Finalized, bond returned.

### 8.2 Bonds and timeouts

Equal bonds escrowed from both parties at challenge time. Every protocol
move has a per-move deadline `T_move`; the party whose move it is and who
misses the deadline loses the whole pot (`claim_timeout()` callable by the
counterparty). Invalid submissions (bad proofs) abort the transaction and
do **not** advance state or reset deadlines — stalling with garbage equals
stalling with silence. Values for bonds/windows are deployment parameters
(Open Question Q3), not consensus rules.

### 8.3 Bisection state machine

```
state:  lo, hi: u64;  root_lo, root_hi: 32B;  mover: Resolver|Challenger;
        deadline;  level: 1|2 (Phase 3)
init:   lo = 0, root_lo = genesis_F  (agreed by construction)
        hi = N, root_hi = root_N     (resolver's claim, challenger disputes)
invariant: parties agree on root_lo, disagree on root_hi
```

Round, while `hi − lo > 1`:

1. `mid = lo + (hi − lo)/2` (floor). **Resolver** posts `root_mid` (deadline-bound).
2. **Challenger** responds: `agree` ⇒ `lo, root_lo ← mid, root_mid`;
   `disagree` ⇒ `hi, root_hi ← mid, root_mid`. (Deadline-bound.)

Each round halves the interval: ⌈log₂ N⌉ rounds ≈ 17–20 for the toy, ≈ 33
single-level for ~5×10⁹ steps. Phase 3 still goes two-level (§8.6) — the
win there is trace-materialization cost, not round count.

When `hi − lo == 1`: **either party** MAY submit `verify_step` (§8.4) before
the deadline; the *outcome is decided by the comparison*, not by who
submitted. If nobody submits a valid proof, whoever's deadline expires
loses. (The resolver always *can* submit honestly if their trace is honest;
the challenger always *can* submit a proof exposing a dishonest root_hi —
the openings are against the agreed `root_lo`, which both parties know the
full state for.)

**Disagreement about N itself** needs no special moves:

- Challenger believes the true machine halts at H < N: traces necessarily
  diverge at or before H's successor… bisection converges to some adjacent
  pair; if the agreed `root_lo` is a halted state, rule V3 makes any claimed
  successor fraudulent ⇒ resolver loses.
- Resolver claims H' < true H (halted too early): `root_N` then differs from
  the true root at that step (true machine isn't halted there) ⇒ ordinary
  divergence, bisection finds it.

### 8.4 One-step verification (`verify_step`)

Payload (logical contents; exact BCS layout fixed in Phase 2):

```
regs:        45 bytes                  — pre-state register file
mem_root:    32 bytes                  — pre-state memory root
instr:       96 bytes + p siblings     — program-tree opening at pc
openA/openB: page (1024 B) + d siblings  — present iff the opcode reads A/B
openW:       page (1024 B) + d siblings  — present iff the opcode writes
```

Verifier algorithm (order normative; ABORT = tx fails, state unchanged):

```
V1  H(0x02 ‖ mem_root ‖ regs) == root_lo                 else ABORT
V2  if regs.halted ≠ 0: CHALLENGER WINS                   (terminality rule —
    checked BEFORE everything else: no fabricated halted state may take the
    trap path instead of auto-fraud)
V3  if regs.pc ≥ 2^p: post = trap(regs); goto V8          (T1 — no opening needed)
V4  verify program opening at index regs.pc → instr       else ABORT
    decode opcode (unknown ⇒ post = trap; goto V8)
V5  compute ea's from regs + instr; check T3..T6 traps that don't need
    memory values (bounds, alignment, shift, k):  trap ⇒ post = trap; goto V8
V6  for each required read: verify opening of page ⌊ea/1024⌋ against
    mem_root (verifier derives the index itself)           else ABORT
    extract operand bytes; value-dependent traps (T5) ⇒ post = trap; goto V8
V7  execute the op (§5): new regs; if it writes, verify openW pre-inclusion
    (else ABORT), patch bytes at ea % 1024, re-fold with the same siblings
    → mem_root'
V8  post_root = H(0x02 ‖ mem_root' ‖ regs')               (mem_root' = mem_root
                                                           if no write)
V9  post_root == root_hi  ⇒ RESOLVER WINS, else CHALLENGER WINS
    pay winner both bonds; FactSpec → Finalized or Rejected
```

Worst-case calldata (d = 20, p = 20): 2 page openings ≈ 2·(1024 + 640) +
instr (96 + 640) + regs/roots ≈ **4.2 KiB**; ~45 SHA3 calls + one 1 KiB page
patch. Trivial for a Sui transaction.

### 8.5 Final-state challenge

A challenger who accepts the execution but disputes the *claim binding* MAY,
instead of bisecting, demand final-state revelation: resolver must submit
(deadline-bound) the `root_N` preimage (`mem_root`, `regs`) plus openings of
all output-region pages. Contract checks `regs.halted == 1`, `regs.step == N`,
and output bytes == `claimed_output`. Mismatch or timeout slashes the
resolver. (This also closes the "correct trace, lying output field" hole.)

### 8.6 Two-level bisection (Phase 3)

`level = 1`: bisect over **checkpoint indices** 0..C exactly as §8.3 (roots
are level-1 checkpoint roots; index 0's root is `genesis_F`). At adjacency
`(i, i+1)`: resolver reveals `step_i, step_{i+1}` with schedule-tree
openings; the dispute switches to `level = 2` over `[step_i, step_{i+1}]`
with the agreed/disputed roots carried over; §8.3 then runs per-step and
terminates in `verify_step`. One `Dispute` object throughout; segment
lengths are ~10⁵–10⁶ steps, so level-2 trace materialization stays cheap.

*(informative)* opML (arXiv 2401.17555) proved this two-phase shape at
LLaMA-7B scale, but its phases bridge two state *representations* (graph
state → MIPS memory image), requiring a verification gadget at the seam.
Here both levels sample the **same** root sequence over the same memory
tree — no bridge exists to attack. See PRIOR_ART.md §1.

### 8.7 Phase 4 (one-shot) — pointer

Resolver additionally commits `trace_root` (tag `0x04` leaves over level-1
roots). A challenger then submits a single transaction: index k, openings
of checkpoints k and k+1, plus either the inner one-step proof or an inner
mini-bisection. Comparison vs interactive play goes in `ANALYSIS.md`
(including the Fiat–Shamir FAQ: bisection moves depend on the challenger's
private trace — they are not public-coin — so FS does not apply).

---

## 9. Determinism & implementation rules

### 9.1 Reference runtime (`vm/` crate)

- **No floats.** `f32`/`f64` are forbidden in `vm/`, `compiler/` (runtime
  paths), and `game/`. Enforced by `#![deny(clippy::float_arithmetic)]` +
  a CI grep. The only float code in the repo lives in offline table/scale
  generation (§6.3) whose *outputs* are golden-hashed artifacts.
- **Sequential normative semantics; algebra-licensed parallelism.** The
  per-step interpreter — the conformance oracle and the dispute-segment
  executor — is single-threaded with fixed iteration order. No
  `HashMap`/`HashSet` iteration anywhere in any execution path — `Vec` or
  `BTreeMap` only. The reference **checkpoint mode** (same op-semantics
  code, hashing only at checkpoints over dirty pages) MAY additionally use
  exactly two transformations, both bit-exact by construction:
  (1) SIMD/reordering *within* a DOT lane set — wrapping-i64 addition is
  associative and commutative; (2) thread parallelism across **independent
  output cells** (e.g. GEMM rows), which share no state. Nothing else —
  changed reduction algorithms, fused requant, etc. belong in the Phase 3
  predictor (§9.2). Checkpoint mode MUST be differentially tested against
  the per-step oracle in CI (full toy runs; sampled Qwen segments).
- All arithmetic uses explicit `wrapping_*`/`saturating_*`/checked calls
  per this spec. Tests build with `overflow-checks = true` so any
  *unspecified* overflow panics instead of silently wrapping.
- Trace digests: `trace_digest = H(root_0 ‖ root_1 ‖ … ‖ root_N)` (or
  running-hash equivalent) — the cross-platform golden value (Invariant 1)
  pinned in CI for x86_64-linux and aarch64-macOS.

### 9.2 Predictor runtime (Phase 3, optional acceleration)

Anything beyond §9.1's two licensed transformations — batched prefill, GPU
kernels, fused ops, alternative reduction algorithms — lives here. The
predictor MAY use any of it; MUST reproduce the reference state byte-exactly
at every level-1 checkpoint (asserted in tests); MUST NOT be the source of
any published trace or root — it exists to produce the *answer* fast
(seconds, §1.4), while the reference checkpoint mode produces the
*commitments*.

*(informative)* Precedent: Cartesi's `machine-kernels-llama2.c` offloads
matmuls to the host CPU while the in-machine kernel can replay them for
proofs — the same predictor/reference split. Gensyn's RepOps and EigenAI
demonstrate bitwise-deterministic inference on production GPUs, which makes
a deterministic-GPU predictor a credible future backend (FW-5,
PRIOR_ART.md §3–4).

### 9.3 Move signed arithmetic (no i64 in Move)

i64 values cross the boundary as their two's-complement u64 bit pattern.
Move helpers (normative formulas; Move aborts on native overflow, so
wrapping is built via u128):

```
wadd(a, b) = ((a as u128 + b as u128)  & 0xFFFF_FFFF_FFFF_FFFF) as u64
wmul(a, b) = ((a as u128 * b as u128)  & 0xFFFF_FFFF_FFFF_FFFF) as u64   // operands < 2^64 ⇒ product < 2^128, no abort
neg(x)     = wadd(x ^ 0xFFFF_FFFF_FFFF_FFFF, 1)
sign(x)    = x >> 63
sar(x, s)  = s == 0 ? x : (sign(x) == 0 ? x >> s
                                        : (x >> s) | (MAX << (64 − s)))  // 1 ≤ s ≤ 63 ⇒ 64−s ∈ [1,63], no shift-by-64
slt(a, b)  = (a ^ (1<<63)) < (b ^ (1<<63))                               // signed compare via offset trick
sext8(b)   = b ≥ 0x80 ? b | 0xFFFF_FFFF_FFFF_FF00 : b                    // sim. sext16/sext32
sdiv: q = umag(a) / umag(b) with umag(x) = sign(x) == 1 ? neg(x) : x;    // b > 0 by T5 ⇒ only a needs sign handling
      result = sign(a) == 1 ? neg(q) : q
```

**Cross-test requirement (the soundness test of the project):** for every
opcode, Rust-VM single-step == Move `verify_step` partial result —
exhaustive over both i8 operands for `MAC8` (65 536 cases), boundary +
randomized vectors for everything else (i64::MIN/MAX, ±1, 0, half-rounding
points, saturation edges).

---

## 10. Known pitfalls → where handled

| Pitfall (from brief) | Resolution |
|---|---|
| Rounding rule ambiguity | One rule: `rnd` round-half-to-even (§5.1), boundary vectors mandatory (§11) |
| Saturation vs wraparound | Stores saturate, accumulator wraps — pinned in §5.2 notes; tested at ±128/±32768 edges |
| LUT out-of-domain | `LUT16` saturates the index (`sat16`) — edge entries are the clamp (§6.3) |
| Merkle endianness / leaf encoding / empty nodes | LE everywhere; tagged preimages; `Z_l` chain (§3.4) |
| Second-preimage (leaf vs node) | Domain tags 0x00–0x06 (§2.2) |
| No i64 in Move | Two's-complement-on-u64 with normative formulas + exhaustive MAC8 cross-test (§9.3) |
| Accelerated runtime poisoning traces | Forbidden by §9.2 — reference re-verifies all checkpoints |
| Hash-map iteration / parallelism nondeterminism | §9.1 bans both in execution paths |
| Genesis agreement | `genesis_F` derived on-chain from audited static root + input insertion proofs (§7.2) |
| Output not bound to trace | Final-state challenge (§8.5) |
| Resolver claims wrong N | Terminality rule V3 + ordinary divergence (§8.3) |

---

## 11. Conformance tests (Phase 0–1 gate)

| ID | Test |
|---|---|
| C-1 | `rnd` boundary table (§5.1) exactly; plus property: `rnd(x,s)` == high-precision reference for 10⁶ random (x, s) |
| C-2 | `sat8`/`sat16` edges: −129 → −128, 128 → 127, i64::MIN/MAX |
| C-3 | `trunc_div`: (7,2) = 3, (−7,2) = −3, (i64::MIN, 1) = i64::MIN, (i64::MIN, 2) = −2⁶² |
| C-4 | Merkle: golden vectors for `Z_0..Z_4`, leaf/node/state-root hashes of a 4-page toy memory; proof verify + incremental update == full rebuild (property test) |
| C-5 | Per-op golden vectors: each opcode once, hand-computed pre/post state |
| C-6 | **Golden trace digest**: fixed toy program + seed ⇒ pinned `trace_digest`, CI on x86_64 + aarch64 (Invariant 1) |
| C-7 | Honest resolver, no challenger ⇒ finalizes after window |
| C-8 | Dishonest resolver (bit-flip / corrupted micro-op at random step, 100 fuzz seeds) vs honest challenger ⇒ challenger wins AND isolated step == injected step |
| C-9 | Honest resolver vs lying challenger ⇒ resolver wins |
| C-10 | Stalling party (either role, any phase) ⇒ loses by timeout |
| C-11 | Final-state challenge: tampered `claimed_output` ⇒ resolver slashed |
| C-12 | (Phase 2) Rust↔Move per-op equivalence incl. exhaustive MAC8 |
| C-13 | DOT≡MAC property: `DOT8(imm=K)` == the K-step `MAC8` chain over the same lanes (likewise `DOT16`/`MAC16`), incl. K = 1, full-cap, and A = B aliasing; plus SIMD-vs-scalar lane-order equality |
| C-14 | Checkpoint-mode ≡ per-step-oracle: identical checkpoint roots on full toy runs and randomly sampled segments (§9.1) |

---

## 12. Parameters

| Constant | Value | Where | Rationale |
|---|---|---|---|
| `H` | SHA3-256 | §2.2 | Sui-native + ubiquitous (Q1: blake2b256 alt) |
| Artifact hash | blake3 | §2.2 | brief; off-chain only |
| `PAGE` | 1024 B | §3.1 | calldata vs depth balance; multiple of 8 |
| `d` (mem depth) | per-program, 10–24 | §3.1 | 1 MiB–16 GiB; path ≤ 768 B |
| `p` (prog depth) | per-program, ≤ 32 | §3.5 | pc is u32 |
| Register encoding | 45 B | §3.2 | pc4 + halted1 + step8 + acc8 + aux8 + idx16 |
| Instruction | 96 B | §4.1 | 3 × 24 B operand descriptors + header |
| Opcodes | 0x01–0x19, append-only | §5.2 | 0x00 reserved as padding-trap |
| DOT line | 64 B; lane caps 64 (i8) / 32 (i16); `ea % 64 == 0` | §5.2 | one cache line per step: divides LLM dims, never straddles a page, ~64× dispatch amortization (Q5) |
| LUT size | 65 536 × i16 | §6.3 | full sat16 domain ⇒ no range reduction logic |
| `out_len` | ≤ 4096 B | §4.3 | one answer; bounds final-state challenge cost |
| Input region | ≤ 16 pages | §4.3 | ≈ 4096 tokens; bounds FactSpec-creation gas (Q4) |
| Rounding | half-to-even | §5.1 | brief recommendation; tested at C-1 |
| Move LoC budget | ≤ 800 / ≤ 450 core | §1.3 | Invariant 4 |
| Bonds, windows, `T_move` | deployment params | §8.2 | economics, not consensus (Q3) |

---

## 13. Open questions / future work

| # | Question |
|---|---|
| Q1 | Commitment hash: stay SHA3-256 or switch to blake2b256 (also Sui-native, faster off-chain)? Affects every golden vector — must decide before Phase 0 code. |
| Q2 | `PAGE = 1024` acceptable? (Smaller pages ⇒ smaller calldata, deeper trees.) |
| Q3 | Bond sizes, challenge window, per-move timeout values (Phase 2 deployment config). |
| Q4 | Input cap 16 pages (~4096 tokens) sufficient for the evidence-text use case? |
| Q5 | DOT line size: 64 B (default — zero padding for common dims) vs 256 B/1024 B (another 4–16× fewer steps and faster traces, but rows must pad to the line size and the Move loop grows). Revisit with Phase 3 benchmark data. |
| FW-1 | On-chain verification of `static_genesis_root` (hash game over artifact bytes). |
| FW-2 | Multi-challenger / griefing-resistant bonds — reference design: Cartesi Dave's Permissionless Refereed Tournaments (PRIOR_ART.md §2). |
| FW-3 | Tokenizer-in-VM (close the text→token trust gap). |
| FW-4 | zk-per-layer hybrid — interface stub + size estimates in Phase 4 `ANALYSIS.md`; compare also the TEE/committee lane (EigenAI, Optimistic TEE-Rollups — PRIOR_ART.md §4). |
| FW-5 | Deterministic-GPU predictor backend (RepOps/EigenAI-style fixed reduction order). Predictor only — the on-chain referee stays a tiny integer contract, so the committed semantics stay integer. |
| FW-7 | ISA additions for the wide-activation Qwen path (the i32 residual carrier + i16 matmul inputs that real outlier ranges force): `LD16`, a mixed `DOT8X16` (i8 weights × i16 activations), and an `AXPY8` line-op (out_line += scalar·in_line — row-major V attention without per-element MACs). The measured quality campaign added the per-(row,block) multiplier structure (Q8_0-style blocks on BOTH operands, exact i64 accumulation) — expressible today via DOT+MUL32+ST32/LD32 chains (~5 ops/block) or one fused `DOTBM` op (dot-line, ×M, accumulate) at 1 op/block. Append-only opcodes; the native runtime already implements all these semantics. |
| FW-6 | **Float-ISA extension (the "EigenAI-fast, Sui-trustless" track).** Add `DOT16F`/`DOT16BF` micro-ops whose normative semantics are the SAME canonical binary-tree fp32 reduction that batch-invariant GPU kernels already perform (EigenAI arXiv 2602.00182 §kernels; vLLM/SGLang deterministic modes). Then: the GPU engine is a bit-exact implementation of the committed semantics at ~native speed (no quantization, full model quality), and the one-step verifier checks a single disputed flop via softfloat in pure-integer Move (~150–300 lines per float op — strains Invariant 4, measured before adoption). Must pin: bf16/fp16 storage, fp32 accumulate, exact tree bracketing per reduction size, RN rounding, FTZ/denormal policy, committed polynomial exp/silu (never libm), NaN propagation. Everything else (Merkle pages, bisection, two-level traces, dispute flow) is unchanged — the protocol is arithmetic-agnostic by construction. |
