//! Toy-model compiler: lowers the 2-layer char transformer to the tensor-VM
//! ISA following the SPEC §6.4 recipes exactly.
//!
//! The token loop is fully unrolled (one program segment per position, with
//! attention length p+1 baked in — SPEC's "causal attention without masks");
//! inner loops use `LOOP` with static trip counts, so the step count of the
//! whole program is a compile-time fact. `Compiled::total_steps` is that
//! prediction, and the equivalence test asserts the VM agrees exactly.

pub mod fqwen;
pub mod qwen;

use toy_model::layout::Layout;
use toy_model::params::*;
use toy_model::scales::*;
use vm::isa::{Instr, Opcode, Operand};

/// Index-register conventions.
const K_TOK: u8 = 0; // embedding row id (LDIDX)
const K_ELEM: u8 = 1; // all inner loops + argmax scans

pub struct Compiled {
    pub program: Vec<Instr>,
    /// Program tree depth: smallest p with len ≤ 2^p.
    pub p: u8,
    /// Predicted total steps (== final `step` register; tested).
    pub total_steps: u64,
    /// Predicted cumulative step count after each position — the toy's
    /// per-token checkpoint schedule (SPEC §7.4 level-1).
    pub token_boundaries: Vec<u64>,
    /// Program index of the next instruction at each boundary (the pc the
    /// VM holds at that step) — checkpoint-mode runtimes reconstruct the
    /// committed register file from this.
    pub boundary_pcs: Vec<u32>,
    pub n_prompt: usize,
    pub n_gen: usize,
}

struct Em {
    prog: Vec<Instr>,
    steps: u64,
    mult: u64, // product of enclosing static trip counts
}

impl Em {
    fn emit(&mut self, i: Instr) {
        self.prog.push(i);
        self.steps += self.mult;
    }

    /// Bottom-tested LOOP: body runs exactly `trips` times (SPEC §5.2).
    fn loop_over(&mut self, k: u8, trips: u32, body: impl FnOnce(&mut Em)) {
        assert!(trips >= 1, "LOOP is bottom-tested; zero trips impossible");
        let start = self.prog.len() as u32;
        self.mult *= trips as u64;
        body(self);
        // The LOOP instruction itself executes once per iteration.
        self.emit(Instr {
            k,
            target: start,
            imm: trips,
            ..Instr::op(Opcode::Loop)
        });
        self.mult /= trips as u64;
    }
}

// -- operand/instruction shorthands ----------------------------------------

fn at(base: u64) -> Operand {
    Operand::at(base)
}

/// Operand whose address strides by idx[k].
fn st(base: u64, k: u8, stride: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k as usize] = stride;
    o
}

fn ldc(v: i32) -> Instr {
    Instr { imm: v as u32, ..Instr::op(Opcode::Ldc) }
}

fn dot8(a: Operand, b: Operand, lanes: u32) -> Instr {
    Instr { imm: lanes, a, b, ..Instr::op(Opcode::Dot8) }
}

fn dot16(a: Operand, b: Operand, lanes: u32) -> Instr {
    Instr { imm: lanes, a, b, ..Instr::op(Opcode::Dot16) }
}

fn mac8(a: Operand, b: Operand) -> Instr {
    Instr { a, b, ..Instr::op(Opcode::Mac8) }
}

fn ld8(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld8) }
}

fn ld32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld32) }
}

fn add32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Add32) }
}

fn mul32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Mul32) }
}

fn div32(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Div32) }
}

fn shift(s: u8) -> Instr {
    Instr { s, ..Instr::op(Opcode::ShiftRndn) }
}

fn clamp8(w: Operand) -> Instr {
    Instr { w, ..Instr::op(Opcode::Clamp8) }
}

fn clamp16(w: Operand) -> Instr {
    Instr { w, ..Instr::op(Opcode::Clamp16) }
}

fn lut16(table_base: u64) -> Instr {
    Instr { a: at(table_base), ..Instr::op(Opcode::Lut16) }
}

fn st32_acc(w: Operand) -> Instr {
    Instr { k: 0, w, ..Instr::op(Opcode::St32) }
}

fn st32_aux(w: Operand) -> Instr {
    Instr { k: 1, w, ..Instr::op(Opcode::St32) }
}

fn ldidx(k: u8, a: Operand) -> Instr {
    Instr { k, a, ..Instr::op(Opcode::Ldidx) }
}

fn argmax(k: u8, a: Operand) -> Instr {
    Instr { k, a, ..Instr::op(Opcode::ArgmaxStep) }
}

// -- recipe blocks (SPEC §6.4) ----------------------------------------------

/// RMSNorm: x → xn. Sum of squares via DOT8(x,x) aliasing, mean via DIV32,
/// rsqrt LUT, then per-element x·r32·γ >> S_NORM → i8.
fn rmsnorm(em: &mut Em, lay: &Layout, gamma: u64) {
    em.emit(ldc(0));
    em.emit(dot8(at(lay.x), at(lay.x), 64));
    em.emit(div32(at(lay.c_d)));
    em.emit(lut16(lay.lut_rsqrt));
    em.emit(st32_acc(at(lay.r32)));
    em.loop_over(K_ELEM, D as u32, |em| {
        em.emit(ld8(st(lay.x, K_ELEM, 1)));
        em.emit(mul32(at(lay.r32)));
        em.emit(mul32(st(gamma, K_ELEM, 4)));
        em.emit(shift(S_NORM));
        em.emit(clamp8(st(lay.xn, K_ELEM, 1)));
    });
}

/// One projection block: rows of `w` (head-row-block offset pre-added by
/// caller) dotted with xn, requantized by `s_out`, stored via `store`
/// (a closure building the per-row write operand and store opcode).
fn proj_rows(
    em: &mut Em,
    w_rows_base: u64,
    xn: u64,
    rows: u32,
    s_out: u8,
    store: impl Fn() -> Instr,
) {
    em.loop_over(K_ELEM, rows, |em| {
        em.emit(ldc(0));
        em.emit(dot8(st(w_rows_base, K_ELEM, D as u32), at(xn), 64));
        em.emit(shift(s_out));
        em.emit(store());
    });
}

// -- the compiler ------------------------------------------------------------

pub fn compile_toy(lay: &Layout, n_prompt: usize, n_gen: usize) -> Compiled {
    assert!(n_prompt >= 1 && n_gen >= 1);
    let n_pos = n_prompt + n_gen - 1; // last generated token is not fed back
    assert!(n_pos <= MAX_SEQ, "sequence exceeds MAX_SEQ");

    let mut em = Em { prog: Vec::new(), steps: 0, mult: 1 };
    let mut token_boundaries = Vec::with_capacity(n_pos);
    let mut boundary_pcs = Vec::with_capacity(n_pos);

    for p in 0..n_pos {
        // Current token id: prompt cell or the decode feedback cell.
        let tok_addr = if p < n_prompt {
            lay.input + 4 + 4 * p as u64 // input region: [n][ids…]
        } else {
            lay.tok
        };
        em.emit(ldidx(K_TOK, at(tok_addr)));

        // Embedding + position embedding: x[c] = sat8(emb[id][c] + pos[p][c]).
        em.loop_over(K_ELEM, D as u32, |em| {
            // ea = emb + id·64 + c — both index registers in one operand.
            let mut a = Operand::at(lay.emb);
            a.stride[K_TOK as usize] = D as u32;
            a.stride[K_ELEM as usize] = 1;
            em.emit(ld8(a));
            em.emit(mac8(st(lay.pos + (D * p) as u64, K_ELEM, 1), at(lay.c_one_i8)));
            em.emit(clamp8(st(lay.x, K_ELEM, 1)));
        });

        for l in 0..LAYERS {
            // ---- attention ----
            rmsnorm(&mut em, lay, lay.g1[l]);
            for h in 0..HEADS {
                let row0 = (h * D_HEAD * D) as u64; // head-major row block
                // Q → q[h] (i16), K → cache row p (i16), V → Vᵀ column p (i8).
                proj_rows(&mut em, lay.wq[l] + row0, lay.xn, D_HEAD as u32, S_QK_I16, || {
                    clamp16(st(lay.q + (h * D_HEAD * 2) as u64, K_ELEM, 2))
                });
                proj_rows(&mut em, lay.wk[l] + row0, lay.xn, D_HEAD as u32, S_QK_I16, || {
                    clamp16(st(lay.kc[l][h] + (p * D_HEAD * 2) as u64, K_ELEM, 2))
                });
                proj_rows(&mut em, lay.wv[l] + row0, lay.xn, D_HEAD as u32, S_PROJ_I8, || {
                    clamp8(st(lay.vt[l][h] + p as u64, K_ELEM, MAX_SEQ as u32))
                });
            }
            for h in 0..HEADS {
                let t1 = (p + 1) as u32; // causal: attend to positions 0..=p
                let q_h = lay.q + (h * D_HEAD * 2) as u64;
                // Logits j = 0..=p: one DOT16 line per cached K row.
                em.loop_over(K_ELEM, t1, |em| {
                    em.emit(ldc(0));
                    em.emit(dot16(at(q_h), st(lay.kc[l][h], K_ELEM, (D_HEAD * 2) as u32), D_HEAD as u32));
                    em.emit(shift(S_LOGIT_Q411));
                    em.emit(st32_acc(st(lay.att32, K_ELEM, 4)));
                });
                // max (argmax value; index unused) → negate via ·(−1).
                em.emit(ldc(i32::MIN));
                em.loop_over(K_ELEM, t1, |em| {
                    em.emit(argmax(K_ELEM, st(lay.att32, K_ELEM, 4)));
                });
                em.emit(mul32(at(lay.c_neg1)));
                em.emit(st32_acc(at(lay.neg_max)));
                // exp(logit − max) via LUT, stored as i32.
                em.loop_over(K_ELEM, t1, |em| {
                    em.emit(ld32(st(lay.att32, K_ELEM, 4)));
                    em.emit(add32(at(lay.neg_max)));
                    em.emit(lut16(lay.lut_exp));
                    em.emit(st32_acc(st(lay.e32, K_ELEM, 4)));
                });
                // sum of exps (≤ 64·2^14 = 2^20, no overflow).
                em.emit(ldc(0));
                em.loop_over(K_ELEM, t1, |em| em.emit(add32(st(lay.e32, K_ELEM, 4))));
                em.emit(st32_acc(at(lay.sum)));
                // probs[j] = (e·2^14)/sum >> 7 → Q0.7. Stale entries beyond p
                // are zero forever: p grows monotonically and genesis is zero.
                em.loop_over(K_ELEM, t1, |em| {
                    em.emit(ld32(st(lay.e32, K_ELEM, 4)));
                    em.emit(mul32(at(lay.c_2p14)));
                    em.emit(div32(at(lay.sum)));
                    em.emit(shift(S_PROB_Q07));
                    em.emit(clamp8(st(lay.probs, K_ELEM, 1)));
                });
                // ctx[r] = Σ_j probs[j]·Vᵀ[r][j] — one DOT8 line per dim.
                em.loop_over(K_ELEM, D_HEAD as u32, |em| {
                    em.emit(ldc(0));
                    em.emit(dot8(at(lay.probs), st(lay.vt[l][h], K_ELEM, MAX_SEQ as u32), MAX_SEQ as u32));
                    em.emit(shift(S_CTX_I8));
                    em.emit(clamp8(st(lay.attnx + (h * D_HEAD) as u64, K_ELEM, 1)));
                });
            }
            // O projection + residual: x = sat8(x + Wo·attnx).
            em.loop_over(K_ELEM, D as u32, |em| {
                em.emit(ldc(0));
                em.emit(dot8(st(lay.wo[l], K_ELEM, D as u32), at(lay.attnx), 64));
                em.emit(shift(S_PROJ_I8));
                em.emit(mac8(st(lay.x, K_ELEM, 1), at(lay.c_one_i8)));
                em.emit(clamp8(st(lay.x, K_ELEM, 1)));
            });

            // ---- FFN ----
            rmsnorm(&mut em, lay, lay.g2[l]);
            em.loop_over(K_ELEM, FFN as u32, |em| {
                em.emit(ldc(0));
                em.emit(dot8(st(lay.w1[l], K_ELEM, D as u32), at(lay.xn), 64));
                em.emit(shift(S_FFN_Q411));
                em.emit(lut16(lay.lut_silu));
                em.emit(shift(S_SILU_I8));
                em.emit(clamp8(st(lay.h_ffn, K_ELEM, 1)));
            });
            em.loop_over(K_ELEM, D as u32, |em| {
                em.emit(ldc(0));
                for j in 0..(FFN / 64) as u64 {
                    // W2 row = 256 bytes = 4 DOT8 lines, unrolled.
                    em.emit(dot8(
                        st(lay.w2[l] + 64 * j, K_ELEM, FFN as u32),
                        at(lay.h_ffn + 64 * j),
                        64,
                    ));
                }
                em.emit(shift(S_FFN_DOWN_I8));
                em.emit(mac8(st(lay.x, K_ELEM, 1), at(lay.c_one_i8)));
                em.emit(clamp8(st(lay.x, K_ELEM, 1)));
            });
        }

        // ---- decode head at decision positions ----
        if p >= n_prompt - 1 {
            rmsnorm(&mut em, lay, lay.gf);
            em.loop_over(K_ELEM, VOCAB as u32, |em| {
                em.emit(ldc(0));
                em.emit(dot8(st(lay.head, K_ELEM, D as u32), at(lay.xn), 64));
                em.emit(st32_acc(st(lay.logits, K_ELEM, 4)));
            });
            // Greedy argmax; ascending scan ⇒ ties to lowest id (SPEC §5.2).
            em.emit(ldc(i32::MIN));
            em.loop_over(K_ELEM, VOCAB as u32, |em| {
                em.emit(argmax(K_ELEM, st(lay.logits, K_ELEM, 4)));
            });
            let g = (p - (n_prompt - 1)) as u64;
            em.emit(st32_aux(at(lay.tok))); // decode feedback
            em.emit(st32_aux(at(lay.output + 4 + 4 * g)));
            em.emit(ldc(g as i32 + 1)); // output count so far
            em.emit(st32_acc(at(lay.output)));
        }
        token_boundaries.push(em.steps);
        boundary_pcs.push(em.prog.len() as u32);
    }
    em.emit(Instr::op(Opcode::Halt));

    let len = em.prog.len() as u64;
    let p = (0..=vm::MAX_PROG_DEPTH).find(|&p| len <= 1u64 << p).unwrap();
    Compiled {
        program: em.prog,
        p,
        total_steps: em.steps,
        token_boundaries,
        boundary_pcs,
        n_prompt,
        n_gen,
    }
}
