//! Qwen3 compiler: lowers the integer Qwen forward pass (the checkpoint-mode
//! native runtime in models/qwen/src/forward.rs) to the tensor-VM ISA.
//!
//! Same discipline as the toy compiler: one program segment per position
//! (causal attention length baked in), inner loops via bottom-tested `LOOP`
//! with static trip counts, so the step count of the whole judgment is a
//! compile-time fact. The native runtime was aligned write-for-write with
//! this lowering (xp staging, r32 sharing, t_* sigmoid scratch, ST32
//! truncation) — C-14 at Qwen scale asserts the memory roots agree at every
//! token boundary.
//!
//! Register conventions: every segment ends with a canonicalizing tail
//! (acc ← 0 via guaranteed-hit ARGMAX on c_zero, idx ← 0 via LOOP
//! auto-reset + LDIDX), so the boundary register file is fully static:
//! { pc: boundary_pc, step: boundary_step, acc: 0, aux: 0, idx: [0; 4] }.
//!
//! The one data-dependent branch in the model (the sigmoid's sign split —
//! the exp LUT is one-sided in the negative domain) is lowered as an exact
//! sign dispatch: CLAMP8(x·2^30) yields a byte in {0x00, 0x7F, 0x80} and
//! JEQ-on-0x80 takes the negative arm. Both arms execute the same number of
//! steps, so static step prediction survives.

use qwen::layout::QwenLayout;
use qwen::quant::{IntModel, NormSite, SHIFT};
use vm::isa::{Instr, Opcode, Operand};

const K0: u8 = 0; // token id (embedding); kv-head loops
const K1: u8 = 1; // element/pair/group/in-chunk-row loops
const K2: u8 = 2; // row / score-position / dim loops
const K3: u8 = 3; // innermost gather loops; absolute vocab row (head)

/// Streaming-head chunk: one cycled page of i32 logits.
const CHUNK: u64 = 256;

pub struct QwenCompiled {
    pub program: Vec<Instr>,
    pub p: u8,
    /// Debug probes: (label, predicted cumulative steps, pc) between blocks
    /// — the C-14 bin walks these to localize a step-prediction bug.
    pub probes: Vec<(String, u64, u32)>,
    /// Predicted total steps (== final `step` register; asserted by C-14).
    pub total_steps: u64,
    /// Cumulative step count at each position boundary.
    pub token_boundaries: Vec<u64>,
    /// pc held at each boundary (the next segment's entry).
    pub boundary_pcs: Vec<u32>,
    pub n_prompt: usize,
    pub n_gen: usize,
}

struct Em {
    prog: Vec<Instr>,
    steps: u64,
    mult: u64,
    probes: Vec<(String, u64, u32)>,
}

impl Em {
    fn emit(&mut self, i: Instr) {
        self.prog.push(i);
        self.steps += self.mult;
    }

    fn pc(&self) -> u32 {
        self.prog.len() as u32
    }

    fn probe(&mut self, label: String) {
        let (s, p) = (self.steps, self.pc());
        self.probes.push((label, s, p));
    }

    /// Bottom-tested LOOP: body runs exactly `trips` times.
    fn loop_over(&mut self, k: u8, trips: u32, body: impl FnOnce(&mut Em)) {
        assert!(trips >= 1, "LOOP is bottom-tested; zero trips impossible");
        let start = self.pc();
        self.mult *= trips as u64;
        body(self);
        self.emit(Instr { k, target: start, imm: trips, ..Instr::op(Opcode::Loop) });
        self.mult /= trips as u64;
    }
}

// -- operand/instruction shorthands (same dialect as the toy compiler) -------

fn at(base: u64) -> Operand {
    Operand::at(base)
}

fn st(base: u64, k: u8, stride: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k as usize] = stride;
    o
}

fn st2(base: u64, k1: u8, s1: u32, k2: u8, s2: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k1 as usize] = s1;
    o.stride[k2 as usize] = s2;
    o
}

fn st3(base: u64, k1: u8, s1: u32, k2: u8, s2: u32, k3: u8, s3: u32) -> Operand {
    let mut o = Operand::at(base);
    o.stride[k1 as usize] = s1;
    o.stride[k2 as usize] = s2;
    o.stride[k3 as usize] = s3;
    o
}

fn ldc(v: i32) -> Instr {
    Instr { imm: v as u32, ..Instr::op(Opcode::Ldc) }
}

fn ld8(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld8) }
}

fn ld16(a: Operand) -> Instr {
    Instr { a, ..Instr::op(Opcode::Ld16) }
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

fn mac16(a: Operand, b: Operand) -> Instr {
    Instr { a, b, ..Instr::op(Opcode::Mac16) }
}

fn dot16(a: Operand, b: Operand, lanes: u32) -> Instr {
    Instr { imm: lanes, a, b, ..Instr::op(Opcode::Dot16) }
}

fn dotbm(a: Operand, b: Operand, w: Operand) -> Instr {
    Instr { imm: 64, a, b, w, ..Instr::op(Opcode::Dotbm) }
}

fn shift(s: u8) -> Instr {
    assert!(s <= 63);
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

fn argmax_off(k: u8, a: Operand) -> Instr {
    Instr { k, a, imm: 0, ..Instr::op(Opcode::ArgmaxOff) }
}

// -- recipe blocks (mirror forward.rs helper-for-helper) ---------------------

/// RMSNorm i32 carrier → i16 (forward.rs rmsnorm_to_i16): stage prescaled
/// sat16 values in xp, MAC16-square them for the exact i64 sum, rsqrt LUT,
/// then per-element x·r (rnd 14) ·γ (rnd shift−14) → xn.
fn rmsnorm(em: &mut Em, lay: &QwenLayout, src: u64, gamma: u64, site: &NormSite, h: u32) {
    assert!(site.elem_pre && site.shift >= 14);
    em.loop_over(K1, h, |em| {
        em.emit(ld32(st(src, K1, 4)));
        em.emit(shift(site.pre_shift));
        em.emit(clamp16(st(lay.xp, K1, 2)));
    });
    em.emit(ldc(0));
    em.loop_over(K1, h, |em| {
        em.emit(mac16(st(lay.xp, K1, 2), st(lay.xp, K1, 2)));
    });
    em.emit(div32(at(lay.c_h)));
    em.emit(lut16(lay.lut_rsqrt));
    em.emit(st32_acc(at(lay.r32)));
    em.loop_over(K1, h, |em| {
        em.emit(ld32(st(src, K1, 4)));
        em.emit(mul32(at(lay.r32)));
        em.emit(shift(14));
        em.emit(mul32(st(gamma, K1, 4)));
        em.emit(shift(site.shift - 14));
        em.emit(clamp16(st(lay.xn, K1, 2)));
    });
}

/// Blocked projection (kernels::gemv_blocked semantics): per row, DOTBM
/// accumulates Σ_b partial_b·M[r][b] exactly in acc, one rnd at the end.
#[allow(clippy::too_many_arguments)] // codegen helper, not API surface
fn proj_blocked(
    em: &mut Em,
    w: u64,
    m_region: u64,
    m_shift: u8,
    rows: u32,
    cols: u64,
    x_base: u64,
    store: impl Fn(&mut Em),
) {
    let blocks = cols / 64;
    em.loop_over(K2, rows, |em| {
        em.emit(ldc(0));
        for b in 0..blocks {
            em.emit(dotbm(
                st(w + 64 * b, K2, cols as u32),
                at(x_base + 128 * b),
                st(m_region + 4 * b, K2, 4 * blocks as u32),
            ));
        }
        em.emit(shift(m_shift));
        store(em);
    });
}

/// QK-norm on one head's i16 vector at `base + idx[K0]·256`, in place.
fn qknorm(em: &mut Em, lay: &QwenLayout, base: u64, gamma: u64, site: &NormSite, dh: u32) {
    em.emit(ldc(0));
    for ch in 0..(dh as u64 / 32) {
        let line = st(base + 64 * ch, K0, 256);
        em.emit(dot16(line, line, 32));
    }
    em.emit(div32(at(lay.c_dh)));
    em.emit(shift(site.pre_shift));
    em.emit(lut16(lay.lut_rsqrt));
    em.emit(st32_acc(at(lay.r32)));
    em.loop_over(K1, dh, |em| {
        em.emit(ld16(st2(base, K0, 256, K1, 2)));
        em.emit(mul32(at(lay.r32)));
        em.emit(mul32(st(gamma, K1, 4)));
        em.emit(shift(site.shift));
        em.emit(clamp16(st2(base, K0, 256, K1, 2)));
    });
}

/// Rotary on one head's i16 vector at `base + idx[K0]·256` (rotate-half).
/// na is staged through t_na16 because q[p2] feeds nb before being replaced.
fn rope(em: &mut Em, lay: &QwenLayout, base: u64, pos: usize, dh: u32) {
    let half = dh / 2;
    let tcos = lay.rope_cos + (pos as u64) * half as u64 * 2;
    let tsin = lay.rope_sin + (pos as u64) * half as u64 * 2;
    let tnsin = lay.rope_nsin + (pos as u64) * half as u64 * 2;
    em.loop_over(K1, half, |em| {
        em.emit(ldc(0));
        em.emit(mac16(st2(base, K0, 256, K1, 2), st(tcos, K1, 2)));
        em.emit(mac16(st2(base + half as u64 * 2, K0, 256, K1, 2), st(tnsin, K1, 2)));
        em.emit(shift(14));
        em.emit(clamp16(at(lay.t_na16)));
        em.emit(ldc(0));
        em.emit(mac16(st2(base, K0, 256, K1, 2), st(tsin, K1, 2)));
        em.emit(mac16(st2(base + half as u64 * 2, K0, 256, K1, 2), st(tcos, K1, 2)));
        em.emit(shift(14));
        em.emit(clamp16(st2(base + half as u64 * 2, K0, 256, K1, 2)));
        em.emit(ld16(at(lay.t_na16)));
        em.emit(clamp16(st2(base, K0, 256, K1, 2)));
    });
}

// -- the compiler -------------------------------------------------------------

pub fn compile_qwen(lay: &QwenLayout, im: &IntModel, n_prompt: usize, n_gen: usize) -> QwenCompiled {
    assert!(n_prompt >= 1 && n_gen >= 1);
    let cfg = &im.cfg;
    let (h, dh, f) = (cfg.hidden_size as u64, cfg.head_dim as u64, cfg.intermediate_size as u64);
    let (nh, nkv) = (cfg.num_attention_heads as u64, cfg.num_key_value_heads as u64);
    let group = (nh / nkv) as u32;
    let vocab = cfg.vocab_size as u64;
    let n_pos = n_prompt + n_gen - 1;
    assert!(n_pos <= qwen::layout::MAX_SEQ);

    let mut em = Em { prog: Vec::new(), steps: 0, mult: 1, probes: Vec::new() };
    let mut token_boundaries = Vec::with_capacity(n_pos);
    let mut boundary_pcs = Vec::with_capacity(n_pos);

    for pos in 0..n_pos {
        // ---- embedding into the i32 residual carrier ----
        // Input region: [count][ids…] (image.rs convention).
        let tok_addr =
            if pos < n_prompt { lay.input + 4 + 4 * pos as u64 } else { lay.tok };
        em.emit(ldidx(K0, at(tok_addr)));
        em.loop_over(K1, h as u32, |em| {
            em.emit(ld8(st2(lay.emb, K0, h as u32, K1, 1)));
            em.emit(mul32(st(lay.m_emb_arr, K1, 4)));
            em.emit(shift(SHIFT));
            em.emit(st32_acc(st(lay.x, K1, 4)));
        });
        // idx0 holds the token id — LOOP is bottom-tested, so reset it
        // before K0 serves as a head-loop counter.
        em.emit(ldidx(K0, at(lay.c_zero)));

        em.probe(format!("p{pos} emb"));
        for (l, lw) in im.layers.iter().enumerate() {
            let a = &lay.layers[l];
            let krow = a.kc + pos as u64 * nkv * dh * 2;
            let vrow = a.vc + pos as u64 * nkv * dh * 2;

            // ---- attention ----
            rmsnorm(&mut em, lay, lay.x, a.g1, &lw.norm1, h as u32);
            em.probe(format!("p{pos} l{l} norm1"));
            proj_blocked(&mut em, a.wq, a.mq, lw.mq.1, (nh * dh) as u32, h, lay.xn, |em| {
                em.emit(clamp16(st(lay.q, K2, 2)));
            });
            proj_blocked(&mut em, a.wk, a.mk, lw.mk.1, (nkv * dh) as u32, h, lay.xn, |em| {
                em.emit(clamp16(st(krow, K2, 2)));
            });
            proj_blocked(&mut em, a.wv, a.mv, lw.mv.1, (nkv * dh) as u32, h, lay.xn, |em| {
                em.emit(clamp16(st(vrow, K2, 2)));
            });
            em.probe(format!("p{pos} l{l} qkv"));
            // QK-norm + rotary, q heads then k heads (head index in K0).
            em.loop_over(K0, nh as u32, |em| {
                qknorm(em, lay, lay.q, a.gq, &lw.qnorm, dh as u32);
                rope(em, lay, lay.q, pos, dh as u32);
            });
            em.loop_over(K0, nkv as u32, |em| {
                qknorm(em, lay, krow, a.gk, &lw.knorm, dh as u32);
                rope(em, lay, krow, pos, dh as u32);
            });
            em.probe(format!("p{pos} l{l} qknorm+rope"));

            // ---- attention proper: kv-head (K0) × group member (K1) ----
            let t1 = (pos + 1) as u32;
            em.loop_over(K0, nkv as u32, |em| {
                em.loop_over(K1, group, |em| {
                    // scores → att32 (q head base: K0·512 + K1·256)
                    em.loop_over(K2, t1, |em| {
                        em.emit(ldc(0));
                        for ch in 0..(dh / 32) {
                            em.emit(dot16(
                                st2(lay.q + 64 * ch, K0, 512, K1, 256),
                                st2(a.kc + 64 * ch, K2, 2048, K0, 256),
                                32,
                            ));
                        }
                        em.emit(mul32(at(a.m_logit_c)));
                        em.emit(shift(SHIFT));
                        em.emit(st32_acc(st(lay.att32, K2, 4)));
                    });
                    // running max → neg_max
                    em.emit(ldc(i32::MIN));
                    em.loop_over(K2, t1, |em| {
                        em.emit(argmax(K2, st(lay.att32, K2, 4)));
                    });
                    em.emit(mul32(at(lay.c_neg1)));
                    em.emit(st32_acc(at(lay.neg_max)));
                    // exp(att − max) → e32
                    em.loop_over(K2, t1, |em| {
                        em.emit(ld32(st(lay.att32, K2, 4)));
                        em.emit(add32(at(lay.neg_max)));
                        em.emit(lut16(lay.lut_exp));
                        em.emit(st32_acc(st(lay.e32, K2, 4)));
                    });
                    // sum
                    em.emit(ldc(0));
                    em.loop_over(K2, t1, |em| em.emit(add32(st(lay.e32, K2, 4))));
                    em.emit(st32_acc(at(lay.sum)));
                    // probs (Q0.14 i16)
                    em.loop_over(K2, t1, |em| {
                        em.emit(ld32(st(lay.e32, K2, 4)));
                        em.emit(mul32(at(lay.c_2p14)));
                        em.emit(div32(at(lay.sum)));
                        em.emit(clamp16(st(lay.probs, K2, 2)));
                    });
                    // ctx[d] = rnd(Σ_j p_j·V[j,d], 14) → attnx
                    em.loop_over(K2, dh as u32, |em| {
                        em.emit(ldc(0));
                        em.loop_over(K3, t1, |em| {
                            em.emit(mac16(
                                st(lay.probs, K3, 2),
                                st3(a.vc, K3, 2048, K0, 256, K2, 2),
                            ));
                        });
                        em.emit(shift(14));
                        em.emit(clamp16(st3(lay.attnx, K0, 512, K1, 256, K2, 2)));
                    });
                });
            });

            em.probe(format!("p{pos} l{l} attn"));
            // ---- O-projection + residual (ST32 truncation) ----
            proj_blocked(&mut em, a.wo, a.mo, lw.mo.1, h as u32, nh * dh, lay.attnx, |em| {
                em.emit(add32(st(lay.x, K2, 4)));
                em.emit(st32_acc(st(lay.x, K2, 4)));
            });
            em.probe(format!("p{pos} l{l} o-proj"));

            // ---- FFN ----
            rmsnorm(&mut em, lay, lay.x, a.g2, &lw.norm2, h as u32);
            let blocks_h = h / 64;
            em.loop_over(K2, f as u32, |em| {
                // gate row → g16
                em.emit(ldc(0));
                for b in 0..blocks_h {
                    em.emit(dotbm(
                        st(a.w_gate + 64 * b, K2, h as u32),
                        at(lay.xn + 128 * b),
                        st(a.m_gate + 4 * b, K2, 4 * blocks_h as u32),
                    ));
                }
                em.emit(shift(lw.m_gate.1));
                em.emit(clamp16(at(lay.g16)));
                // up row → u16, up32
                em.emit(ldc(0));
                for b in 0..blocks_h {
                    em.emit(dotbm(
                        st(a.w_up + 64 * b, K2, h as u32),
                        at(lay.xn + 128 * b),
                        st(a.m_up + 4 * b, K2, 4 * blocks_h as u32),
                    ));
                }
                em.emit(shift(lw.m_up.1));
                em.emit(clamp16(at(lay.u16)));
                em.emit(ld16(at(lay.u16)));
                em.emit(st32_acc(at(lay.up32)));
                // x411 = rnd(g·m_sig, SHIFT); sign byte = clamp8(x411·2^30)
                em.emit(ld16(at(lay.g16)));
                em.emit(mul32(at(a.m_sig_c)));
                em.emit(shift(SHIFT));
                em.emit(st32_acc(at(lay.t_x)));
                em.emit(mul32(at(lay.c_2p30)));
                em.emit(clamp8(at(lay.t_sign)));
                // sign dispatch: byte 0x80 ⇔ x411 < 0. Equal-length arms.
                let jeq_at = em.prog.len();
                em.emit(Instr {
                    a: at(lay.t_sign),
                    imm: 0x80,
                    ..Instr::op(Opcode::Jeq)
                });
                // x ≥ 0: σ = 2^28 / (2^14 + exp(−x))
                let pos_start = em.prog.len();
                em.emit(ld32(at(lay.t_x)));
                em.emit(mul32(at(lay.c_neg1)));
                em.emit(lut16(lay.lut_exp));
                em.emit(add32(at(lay.c_2p14)));
                em.emit(st32_acc(at(lay.t_den)));
                em.emit(ldc(1 << 28));
                em.emit(div32(at(lay.t_den)));
                let jmp_at = em.prog.len();
                em.emit(Instr::op(Opcode::Jmp));
                let pos_len = em.prog.len() - pos_start;
                // x < 0: σ = exp(x)·2^14 / (2^14 + exp(x)) — steps NOT
                // double-counted: exactly one arm runs (lengths asserted).
                em.prog[jeq_at].target = em.pc();
                let neg_start = em.prog.len();
                let saved_mult = em.mult;
                em.mult = 0;
                em.emit(ld32(at(lay.t_x)));
                em.emit(lut16(lay.lut_exp));
                em.emit(st32_acc(at(lay.t_ep)));
                em.emit(add32(at(lay.c_2p14)));
                em.emit(st32_acc(at(lay.t_den)));
                em.emit(ld32(at(lay.t_ep)));
                em.emit(mul32(at(lay.c_2p14)));
                em.emit(div32(at(lay.t_den)));
                em.mult = saved_mult;
                assert_eq!(em.prog.len() - neg_start, pos_len); // equal-step arms
                em.prog[jmp_at].target = em.pc();
                em.emit(st32_acc(at(lay.t_sig)));
                // hpre = rnd(g·σ, 14); h_ffn[r] = sat16(rnd(hpre·u·m_h[r], SHIFT))
                em.emit(ld16(at(lay.g16)));
                em.emit(mul32(at(lay.t_sig)));
                em.emit(shift(14));
                em.emit(st32_acc(at(lay.silu32)));
                em.emit(mul32(at(lay.up32)));
                em.emit(mul32(st(a.m_h_arr, K2, 4)));
                em.emit(shift(SHIFT));
                em.emit(clamp16(st(lay.h_ffn, K2, 2)));
            });
            em.probe(format!("p{pos} l{l} ffn"));
            // down + residual
            proj_blocked(&mut em, a.w_down, a.m_down, lw.m_down.1, h as u32, f, lay.h_ffn, |em| {
                em.emit(add32(st(lay.x, K2, 4)));
                em.emit(st32_acc(st(lay.x, K2, 4)));
            });
            em.probe(format!("p{pos} l{l} down"));
        }

        // ---- streaming decode head at decision positions ----
        if pos >= n_prompt - 1 {
            rmsnorm(&mut em, lay, lay.x, lay.gf, &im.norm_f, h as u32);
            em.emit(ldc(i32::MIN));
            em.emit(st32_acc(at(lay.saved_max)));
            em.emit(ldc(0));
            em.emit(st32_acc(at(lay.v_cell)));
            let blocks = h / 64;
            let row_body = |em: &mut Em, w_base: u64, m_base: u64, k2_strided: bool| {
                em.emit(ldidx(K3, at(lay.v_cell)));
                em.emit(ldc(0));
                for b in 0..blocks {
                    let (wa, wm) = if k2_strided {
                        (
                            st2(w_base + 64 * b, K2, (CHUNK * h) as u32, K1, h as u32),
                            st2(m_base + 4 * b, K2, (CHUNK * blocks * 4) as u32, K1, 4 * blocks as u32),
                        )
                    } else {
                        (
                            st(w_base + 64 * b, K1, h as u32),
                            st(m_base + 4 * b, K1, 4 * blocks as u32),
                        )
                    };
                    em.emit(dotbm(wa, at(lay.xn + 128 * b), wm));
                }
                em.emit(shift(im.m_head.1));
                em.emit(st32_acc(st(lay.logit_buf, K1, 4)));
                em.emit(ld32(at(lay.saved_max)));
                em.emit(argmax_off(K3, st(lay.logit_buf, K1, 4)));
                em.emit(st32_acc(at(lay.saved_max)));
                em.emit(ldc(1));
                em.emit(add32(at(lay.v_cell)));
                em.emit(st32_acc(at(lay.v_cell)));
            };
            let full_chunks = (vocab / CHUNK) as u32;
            let tail = (vocab % CHUNK) as u32;
            em.loop_over(K2, full_chunks, |em| {
                em.loop_over(K1, CHUNK as u32, |em| {
                    row_body(em, lay.head_w, lay.m_head, true);
                });
            });
            if tail > 0 {
                let base_rows = full_chunks as u64 * CHUNK;
                em.loop_over(K1, tail, |em| {
                    row_body(
                        em,
                        lay.head_w + base_rows * h,
                        lay.m_head + base_rows * blocks * 4,
                        false,
                    );
                });
            }
            // decode feedback: aux holds the winning vocab row.
            em.emit(st32_aux(at(lay.tok)));
            em.probe(format!("p{pos} head"));
        }

        // ---- canonicalizing tail: acc ← 0, aux ← 0, idx ← 0 ----
        em.emit(ldc(i32::MIN));
        em.emit(argmax(K1, at(lay.c_zero))); // 0 > MIN always: acc ← 0, aux ← idx1 = 0
        em.emit(ldidx(K3, at(lay.c_zero)));

        token_boundaries.push(em.steps);
        boundary_pcs.push(em.pc());
    }
    em.emit(Instr::op(Opcode::Halt));

    let len = em.prog.len() as u64;
    let p = (0..=vm::MAX_PROG_DEPTH).find(|&p| len <= 1u64 << p).unwrap();
    QwenCompiled {
        program: em.prog,
        p,
        probes: em.probes,
        total_steps: em.steps,
        token_boundaries,
        boundary_pcs,
        n_prompt,
        n_gen,
    }
}
